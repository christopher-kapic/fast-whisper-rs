use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3};
use std::collections::{BTreeSet, HashMap};

use super::{PldaModel, PldaTransform};

/// Configuration for the clustering stage.
pub struct ClusteringConfig {
    /// Distance threshold for stopping AHC (default: 0.6).
    pub threshold: f64,
    /// Minimum activity ratio for a speaker to be considered active (default: 0.2).
    pub min_activity_ratio: f32,
    /// Optional exact number of speakers.
    pub num_speakers: Option<usize>,
    /// Optional minimum number of speakers.
    pub min_speakers: Option<usize>,
    /// Optional maximum number of speakers.
    pub max_speakers: Option<usize>,
    /// LDA dimension reduction target (default: 128).
    pub lda_dim: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            min_activity_ratio: 0.2,
            num_speakers: None,
            min_speakers: None,
            max_speakers: None,
            lda_dim: 128,
        }
    }
}

/// Result of the clustering stage.
pub struct ClusteringOutput {
    /// Mapping from (chunk_index, speaker_index) to global speaker ID.
    pub assignments: HashMap<(usize, usize), usize>,
    /// Number of unique speakers found.
    pub num_speakers: usize,
}

/// L2-normalize each row of a matrix.
pub fn l2_normalize_rows(m: &Array2<f32>) -> Array2<f32> {
    let mut result = m.clone();
    for mut row in result.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            row.mapv_inplace(|x| x / norm);
        }
    }
    result
}

/// Compute cosine distance between two vectors: 1 - cosine_similarity.
pub fn cosine_distance(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
    let dot: f32 = a.dot(b);
    let norm_a: f32 = a.dot(a).sqrt();
    let norm_b: f32 = b.dot(b).sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b)) as f64
}

/// Filter embeddings: remove NaN and speakers with activity below threshold.
///
/// Returns (filtered_embeddings of shape (N, D), corresponding (chunk_idx, speaker_idx) pairs).
pub fn filter_embeddings(
    embeddings: &Array3<f32>,
    chunk_binarized: &Array3<f32>,
    min_activity_ratio: f32,
) -> (Array2<f32>, Vec<(usize, usize)>) {
    let num_chunks = embeddings.shape()[0];
    let num_speakers = embeddings.shape()[1];
    let embedding_dim = embeddings.shape()[2];
    let num_frames = chunk_binarized.shape()[1];

    let mut filtered: Vec<Vec<f32>> = Vec::new();
    let mut indices = Vec::new();

    for c in 0..num_chunks {
        for s in 0..num_speakers {
            // Check for NaN
            let has_nan = (0..embedding_dim).any(|d| embeddings[[c, s, d]].is_nan());
            if has_nan {
                continue;
            }

            // Check activity ratio
            let active_frames: usize = (0..num_frames)
                .filter(|&f| chunk_binarized[[c, f, s]] > 0.5)
                .count();
            let ratio = active_frames as f32 / num_frames as f32;
            if ratio < min_activity_ratio {
                continue;
            }

            let emb: Vec<f32> = (0..embedding_dim).map(|d| embeddings[[c, s, d]]).collect();
            filtered.push(emb);
            indices.push((c, s));
        }
    }

    if filtered.is_empty() {
        return (Array2::zeros((0, embedding_dim)), indices);
    }

    let n = filtered.len();
    let flat: Vec<f32> = filtered.into_iter().flatten().collect();
    (
        Array2::from_shape_vec((n, embedding_dim), flat).unwrap(),
        indices,
    )
}

/// Apply PLDA transform to embeddings:
/// 1. Center by subtracting mean
/// 2. Apply LDA transform matrix
/// 3. Reduce to lda_dim dimensions
/// 4. L2 normalize
pub fn apply_plda_transform(
    embeddings: &Array2<f32>,
    transform: &PldaTransform,
    lda_dim: usize,
) -> Array2<f32> {
    if embeddings.nrows() == 0 {
        return Array2::zeros((0, lda_dim));
    }

    let d = embeddings.ncols();

    // Extract mean as 1D vector
    let mean: Array1<f32> = if transform.mean.shape()[0] == 1 {
        transform.mean.row(0).to_owned()
    } else {
        Array1::from_iter(transform.mean.iter().copied())
    };

    // Center: subtract mean
    let mut centered = embeddings.clone();
    for mut row in centered.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            if i < mean.len() {
                *val -= mean[i];
            }
        }
    }

    // Apply LDA transform matrix
    let tf = &transform.transform;
    let (tf_rows, tf_cols) = (tf.shape()[0], tf.shape()[1]);

    let transformed = if tf_cols == d && tf_rows >= lda_dim {
        // transform is (output_dim, input_dim): result = centered @ transform.T
        centered.dot(&tf.t())
    } else if tf_rows == d {
        // transform is (input_dim, output_dim): result = centered @ transform
        centered.dot(tf)
    } else {
        // Dimension mismatch — fallback to centered embeddings
        centered
    };

    // Truncate to lda_dim if needed
    let actual_dim = transformed.ncols().min(lda_dim);
    let truncated = if transformed.ncols() > actual_dim {
        transformed.slice(s![.., ..actual_dim]).to_owned()
    } else {
        transformed
    };

    l2_normalize_rows(&truncated)
}

/// Compute pairwise cosine distance matrix (N x N).
pub fn pairwise_cosine_distance(embeddings: &Array2<f32>) -> Array2<f64> {
    let n = embeddings.nrows();
    let mut distances = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let d = cosine_distance(&embeddings.row(i).to_owned(), &embeddings.row(j).to_owned());
            distances[[i, j]] = d;
            distances[[j, i]] = d;
        }
    }

    distances
}

/// Agglomerative hierarchical clustering with average linkage and cosine distance.
///
/// Returns cluster assignment for each input embedding (0-indexed contiguous IDs).
pub fn agglomerative_clustering(
    distances: &Array2<f64>,
    threshold: f64,
    num_speakers: Option<usize>,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> Vec<usize> {
    let n = distances.nrows();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    // Each element starts in its own cluster
    let mut cluster_assignments: Vec<usize> = (0..n).collect();
    let mut num_clusters = n;

    // Track cluster members for average linkage computation
    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        cluster_members.insert(i, vec![i]);
    }

    // Working distance matrix between active clusters
    let mut dist = distances.clone();
    for i in 0..n {
        dist[[i, i]] = f64::INFINITY;
    }

    let mut active: Vec<bool> = vec![true; n];

    loop {
        // Determine effective speaker count bounds
        let target = num_speakers;
        let min_bound = target.unwrap_or(min_speakers.unwrap_or(1));

        // Stop if we've reached the target or minimum
        if num_clusters <= min_bound {
            break;
        }

        // Find the closest pair of active clusters
        let mut min_dist = f64::INFINITY;
        let mut merge_i = 0;
        let mut merge_j = 0;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if dist[[i, j]] < min_dist {
                    min_dist = dist[[i, j]];
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        // Check threshold-based stopping (unless constrained by speaker count)
        if target.is_none() && min_dist > threshold {
            if let Some(max) = max_speakers {
                if num_clusters <= max {
                    break;
                }
                // Must keep merging to satisfy max_speakers
            } else {
                break;
            }
        }

        // Merge cluster merge_j into merge_i
        let members_j = cluster_members.remove(&merge_j).unwrap_or_default();
        cluster_members
            .get_mut(&merge_i)
            .unwrap()
            .extend(members_j);

        // Update assignments for all members of the merged cluster
        for &idx in &cluster_members[&merge_i] {
            cluster_assignments[idx] = merge_i;
        }

        // Update working distance matrix using average linkage
        for k in 0..n {
            if !active[k] || k == merge_i || k == merge_j {
                continue;
            }

            let members_i = &cluster_members[&merge_i];
            let members_k = &cluster_members[&k];

            let mut total = 0.0f64;
            let mut count = 0usize;
            for &mi in members_i.iter() {
                for &mk in members_k.iter() {
                    total += distances[[mi, mk]]; // Use original pairwise distances
                    count += 1;
                }
            }
            let avg = if count > 0 {
                total / count as f64
            } else {
                f64::INFINITY
            };

            dist[[merge_i, k]] = avg;
            dist[[k, merge_i]] = avg;
        }

        // Deactivate merge_j
        active[merge_j] = false;
        for k in 0..n {
            dist[[merge_j, k]] = f64::INFINITY;
            dist[[k, merge_j]] = f64::INFINITY;
        }

        num_clusters -= 1;
    }

    // Renumber clusters to contiguous 0-based IDs
    let unique_clusters: Vec<usize> = {
        let mut seen = BTreeSet::new();
        for &c in &cluster_assignments {
            seen.insert(c);
        }
        seen.into_iter().collect()
    };

    let cluster_map: HashMap<usize, usize> = unique_clusters
        .iter()
        .enumerate()
        .map(|(new_id, &old_id)| (old_id, new_id))
        .collect();

    cluster_assignments
        .iter()
        .map(|&c| cluster_map[&c])
        .collect()
}

/// Run the full clustering pipeline on speaker embeddings.
///
/// Steps: filter → L2 normalize → PLDA transform → pairwise distances → AHC.
pub fn cluster_embeddings(
    embeddings: &Array3<f32>,
    chunk_binarized: &Array3<f32>,
    plda_transform_params: &PldaTransform,
    _plda_model: &PldaModel, // Reserved for future VBx refinement
    config: &ClusteringConfig,
) -> Result<ClusteringOutput> {
    // Step 1: Filter out NaN and inactive speakers
    let (filtered, indices) =
        filter_embeddings(embeddings, chunk_binarized, config.min_activity_ratio);

    if filtered.nrows() == 0 {
        return Ok(ClusteringOutput {
            assignments: HashMap::new(),
            num_speakers: 0,
        });
    }

    // Step 2: L2 normalize raw embeddings
    let normalized = l2_normalize_rows(&filtered);

    // Step 3: PLDA transform (center + LDA + L2 norm)
    let transformed = apply_plda_transform(&normalized, plda_transform_params, config.lda_dim);

    // Step 4: Compute pairwise cosine distances
    let distances = pairwise_cosine_distance(&transformed);

    // Step 5: Agglomerative hierarchical clustering
    let cluster_ids = agglomerative_clustering(
        &distances,
        config.threshold,
        config.num_speakers,
        config.min_speakers,
        config.max_speakers,
    );

    // Step 6: Build (chunk_idx, speaker_idx) → global_speaker_id map
    let mut assignments = HashMap::new();
    let mut max_id = 0usize;
    for (i, &(chunk_idx, speaker_idx)) in indices.iter().enumerate() {
        let id = cluster_ids[i];
        assignments.insert((chunk_idx, speaker_idx), id);
        if id > max_id {
            max_id = id;
        }
    }

    let num_speakers = if assignments.is_empty() {
        0
    } else {
        max_id + 1
    };

    Ok(ClusteringOutput {
        assignments,
        num_speakers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize_rows() {
        let m = Array2::from_shape_vec((2, 3), vec![3.0, 0.0, 4.0, 0.0, 5.0, 0.0]).unwrap();
        let normed = l2_normalize_rows(&m);

        // Row 0: [3,0,4] / 5 = [0.6, 0.0, 0.8]
        assert!((normed[[0, 0]] - 0.6).abs() < 1e-5);
        assert!((normed[[0, 2]] - 0.8).abs() < 1e-5);

        // Row 1: [0,5,0] / 5 = [0.0, 1.0, 0.0]
        assert!((normed[[1, 1]] - 1.0).abs() < 1e-5);

        // Each row should have unit norm
        for row in normed.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Row norm should be 1.0, got {norm}");
        }
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let m = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let normed = l2_normalize_rows(&m);
        // Zero vector stays zero
        for val in normed.iter() {
            assert!(*val == 0.0);
        }
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let d = cosine_distance(&a, &a);
        assert!(d.abs() < 1e-6, "Distance between identical vectors should be ~0");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "Distance between orthogonal vectors should be 1.0"
        );
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0]);
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 2.0).abs() < 1e-6,
            "Distance between opposite vectors should be 2.0"
        );
    }

    #[test]
    fn test_pairwise_cosine_distance() {
        let emb = Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, 0.0, // vec 0
                0.0, 1.0, // vec 1
                1.0, 0.0, // vec 2 (same as 0)
            ],
        )
        .unwrap();

        let dist = pairwise_cosine_distance(&emb);

        // dist[0,2] and dist[2,0] should be ~0 (identical)
        assert!(dist[[0, 2]].abs() < 1e-6);
        assert!(dist[[2, 0]].abs() < 1e-6);

        // dist[0,1] should be 1.0 (orthogonal)
        assert!((dist[[0, 1]] - 1.0).abs() < 1e-6);

        // Diagonal should be 0
        assert!(dist[[0, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_ahc_two_clusters() {
        // 4 embeddings: 0,1 are similar; 2,3 are similar; groups are distant
        let emb = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 0.1, // cluster A
                1.0, 0.2, // cluster A
                -1.0, 0.1, // cluster B
                -1.0, 0.2, // cluster B
            ],
        )
        .unwrap();

        let normed = l2_normalize_rows(&emb);
        let dist = pairwise_cosine_distance(&normed);
        let clusters = agglomerative_clustering(&dist, 0.5, None, None, None);

        assert_eq!(clusters.len(), 4);
        // 0 and 1 should be in same cluster
        assert_eq!(clusters[0], clusters[1]);
        // 2 and 3 should be in same cluster
        assert_eq!(clusters[2], clusters[3]);
        // The two groups should be different
        assert_ne!(clusters[0], clusters[2]);
    }

    #[test]
    fn test_ahc_single_cluster_high_threshold() {
        // With a very high threshold, everything merges into one cluster
        let emb = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
        )
        .unwrap();
        let dist = pairwise_cosine_distance(&emb);
        let clusters = agglomerative_clustering(&dist, 3.0, None, None, None);

        assert_eq!(clusters[0], clusters[1]);
        assert_eq!(clusters[1], clusters[2]);
    }

    #[test]
    fn test_ahc_num_speakers_constraint() {
        // Force exactly 2 speakers
        let emb = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9],
        )
        .unwrap();
        let normed = l2_normalize_rows(&emb);
        let dist = pairwise_cosine_distance(&normed);
        let clusters = agglomerative_clustering(&dist, 0.5, Some(2), None, None);

        let unique: BTreeSet<usize> = clusters.iter().copied().collect();
        assert_eq!(unique.len(), 2, "Should have exactly 2 clusters");
    }

    #[test]
    fn test_ahc_max_speakers_constraint() {
        // 4 distinct embeddings but max 2 speakers
        let emb = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        )
        .unwrap();
        let dist = pairwise_cosine_distance(&emb);
        // Very low threshold so nothing would merge normally
        let clusters = agglomerative_clustering(&dist, 0.01, None, None, Some(2));

        let unique: BTreeSet<usize> = clusters.iter().copied().collect();
        assert!(
            unique.len() <= 2,
            "Should have at most 2 clusters, got {}",
            unique.len()
        );
    }

    #[test]
    fn test_ahc_empty_input() {
        let dist = Array2::<f64>::zeros((0, 0));
        let clusters = agglomerative_clustering(&dist, 0.5, None, None, None);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_ahc_single_element() {
        let dist = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let clusters = agglomerative_clustering(&dist, 0.5, None, None, None);
        assert_eq!(clusters, vec![0]);
    }

    #[test]
    fn test_filter_embeddings_removes_nan() {
        let embeddings = Array3::from_shape_vec(
            (2, 1, 2),
            vec![
                1.0, 2.0, // chunk 0, speaker 0: valid
                f32::NAN, 1.0, // chunk 1, speaker 0: NaN
            ],
        )
        .unwrap();

        let binarized = Array3::from_shape_vec(
            (2, 4, 1),
            vec![
                1.0, 1.0, 1.0, 1.0, // chunk 0: fully active
                1.0, 1.0, 1.0, 1.0, // chunk 1: fully active
            ],
        )
        .unwrap();

        let (filtered, indices) = filter_embeddings(&embeddings, &binarized, 0.2);
        assert_eq!(filtered.nrows(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], (0, 0));
    }

    #[test]
    fn test_filter_embeddings_activity_threshold() {
        let embeddings = Array3::from_shape_vec(
            (1, 2, 2),
            vec![
                1.0, 2.0, // speaker 0
                3.0, 4.0, // speaker 1
            ],
        )
        .unwrap();

        let binarized = Array3::from_shape_vec(
            (1, 10, 2),
            vec![
                // speaker 0: 1 active frame (10%) — below 20% threshold
                // speaker 1: 5 active frames (50%) — above threshold
                1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();

        let (filtered, indices) = filter_embeddings(&embeddings, &binarized, 0.2);
        assert_eq!(filtered.nrows(), 1);
        assert_eq!(indices[0], (0, 1)); // Only speaker 1 passes
    }

    #[test]
    fn test_plda_transform_shapes() {
        // 3 embeddings of dim 4, LDA reduces to 2
        let embeddings = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let transform = PldaTransform {
            mean: Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
            transform: Array2::from_shape_vec(
                (2, 4),
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            .unwrap(),
        };

        let result = apply_plda_transform(&embeddings, &transform, 2);
        assert_eq!(result.shape(), &[3, 2]);

        // Each row should be L2-normalized
        for row in result.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "Expected unit norm, got {norm}"
            );
        }
    }

    #[test]
    fn test_plda_transform_empty() {
        let embeddings = Array2::<f32>::zeros((0, 4));
        let transform = PldaTransform {
            mean: Array2::from_shape_vec((1, 4), vec![0.0; 4]).unwrap(),
            transform: Array2::from_shape_vec((2, 4), vec![0.0; 8]).unwrap(),
        };
        let result = apply_plda_transform(&embeddings, &transform, 2);
        assert_eq!(result.shape(), &[0, 2]);
    }

    #[test]
    fn test_cluster_embeddings_full_pipeline() {
        // Two chunks, 2 speakers each, embeddings designed to form 2 clusters
        let embeddings = Array3::from_shape_vec(
            (2, 2, 4),
            vec![
                // chunk 0, speaker 0: cluster A direction
                1.0, 0.1, 0.0, 0.0,
                // chunk 0, speaker 1: NaN (inactive)
                f32::NAN, f32::NAN, f32::NAN, f32::NAN,
                // chunk 1, speaker 0: cluster A direction (similar to chunk 0 spk 0)
                0.9, 0.2, 0.0, 0.0,
                // chunk 1, speaker 1: cluster B direction (different)
                0.0, 0.0, 1.0, 0.1,
            ],
        )
        .unwrap();

        let binarized = Array3::from_shape_vec(
            (2, 4, 2),
            vec![
                // chunk 0: spk 0 active, spk 1 inactive
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                // chunk 1: both active
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();

        let plda_tf = PldaTransform {
            mean: Array2::from_shape_vec((1, 4), vec![0.0; 4]).unwrap(),
            // Identity-like transform (4x4, lda_dim will truncate)
            transform: Array2::from_shape_vec(
                (4, 4),
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0,
                ],
            )
            .unwrap(),
        };

        let plda_model = PldaModel {
            phi: Array2::zeros((4, 4)),
            sigma: Array2::zeros((4, 4)),
        };

        let config = ClusteringConfig {
            lda_dim: 4,
            threshold: 0.5,
            ..Default::default()
        };

        let result = cluster_embeddings(&embeddings, &binarized, &plda_tf, &plda_model, &config)
            .unwrap();

        // Should have 3 valid embeddings (chunk 0 spk 0, chunk 1 spk 0, chunk 1 spk 1)
        assert_eq!(result.assignments.len(), 3);

        // chunk 0 spk 0 and chunk 1 spk 0 should be same cluster (similar directions)
        let c0s0 = result.assignments[&(0, 0)];
        let c1s0 = result.assignments[&(1, 0)];
        let c1s1 = result.assignments[&(1, 1)];

        assert_eq!(c0s0, c1s0, "Similar embeddings should cluster together");
        assert_ne!(c0s0, c1s1, "Different embeddings should be in different clusters");
        assert_eq!(result.num_speakers, 2);
    }

    #[test]
    fn test_cluster_embeddings_empty() {
        let embeddings = Array3::from_shape_vec(
            (1, 1, 4),
            vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN],
        )
        .unwrap();

        let binarized = Array3::from_shape_vec((1, 4, 1), vec![0.0; 4]).unwrap();

        let plda_tf = PldaTransform {
            mean: Array2::zeros((1, 4)),
            transform: Array2::eye(4),
        };
        let plda_model = PldaModel {
            phi: Array2::zeros((4, 4)),
            sigma: Array2::zeros((4, 4)),
        };
        let config = ClusteringConfig::default();

        let result =
            cluster_embeddings(&embeddings, &binarized, &plda_tf, &plda_model, &config).unwrap();
        assert!(result.assignments.is_empty());
        assert_eq!(result.num_speakers, 0);
    }
}
