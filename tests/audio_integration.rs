use std::io::Write;

/// Generate a minimal WAV file with a sine wave.
///
/// Parameters:
/// - `sample_rate`: samples per second (e.g. 16000)
/// - `channels`: number of channels (1 = mono, 2 = stereo)
/// - `duration_secs`: duration in seconds
/// - `freq_hz`: sine wave frequency in Hz
///
/// Returns a Vec<u8> containing a valid WAV file.
fn generate_wav(sample_rate: u32, channels: u16, duration_secs: f64, freq_hz: f64) -> Vec<u8> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = num_samples * channels as usize * (bits_per_sample as usize / 8);

    let mut buf: Vec<u8> = Vec::new();

    // RIFF header
    buf.write_all(b"RIFF").unwrap();
    buf.write_all(&((36 + data_size) as u32).to_le_bytes())
        .unwrap();
    buf.write_all(b"WAVE").unwrap();

    // fmt sub-chunk
    buf.write_all(b"fmt ").unwrap();
    buf.write_all(&16u32.to_le_bytes()).unwrap(); // sub-chunk size
    buf.write_all(&1u16.to_le_bytes()).unwrap(); // PCM format
    buf.write_all(&channels.to_le_bytes()).unwrap();
    buf.write_all(&sample_rate.to_le_bytes()).unwrap();
    buf.write_all(&byte_rate.to_le_bytes()).unwrap();
    buf.write_all(&block_align.to_le_bytes()).unwrap();
    buf.write_all(&bits_per_sample.to_le_bytes()).unwrap();

    // data sub-chunk
    buf.write_all(b"data").unwrap();
    buf.write_all(&(data_size as u32).to_le_bytes()).unwrap();

    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (t * freq_hz * 2.0 * std::f64::consts::PI).sin();
        let sample_i16 = (sample * 32000.0) as i16;
        for _ in 0..channels {
            buf.write_all(&sample_i16.to_le_bytes()).unwrap();
        }
    }

    buf
}

#[test]
fn test_load_mono_16khz_wav() {
    // Generate a 1-second mono 16kHz WAV (440Hz sine wave)
    let wav_data = generate_wav(16000, 1, 1.0, 440.0);
    let tmp = std::env::temp_dir().join("fast-whisper-rs-test-mono16k.wav");
    std::fs::write(&tmp, &wav_data).unwrap();

    // Use the binary's audio module indirectly — we test by running the CLI
    // which will attempt to load the audio (and fail at model loading, not audio)
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fast-whisper-rs"))
        .args(["--file-name", tmp.to_str().unwrap()])
        .output()
        .unwrap();

    // Should fail at model loading (exit 1), NOT at audio loading
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should NOT contain audio loading errors
    assert!(
        !stderr.contains("Failed to open audio file"),
        "Audio loading should succeed, got: {stderr}"
    );
    assert!(
        !stderr.contains("Failed to probe audio format"),
        "Audio probe should succeed, got: {stderr}"
    );
    assert!(
        !stderr.contains("No audio track found"),
        "Audio track should be found, got: {stderr}"
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_load_stereo_44100_wav() {
    // Generate a 0.5-second stereo 44100Hz WAV — tests resampling + mono conversion
    let wav_data = generate_wav(44100, 2, 0.5, 440.0);
    let tmp = std::env::temp_dir().join("fast-whisper-rs-test-stereo44k.wav");
    std::fs::write(&tmp, &wav_data).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fast-whisper-rs"))
        .args(["--file-name", tmp.to_str().unwrap()])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Audio loading should succeed (fail at model, not audio)
    assert!(
        !stderr.contains("Failed to probe audio format"),
        "Stereo 44.1kHz WAV should be loadable, got: {stderr}"
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_generate_wav_correct_format() {
    let wav = generate_wav(16000, 1, 0.1, 440.0);

    // Check RIFF header
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");

    // Check PCM format tag
    let format = u16::from_le_bytes([wav[20], wav[21]]);
    assert_eq!(format, 1); // PCM

    // Check sample rate
    let sr = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
    assert_eq!(sr, 16000);

    // Check channels
    let ch = u16::from_le_bytes([wav[22], wav[23]]);
    assert_eq!(ch, 1);

    // Check data chunk exists
    assert_eq!(&wav[36..40], b"data");

    // Check data size: 0.1s * 16000 * 1ch * 2 bytes = 3200
    let data_size = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
    assert_eq!(data_size, 3200);
}
