use hound::{SampleFormat, WavReader};
use std::path::Path;

type WavData = (Vec<f64>, Option<Vec<f64>>, u32);
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<WavData, hound::Error> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    // それぞれの形式で処理をわける
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f64 / i16::MAX as f64))
            .collect::<Result<Vec<f64>, _>>()?,
        (SampleFormat::Int, 24) => {
            const I24MAX: i32 = 8388607;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f64 / I24MAX as f64))
                .collect::<Result<Vec<f64>, _>>()?
        }
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f64 / i32::MAX as f64))
            .collect::<Result<Vec<f64>, _>>()?,
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.map(|v| v as f64))
            .collect::<Result<Vec<f64>, _>>()?,
        _ => return Err(hound::Error::FormatError("Unsupported WAV format")),
    };

    match spec.channels {
        1 => {
            let mut left = Vec::new();
            for s in samples {
                left.push(s);
            }
            Ok((left, None, spec.sample_rate))
        }
        2 => {
            let mut left = Vec::new();
            let mut right = Vec::new();

            for (i, s) in samples.into_iter().enumerate() {
                if i % 2 == 0 {
                    left.push(s);
                } else {
                    right.push(s);
                }
            }
            Ok((left, Some(right), spec.sample_rate))
        }
        _ => Err(hound::Error::FormatError(
            "Only mono or stereo WAV files are supported",
        )),
    }
}
