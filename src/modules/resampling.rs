use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::direct::InterleavedSlice};

pub fn resample_fft(data: &[f64], orig_sr: u32, target_sr: u32) -> Vec<f64> {
    if data.is_empty() || orig_sr == target_sr {
        return data.to_vec();
    }

    let input =
        InterleavedSlice::new(data, 1, data.len()).expect("Invalid input buffer dimensions");

    // FFT resamplerの作成
    let mut resampler = Fft::<f64>::new(
        orig_sr as usize,
        target_sr as usize,
        4096,
        1,
        FixedSync::Both,
    )
    .expect("Failed to create FFT resampler");

    // リサンプリング実行
    let output = resampler
        .process_all(&input, data.len(), None)
        .expect("FFT resampling failed");

    // Vec<f64>を取り出す
    output.take_data()
}

pub fn resampling_n(data: Vec<f64>, sr: u32, sr_list: Vec<u32>) -> Vec<Vec<f64>> {
    let mut resampled_data_list = Vec::with_capacity(sr_list.len());
    for new_sr in sr_list {
        resampled_data_list.push(resample_fft(&data, sr, new_sr))
    }

    resampled_data_list
}
