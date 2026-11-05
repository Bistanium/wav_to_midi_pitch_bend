use indicatif::{ProgressBar, ProgressStyle};

use super::audio_split::AudioSplit;
use super::fft::Fft;
use super::fft2midi::Fft2midi;
use super::math::pad;
use super::midi::Midi;
use super::resampling::resampling_n;
use super::settings::Settings;

pub fn midi_from_wav(data: &[f64], sr: u32, output_name: String, settings: &Settings) {
    // オーディオ分割用の値
    let window_size_low = 128; // 時間分解能がhighの1/4
    let window_size_mid = 512; // 時間分解能がhighの1/2
    let window_size_high = 1024;
    let sr_low = 1280;
    let sr_mid = 10240;
    let sr_high = 40960;

    let pad_multiple = settings.pad_multiple;
    let window_size_low_pad = window_size_low * pad_multiple;
    let window_size_mid_pad = window_size_mid * pad_multiple;
    let window_size_high_pad = window_size_high * pad_multiple;
    let low_hop = window_size_low / 4;
    let mid_hop = window_size_mid / 4;
    let high_hop = window_size_high / 4;

    // midi定義
    let mut midi = Midi::new(settings);

    let (low_f2m, mid_f2m, high_f2m, segments_data_range) = {
        // 異なる窓用にそれぞれ用意
        let low_split = AudioSplit::new(window_size_low, low_hop, pad_multiple);
        let mid_split = AudioSplit::new(window_size_mid, mid_hop, pad_multiple);
        let high_split = AudioSplit::new(window_size_high, high_hop, pad_multiple);
        let mut low_fft = Fft::new(window_size_low_pad);
        let mut mid_fft = Fft::new(window_size_mid_pad);
        let mut high_fft = Fft::new(window_size_high_pad);

        // 最後の方まで音があるとmidiで音が残ったままになるため1秒の余白を設ける
        let new_data = pad(data, sr as i32 / 2, sr as i32 / 2);

        // リサンプリング
        println!("Resampling in progress");
        let sr_list = vec![sr_high, sr_mid, sr_low];
        let resampled_data = resampling_n(new_data, sr, sr_list);

        // 端数で各配列の長さがずれてもIndexErrorにならないように一番短いものに合わせる
        let segments_data_range = ((resampled_data[2].len() / window_size_low - 1) * 4 * 4)
            .min((resampled_data[1].len() / window_size_mid - 1) * 4 * 2)
            .min((resampled_data[0].len() / window_size_high - 1) * 4);

        // midi化下準備
        let mut low_f2m = Fft2midi::new(
            window_size_low_pad,
            low_hop,
            sr_low,
            segments_data_range / 4,
            36,
            60,
            settings,
        );
        let mut mid_f2m = Fft2midi::new(
            window_size_mid_pad,
            mid_hop,
            sr_mid,
            segments_data_range / 2,
            60,
            108,
            settings,
        );
        let mut high_f2m = Fft2midi::new(
            window_size_high_pad,
            high_hop,
            sr_high,
            segments_data_range,
            108,
            128,
            settings,
        );

        // プログレスバー範囲
        let pb = ProgressBar::new(segments_data_range as u64 - 4);
        // 見た目
        pb.set_style(
            ProgressStyle::with_template(
                "[{percent}%] {bar:40} {pos}/{len} [{elapsed_precise}<{eta_precise}]",
            )
            .unwrap(),
        );

        println!("Converting to MIDI");
        // すべての窓で一回音を入れるために4から始める
        for i in 4..segments_data_range {
            pb.inc(1);
            if i % 4 == 0 {
                let j = i / 4;
                let temp_seg = low_split.audio_split(&resampled_data[2], j);
                let ffted_data = low_fft.fft(&temp_seg);
                low_f2m.convert(ffted_data, j);
            }

            if i % 2 == 0 {
                let j = i / 2;
                let temp_seg = mid_split.audio_split(&resampled_data[1], j);
                let ffted_data = mid_fft.fft(&temp_seg);
                mid_f2m.convert(ffted_data, j);
            }

            let temp_seg = high_split.audio_split(&resampled_data[0], i);
            let ffted_data = high_fft.fft(&temp_seg);
            high_f2m.convert(ffted_data, i);
        }
        pb.finish_with_message("Conversion to MIDI complete");

        (low_f2m, mid_f2m, high_f2m, segments_data_range)
    };

    midi.normalize_velocity(low_f2m, mid_f2m, high_f2m, segments_data_range, settings);

    midi.export(output_name);
}
