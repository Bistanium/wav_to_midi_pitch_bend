use indicatif::{ProgressBar, ProgressStyle};

use super::audio_split::AudioSplit;
use super::fft::Fft;
use super::fft2midi::Fft2midi;
use super::math::pad;
use super::midi::Midi;
use super::resampling::resampling_n;
use super::settings::Settings;

pub fn midi_from_wav(data: Vec<f64>, sr: u32, output_name: String, settings: &Settings) {
    // オーディオ分割用の値
    let window_size_low = 128; // 時間分解能がhighの1/4
    let window_size_mid = 512; // 時間分解能がhighの1/2
    let window_size_high = 1024;
    let sr_low = 1280;
    let sr_mid = 10240;
    let sr_high = 40960;

    let pad_multiple = settings.pad_multiple;
    let pad_window_size_low = window_size_low * pad_multiple;
    let pad_window_size_mid = window_size_mid * pad_multiple;
    let pad_window_size_high = window_size_high * pad_multiple;
    let hop_low = window_size_low / 4;
    let hop_mid = window_size_mid / 4;
    let hop_high = window_size_high / 4;

    // midi定義
    let mut midi = Midi::new(settings);

    let resampled_data = {
        // 最後の方まで音があるとmidiで音が残ったままになるため余白を設ける
        let new_data = pad(data, sr as i32 / 2, sr as i32 / 2);

        // リサンプリング
        println!("\nResampling in progress");
        let sr_list = vec![sr_high, sr_mid, sr_low];
        resampling_n(new_data, sr, sr_list)
    };

    // 端数で各配列の長さがずれてもIndexErrorにならないように一番短いものに合わせる
    let segments_data_range = ((resampled_data[2].len() / window_size_low - 1) * 4 * 4)
        .min((resampled_data[1].len() / window_size_mid - 1) * 4 * 2)
        .min((resampled_data[0].len() / window_size_high - 1) * 4);

    // プログレスバー範囲
    let pb = ProgressBar::new(segments_data_range as u64);
    // 見た目
    pb.set_style(
        ProgressStyle::with_template(
            "[{percent}%] {bar:40} {pos}/{len} [{elapsed_precise}<{eta_precise}]",
        )
        .unwrap(),
    );

    let (f2m_low, f2m_mid, f2m_high, segments_data_range) = {
        // 異なる窓用にそれぞれ用意
        let mut split_low = AudioSplit::new(window_size_low, hop_low, pad_multiple);
        let mut split_mid = AudioSplit::new(window_size_mid, hop_mid, pad_multiple);
        let mut split_high = AudioSplit::new(window_size_high, hop_high, pad_multiple);
        let mut low_fft = Fft::new(pad_window_size_low);
        let mut mid_fft = Fft::new(pad_window_size_mid);
        let mut high_fft = Fft::new(pad_window_size_high);

        let start_note_low = settings.start_note.clamp(12, 60) as usize;
        let end_note_low = settings.end_note.clamp(12, 60) as usize;
        let start_note_mid = settings.start_note.clamp(60, 108) as usize;
        let end_note_mid = settings.end_note.clamp(60, 108) as usize;
        let start_note_high = settings.start_note.clamp(108, 128) as usize;
        let end_note_high = settings.end_note.clamp(108, 128) as usize;

        let mut f2m_low = Fft2midi::new(
            pad_window_size_low,
            hop_low,
            sr_low,
            segments_data_range / 4,
            start_note_low,
            end_note_low,
            settings,
        );
        let mut f2m_mid = Fft2midi::new(
            pad_window_size_mid,
            hop_mid,
            sr_mid,
            segments_data_range / 2,
            start_note_mid,
            end_note_mid,
            settings,
        );
        let mut f2m_high = Fft2midi::new(
            pad_window_size_high,
            hop_high,
            sr_high,
            segments_data_range,
            start_note_high,
            end_note_high,
            settings,
        );

        println!("Converting to MIDI");
        for i in 0..segments_data_range {
            pb.inc(1);
            if i % 4 == 0 {
                let temp_seg = split_low.audio_split(&resampled_data[2]);
                let ffted_data = low_fft.fft(&temp_seg);
                f2m_low.convert(ffted_data);
            }

            if i % 2 == 0 {
                let temp_seg = split_mid.audio_split(&resampled_data[1]);
                let ffted_data = mid_fft.fft(&temp_seg);
                f2m_mid.convert(ffted_data);
            }

            let temp_seg = split_high.audio_split(&resampled_data[0]);
            let ffted_data = high_fft.fft(&temp_seg);
            f2m_high.convert(ffted_data);
        }
        pb.finish_with_message("Conversion to MIDI complete");

        (f2m_low, f2m_mid, f2m_high, segments_data_range)
    };

    midi.normalize_velocity(f2m_low, f2m_mid, f2m_high, segments_data_range);

    midi.export(output_name);
}
