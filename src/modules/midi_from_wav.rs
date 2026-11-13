use indicatif::{ProgressBar, ProgressStyle};

use super::audio_split::AudioSplit;
use super::fft::Fft;
use super::fft2midi::Fft2midi;
use super::math::pad;
use super::midi::Midi;
use super::resampling::resampling_n;
use super::settings::Settings;

pub fn midi_from_wav(data: Vec<f64>, sr: u32, output_name: String, settings: &Settings) {
    // high: 0.25s, mid: 0.5s, low: 1s
    let window_size = vec![128, 512, 1024];
    let sr_vec = vec![1280, 10240, 40960];

    let pad_multiple = settings.pad_multiple;
    let pad_window_size = vec![
        window_size[0] * pad_multiple,
        window_size[1] * pad_multiple,
        window_size[2] * pad_multiple,
    ];
    let hop_size = vec![window_size[0] / 4, window_size[1] / 4, window_size[2] / 4];

    // midi定義
    let mut midi = Midi::new(settings);

    let resampled_data = {
        // 最後の方まで音があるとmidiで音が残ったままになるため余白を設ける
        let amount = sr as i32 / 2;
        let new_data = pad(data, amount, amount);

        // リサンプリング
        println!("\nResampling in progress");
        resampling_n(new_data, sr, &sr_vec)
    };

    // 端数で各配列の長さがずれてもIndexErrorにならないように一番短いものに合わせる
    // 共通の*4はオーバーラップによるもの
    let segments_data_range = ((resampled_data[0].len() / window_size[0] - 1) * 4 * 4)
        .min((resampled_data[1].len() / window_size[1] - 1) * 4 * 2)
        .min((resampled_data[2].len() / window_size[2] - 1) * 4);

    // プログレスバー範囲
    let pb = ProgressBar::new(segments_data_range as u64);
    // 見た目
    pb.set_style(
        ProgressStyle::with_template(
            "[{percent}%] {bar:40} {pos}/{len} [{elapsed_precise}<{eta_precise}]",
        )
        .unwrap(),
    );

    let fft2midi = {
        // 異なる窓用にそれぞれ用意
        let mut split = vec![
            AudioSplit::new(window_size[0], hop_size[0], pad_multiple),
            AudioSplit::new(window_size[1], hop_size[1], pad_multiple),
            AudioSplit::new(window_size[2], hop_size[2], pad_multiple),
        ];

        let mut fft = vec![
            Fft::new(pad_window_size[0]),
            Fft::new(pad_window_size[1]),
            Fft::new(pad_window_size[2]),
        ];

        // 境界値に収める
        let start_note_low = settings.start_note.clamp(12, 60) as usize;
        let end_note_low = settings.end_note.clamp(12, 60) as usize;
        let start_note_mid = settings.start_note.clamp(60, 108) as usize;
        let end_note_mid = settings.end_note.clamp(60, 108) as usize;
        let start_note_high = settings.start_note.clamp(108, 128) as usize;
        let end_note_high = settings.end_note.clamp(108, 128) as usize;

        let mut fft2midi = vec![
            Fft2midi::new(
                pad_window_size[0],
                hop_size[0],
                sr_vec[0],
                segments_data_range / 4,
                start_note_low,
                end_note_low,
                settings,
            ),
            Fft2midi::new(
                pad_window_size[1],
                hop_size[1],
                sr_vec[1],
                segments_data_range / 2,
                start_note_mid,
                end_note_mid,
                settings,
            ),
            Fft2midi::new(
                pad_window_size[2],
                hop_size[2],
                sr_vec[2],
                segments_data_range,
                start_note_high,
                end_note_high,
                settings,
            ),
        ];

        println!("Converting to MIDI");
        for i in 0..segments_data_range {
            pb.inc(1);
            // 低域
            if i % 4 == 0 {
                let j = i / 4;
                let seg = split[0].audio_split(&resampled_data[0], j);
                let fft_seg = fft[0].fft(seg);
                fft2midi[0].convert(fft_seg, j);
            }

            // 中域
            if i % 2 == 0 {
                let j = i / 2;
                let seg = split[1].audio_split(&resampled_data[1], j);
                let fft_seg = fft[1].fft(seg);
                fft2midi[1].convert(fft_seg, j);
            }

            // 高域
            let seg = split[2].audio_split(&resampled_data[2], i);
            let fft_seg = fft[2].fft(seg);
            fft2midi[2].convert(fft_seg, i);
        }
        pb.finish_with_message("Conversion to MIDI complete");

        fft2midi
    };

    midi.normalize_velocity(fft2midi, segments_data_range);

    midi.export(output_name);
}
