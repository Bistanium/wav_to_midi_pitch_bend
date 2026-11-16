use std::f64::consts::PI;

use indicatif::{ProgressBar, ProgressStyle};
use rustfft::num_complex::Complex;

use super::audio_split::AudioSplit;
use super::fft::Fft;
use super::math::{filter_range, pad, to_mag_phase, wrap_to_pi};
use super::midi::Midi;
use super::regions::Regions;
use super::resampling::resampling_n;
use super::settings::Settings;

pub struct Track {
    pub temp_track: Vec<Vec<f64>>,
    pub start_note: usize,
    pub end_note: usize,
    number_of_tracks: usize,
}

impl Track {
    pub fn new(
        number_of_tracks: usize,
        start_note: usize,
        end_note: usize,
        segments_data_range: usize,
    ) -> Self {
        let notes_range = end_note - start_note;
        Self {
            temp_track: vec![vec![0.0; number_of_tracks * notes_range]; segments_data_range],
            start_note,
            end_note,
            number_of_tracks,
        }
    }

    // ピッチベンド込みの音階とその音量をcurrent_volume_listに書き込む
    fn write_volume(&mut self, note_value: f64, volume_value: f64, frame: usize) {
        let round_note = note_value.round() as usize;
        // 範囲外のノートは納められないので除外する
        if (self.start_note..self.end_note).contains(&round_note) {
            let decimal = note_value.fract();
            let track = (decimal * self.number_of_tracks as f64).round() as usize;
            let idx = track * (self.end_note - self.start_note) + round_note - self.start_note;
            // 同じ音階に複数回書き込まれたときはそのまま加算する
            self.temp_track[frame][idx] += volume_value;
        }
    }
}

pub struct Fft2midi {
    mag: Vec<f64>,
    phase: Vec<f64>,
    phase_prev: Vec<f64>,
    omega: Vec<f64>,
    truth_frequency: Vec<f64>,

    regions: Regions,
    win_size: usize,
    hop_size: usize,
    range_start: usize,
    range_end: usize,
    sr: u32,
}

impl Fft2midi {
    pub fn new(
        win_size: usize,
        hop_size: usize,
        sr: u32,
        start_note: usize,
        end_note: usize,
    ) -> Self {
        const TWELFTH_ROOT_OF_2: f64 = 1.059463094359295264561825294946; // 2^(1/12)

        let half_win_size = win_size / 2 + 1;

        let mut omega = vec![0.0; half_win_size];
        for (i, o) in omega.iter_mut().enumerate() {
            *o = 2.0 * PI * i as f64 / win_size as f64;
        }

        // 0.5と4.0の結果がよかった
        let regions = Regions::new(half_win_size, 0.5, 4.0);

        let sec = win_size as f64 / sr as f64;
        // 開始ビン±6から開始してノート周辺に対応するビンの振幅を拾えるようにする
        let range_start =
            (sec * 440.0 * TWELFTH_ROOT_OF_2.powi(start_note as i32 - 69) - 6.0) as usize;
        let range_end = (sec * 440.0 * TWELFTH_ROOT_OF_2.powi(end_note as i32 - 69) + 6.0) as usize;

        Self {
            mag: vec![0.0; half_win_size],
            phase: vec![0.0; half_win_size],
            phase_prev: vec![0.0; half_win_size],
            omega,
            truth_frequency: vec![0.0; half_win_size],

            regions,
            win_size,
            hop_size,
            range_start,
            range_end,
            sr,
        }
    }

    fn estimate_frequency(&mut self) {
        // フレーム間隔
        let delta_t = self.hop_size as f64;
        // フレーム間の周波数の位相の変化と対応する周波数のビンから計算される位相の変化との差
        let delta_phi = self
            .phase
            .iter()
            .zip(self.phase_prev.iter())
            .zip(self.omega.iter())
            .map(|((p, pp), o)| wrap_to_pi(p - pp - o * delta_t));
        // omegaを補正して位相の変化がより正確になるようにする
        let true_omega = self
            .omega
            .iter()
            .zip(delta_phi)
            .map(|(o, d)| o + d / delta_t);
        // 真のビンを得る
        let f_bin = true_omega.map(|x| x * self.win_size as f64 / (2.0 * PI));
        // ビンを周波数に変換
        let frequency = f_bin.map(|x| x / self.win_size as f64 * self.sr as f64);

        // -inf回避
        for (t, f) in self.truth_frequency.iter_mut().zip(frequency) {
            *t = f.max(1.0);
        }
    }

    pub fn convert(&mut self, track: &mut Track, fft: &[Complex<f64>], frame: usize) {
        const LOG2_440_TIMES_12_MINUS_69: f64 = 36.37631656229591524883618971458; // 12*log2(440)-69

        let mut before_note = 0.0;
        let mut sum_volume = 0.0;
        let number_of_tracks = track.number_of_tracks as f64;
        let threshold = 1.0 / (2.0 * number_of_tracks);

        to_mag_phase(fft, &mut self.mag, &mut self.phase);

        // 位相差から正確な周波数を推定する
        self.estimate_frequency();

        // 振幅スペクトルのピークとそれが支配する領域を得る
        let (regions, peak_indices) = self.regions.divide_into_regions(&self.mag);

        let (peak_left, peak_right) = filter_range(peak_indices, self.range_start, self.range_end);
        let peak_indices = &peak_indices[peak_left..peak_right];
        let regions = &regions[peak_left..peak_right];

        for (region, peak_index) in regions.iter().zip(peak_indices) {
            // ピーク周辺の音はピークのものとみなす
            let regions_left = region[0];
            let regions_right = region[1];
            let volume = self.mag[regions_left..regions_right]
                .iter()
                .fold(0.0, |volume, x| volume + x);
            // 窓サイズによる音量の違いを統一する
            let volume = volume / self.win_size as f64;

            // ノート番号計算
            let note = 12.0 * self.truth_frequency[*peak_index].log2() - LOG2_440_TIMES_12_MINUS_69;
            let note = (note * number_of_tracks).round() / number_of_tracks;

            // 音が変わったら前の音階のデータをVecに書き込む
            if (note - before_note).abs() >= threshold {
                track.write_volume(before_note, sum_volume, frame);

                before_note = note;
                sum_volume = volume;
            } else {
                sum_volume += volume;
            }
        }
        // 最後の音階はループ内で書き込まれないのでここで書き込む
        track.write_volume(before_note, sum_volume, frame);

        self.phase_prev.copy_from_slice(&self.phase);
    }
}

pub fn midi_from_wav(data: Vec<f64>, sr: u32, output_name: String, settings: Settings) {
    // 0: 0.2s, 1: 0.1s, 2: 0.05s, 3: 0.025s
    let window_size = [128, 256, 512, 1024];
    let sr_vec = [640, 2560, 10240, 40960];

    let start_note = settings.start_note as usize;
    let end_note = settings.end_note as usize;
    // 境界値に収める
    let bounds = [12, 56, 80, 104, 128];
    let start_notes = [
        start_note.clamp(bounds[0], bounds[1]),
        start_note.clamp(bounds[1], bounds[2]),
        start_note.clamp(bounds[2], bounds[3]),
        start_note.clamp(bounds[3], bounds[4]),
    ];
    let end_notes = [
        end_note.clamp(bounds[0], bounds[1]),
        end_note.clamp(bounds[1], bounds[2]),
        end_note.clamp(bounds[2], bounds[3]),
        end_note.clamp(bounds[3], bounds[4]),
    ];

    assert!(
        window_size.len() == sr_vec.len()
            && window_size.len() == start_notes.len()
            && window_size.len() == end_notes.len(),
        "slice lengths do not match"
    );

    let rate = [
        40 * window_size[0] / sr_vec[0] as usize,
        40 * window_size[1] / sr_vec[1] as usize,
        40 * window_size[2] / sr_vec[2] as usize,
        40 * window_size[3] / sr_vec[3] as usize,
    ];

    let pad_multiple = settings.pad_multiple;
    let pad_window_size = [
        window_size[0] * pad_multiple,
        window_size[1] * pad_multiple,
        window_size[2] * pad_multiple,
        window_size[3] * pad_multiple,
    ];
    let hop_size = [
        window_size[0] / 4,
        window_size[1] / 4,
        window_size[2] / 4,
        window_size[3] / 4,
    ];

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
    let segments_data_range = ((resampled_data[0].len() / window_size[0] - 1) * 4 * rate[0])
        .min((resampled_data[1].len() / window_size[1] - 1) * 4 * rate[1])
        .min((resampled_data[2].len() / window_size[2] - 1) * 4 * rate[2])
        .min((resampled_data[3].len() / window_size[3] - 1) * 4 * rate[3]);

    let number_of_tracks = settings.number_of_tracks;
    let mut tracks = [
        Track::new(
            number_of_tracks,
            start_notes[0],
            end_notes[0],
            segments_data_range / rate[0],
        ),
        Track::new(
            number_of_tracks,
            start_notes[1],
            end_notes[1],
            segments_data_range / rate[1],
        ),
        Track::new(
            number_of_tracks,
            start_notes[2],
            end_notes[2],
            segments_data_range / rate[2],
        ),
        Track::new(
            number_of_tracks,
            start_notes[3],
            end_notes[3],
            segments_data_range / rate[3],
        ),
    ];

    {
        // 異なる窓用にそれぞれ用意
        let mut split = [
            AudioSplit::new(window_size[0], hop_size[0], pad_multiple),
            AudioSplit::new(window_size[1], hop_size[1], pad_multiple),
            AudioSplit::new(window_size[2], hop_size[2], pad_multiple),
            AudioSplit::new(window_size[3], hop_size[3], pad_multiple),
        ];

        let mut fft = [
            Fft::new(pad_window_size[0]),
            Fft::new(pad_window_size[1]),
            Fft::new(pad_window_size[2]),
            Fft::new(pad_window_size[3]),
        ];

        let mut fft2midi = [
            Fft2midi::new(
                pad_window_size[0],
                hop_size[0],
                sr_vec[0],
                start_notes[0],
                end_notes[0],
            ),
            Fft2midi::new(
                pad_window_size[1],
                hop_size[1],
                sr_vec[1],
                start_notes[1],
                end_notes[1],
            ),
            Fft2midi::new(
                pad_window_size[2],
                hop_size[2],
                sr_vec[2],
                start_notes[2],
                end_notes[2],
            ),
            Fft2midi::new(
                pad_window_size[3],
                hop_size[3],
                sr_vec[3],
                start_notes[3],
                end_notes[3],
            ),
        ];

        // プログレスバー範囲
        let pb = ProgressBar::new(segments_data_range as u64);
        // 見た目
        pb.set_style(
            ProgressStyle::with_template(
                "[{percent}%] {bar:40} {pos}/{len} [{elapsed_precise}<{eta_precise}]",
            )
            .unwrap(),
        );

        println!("Converting to MIDI");
        for i in 0..segments_data_range {
            if i % 8 == 0 {
                let j = i / 8;
                let seg = split[0].audio_split(&resampled_data[0], j);
                let fft_seg = fft[0].fft(seg);
                fft2midi[0].convert(&mut tracks[0], fft_seg, j);
            }

            if i % 4 == 0 {
                let j = i / 4;
                let seg = split[1].audio_split(&resampled_data[1], j);
                let fft_seg = fft[1].fft(seg);
                fft2midi[1].convert(&mut tracks[1], fft_seg, j);
            }

            if i % 2 == 0 {
                let j = i / 2;
                let seg = split[2].audio_split(&resampled_data[2], j);
                let fft_seg = fft[2].fft(seg);
                fft2midi[2].convert(&mut tracks[2], fft_seg, j);
            }

            let seg = split[3].audio_split(&resampled_data[3], i);
            let fft_seg = fft[3].fft(seg);
            fft2midi[3].convert(&mut tracks[3], fft_seg, i);

            pb.inc(1);
        }
        pb.finish_with_message("Conversion to MIDI complete");
    }

    // midi定義
    let mut midi = Midi::new(settings);

    midi.normalize_velocity(&mut tracks, segments_data_range);

    midi.export(output_name);
}
