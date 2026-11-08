use super::math::{to_mag_phase, wrap_to_pi};
use super::regions::Regions;
use super::settings::Settings;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

pub struct Fft2midi {
    temp_track: Vec<Vec<f64>>,
    mag: Vec<f64>,
    phase: Vec<f64>,
    phase_prev: Vec<f64>,
    omega: Vec<f64>,
    truth_frequency: Vec<f64>,

    regions: Regions,
    win_size: usize,
    hop_size: usize,
    number_of_tracks: usize,
    start_note: usize,
    end_note: usize,
    notes_range: usize,
    range_start: usize,
    range_end: usize,
    current_frame: usize,
    sr: u32,
}

impl Fft2midi {
    pub fn new(
        win_size: usize,
        hop_size: usize,
        sr: u32,
        segments_data_range: usize,
        start_note: usize,
        end_note: usize,
        settings: &Settings,
    ) -> Self {
        const TWELFTH_ROOT_OF_2: f64 = 1.059463094359295264561825294946; // 2^(1/12)

        let number_of_tracks = settings.number_of_tracks;
        let half_win_size = win_size / 2 + 1;

        let mut omega = vec![0.0; half_win_size];
        for (i, o) in omega.iter_mut().enumerate() {
            *o = 2.0 * PI * i as f64 / win_size as f64;
        }

        // 2.0と2.0が最も結果がよかった
        let regions = Regions::new(half_win_size, 2.0, 2.0);

        // temp_trackのサイズを削減する
        let notes_range = end_note - start_note;

        let sec = win_size as f64 / sr as f64;
        // 開始ビン±8から開始してmidiノート周辺に対応するビンの振幅を拾えるようにする
        let range_start = (sec * 440.0 * TWELFTH_ROOT_OF_2.powi(start_note as i32 - 69) - 8.0) as usize;
        let range_end = (sec * 440.0 * TWELFTH_ROOT_OF_2.powi(end_note as i32 - 69) + 8.0) as usize;

        Self {
            temp_track: vec![vec![0.0; number_of_tracks * notes_range]; segments_data_range],
            mag: vec![0.0; half_win_size],
            phase: vec![0.0; half_win_size],
            phase_prev: vec![0.0; half_win_size],
            omega,
            truth_frequency: vec![0.0; half_win_size],

            regions,
            win_size,
            hop_size,
            number_of_tracks,
            start_note,
            end_note,
            notes_range,
            range_start,
            range_end,
            current_frame: 0,
            sr,
        }
    }

    pub fn get_temp_track(&self) -> &[Vec<f64>] {
        &self.temp_track
    }

    pub fn set_temp_track(&mut self, i: usize, j: usize, value: f64) {
        self.temp_track[i][j] = value;
    }

    pub fn get_note_info(&self) -> (usize, usize, usize) {
        (self.start_note, self.end_note, self.notes_range)
    }

    fn estimate_frequency(&mut self) {
        // フレーム間隔
        let delta_t = self.hop_size as f64;
        // フレーム間の周波数の位相の変化と対応する周波数のビンから計算される位相の変化との差
        let delta_phi = self.phase
            .iter()
            .zip(self.phase_prev.iter())
            .zip(self.omega.iter())
            .map(|((p, pp), o)| wrap_to_pi(p - pp - o * delta_t));
        // omegaを補正して位相の変化がより正確になるようにする
        let true_omega = self.omega
            .iter()
            .zip(delta_phi)
            .map(|(o, d)| o + d / delta_t);
        // omegaの式より真の周波数ビンを得る
        let f_bin = true_omega.map(|x| x * self.win_size as f64 / (2.0 * PI));
        // 周波数ビンを周波数に変換
        let frequency = f_bin.map(|x| x / self.win_size as f64 * self.sr as f64);

        // 対数がエラーにならないように
        for (t, f) in self.truth_frequency.iter_mut().zip(frequency) {
            *t = f.max(1.0);
        }
    }

    // ピッチベンド込みの音階とその音量をcurrent_volume_listに書き込む
    fn write_volume(&mut self, note_value: f64, volume_value: f64) {
        let round_note = note_value.round() as usize;
        // 範囲外のノートは納められないので除外する
        if (self.start_note..self.end_note).contains(&round_note) {
            let decimal = note_value.fract();
            let track = (decimal * self.number_of_tracks as f64).round() as usize;
            let idx = track * self.notes_range + round_note - self.start_note;
            // 同じ音階に複数回書き込まれたときはそのまま加算する
            self.temp_track[self.current_frame][idx] += volume_value;
        }
    }

    pub fn convert(&mut self, fft: &[Complex<f64>]) {
        const LOG2_440_TIMES_12_MINUS_69: f64 = 36.37631656229591524883618971458; // 12*log2(440)-69

        let mut before_note = 0.0;
        let mut sum_volume = 0.0;
        to_mag_phase(fft, &mut self.mag, &mut self.phase);

        // 位相差から正確な周波数を推定する
        self.estimate_frequency();

        // 振幅スペクトルのピークとそれが支配する領域を得る
        let (regions, peak_indices) = self.regions.divide_into_regions(&self.mag);

        let (peak_left, peak_right) = filter_range(peak_indices, self.range_start, self.range_end);
        let peak_indices = peak_indices[peak_left..peak_right].to_vec();
        let regions = regions[peak_left..peak_right].to_vec();

        let threshold = 1.0 / (2 * self.number_of_tracks) as f64;
        for (region, peak_index) in regions.iter().zip(peak_indices.iter()) {
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
            let note = (note * self.number_of_tracks as f64).round() / self.number_of_tracks as f64;

            // 音が変わったら前の音階のデータをVecに書き込む
            if (note - before_note).abs() >= threshold {
                self.write_volume(before_note, sum_volume);

                before_note = note;
                sum_volume = volume;
            } else {
                sum_volume += volume;
            }
        }
        // 最後の音階はループ内で書き込まれないのでここで書き込む
        self.write_volume(before_note, sum_volume);

        self.phase_prev.copy_from_slice(&self.phase);
        self.current_frame += 1;
    }
}

fn filter_range(data: &[usize], start: usize, end: usize) -> (usize, usize) {
    let left = data.partition_point(|&x| x < start);
    let right = data.partition_point(|&x| x <= end);

    (left, right)
}
