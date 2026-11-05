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

    divide: Regions,
    win_size: usize,
    hop_size: usize,
    number_of_tracks: usize,
    start_note: usize,
    end_note: usize,
    note_nums: usize,
    sec: f64,
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
        let number_of_tracks = settings.number_of_tracks;
        let half_win_size = win_size / 2 + 1;
        let mut omega = vec![0.0; half_win_size];
        for (i, o) in omega.iter_mut().enumerate() {
            *o = 2.0 * PI * i as f64 / win_size as f64;
        }
        let sec = win_size as f64 / sr as f64;
        // tenp_trackのサイズを削減する
        let note_nums = end_note - start_note;

        Self {
            temp_track: vec![vec![0.0; number_of_tracks * note_nums]; segments_data_range],
            mag: vec![0.0; half_win_size],
            phase: vec![0.0; half_win_size],
            phase_prev: vec![0.0; half_win_size],
            omega,

            divide: Regions::new(half_win_size),
            win_size,
            hop_size,
            number_of_tracks,
            start_note,
            end_note,
            note_nums,
            sec,
            sr,
        }
    }

    pub fn get_temp_track(&self) -> &[Vec<f64>] {
        &self.temp_track
    }

    pub fn set_temp_track(&mut self, i: usize, j: usize, v: f64) {
        self.temp_track[i][j] = v;
    }

    pub fn get_start_end_note(&self) -> (usize, usize) {
        (self.start_note, self.end_note)
    }

    fn estimate_frequency(&self) -> Vec<f64> {
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
        // omegaの式より真の周波数ビン(np.arrangeの値)を得る
        let f_bin = true_omega.map(|x| x * self.win_size as f64 / (2.0 * PI));
        // 周波数ビンを周波数に変換
        let mut frequency = f_bin
            .map(|x| x / self.win_size as f64 * self.sr as f64)
            .collect::<Vec<f64>>();
        // 対数がエラーにならないように
        for x in frequency.iter_mut() {
            *x = x.max(1.0);
            // ノート番号を求める前準備
            *x = x.log10();
        }

        frequency
    }

    // ピッチベンド込みの音階とその音量をcurrent_volume_listに書き込む
    fn write_volume(&mut self, note_value: f64, volume_value: f64, frame: usize) {
        let round_0_note = (note_value + 0.5) as usize;
        // 範囲外のノートは納められないので除外する
        if self.start_note <= round_0_note && round_0_note < self.end_note {
            let midi_note_decimal = note_value - note_value.floor();
            let track_num = (midi_note_decimal * self.number_of_tracks as f64 + 0.5) as usize;
            let idx = track_num * self.note_nums + round_0_note - self.start_note;
            // 平方根をとると音がいい感じになる
            // 同じ音階に複数回書き込まれたときは大きい方を採用する
            self.temp_track[frame][idx] =
                self.temp_track[frame][idx].max(volume_value.sqrt());
        }
    }

    pub fn convert(&mut self, fft: &[Complex<f64>], frame: usize) {
        const TWELFTH_ROOT_OF_2: f64 = 1.059463094359295264561825294946; // 2^(1/12)
        const TWELVE_OVER_LOG10_2: f64 = 39.86313713864834817444383315387; // 12/log10(2)
        const LOG10_440_TIMES_TOL_MINUS_69: f64 = 36.37631656229591524883618971458; // 12*log10(440)/log10(2)-69

        let mut before_note = 0.0;
        let mut sum_volume = 0.0;
        to_mag_phase(fft, &mut self.mag, &mut self.phase);
        // 窓サイズによる音量の違いを統一する
        let norm_mag = self
            .mag
            .iter()
            .map(|x| x / self.win_size as f64)
            .collect::<Vec<f64>>();

        // 振幅スペクトルのピークとそれが支配する領域を得る
        let (regions, peak_indices) = self.divide.divide_into_regions(&norm_mag, 2.0, 2.0);
        // 位相差から正確な周波数を推定する
        let truth_notes = self.estimate_frequency();

        // 開始ビン±8から開始してmidiノート周辺に対応するビンの振幅を拾えるようにする
        let range_start = 
            (self.sec * 440.0 * TWELFTH_ROOT_OF_2.powf(self.start_note as f64 - 69.0) - 8.0) as usize;
        let range_end =
            (self.sec * 440.0 * TWELFTH_ROOT_OF_2.powf(self.end_note as f64 - 69.0) + 8.0) as usize;

        let (peak_left, peak_right) = filter_range(&peak_indices, range_start, range_end);
        let filtered_peak_indices = &peak_indices[peak_left..peak_right];

        let threshold = 1.0 / (2 * self.number_of_tracks) as f64;
        for (i, peak_index) in filtered_peak_indices.iter().enumerate() {
            let j = i + peak_left;
            // ピーク周辺の音はピークのものとみなす
            let regions_left = regions[j][0];
            let regions_right = regions[j][1];
            let volume = norm_mag[regions_left..regions_right].iter().sum();

            // ノート番号計算
            let note =
                truth_notes[*peak_index] * TWELVE_OVER_LOG10_2 - LOG10_440_TIMES_TOL_MINUS_69;
            let round_note = 
                ((note * self.number_of_tracks as f64 + 0.5) as i32) as f64 / self.number_of_tracks as f64;

            // 音が変わったら前の音階のデータをVecに書き込む
            if (round_note - before_note).abs() >= threshold {
                self.write_volume(before_note, sum_volume, frame);

                before_note = round_note;
                sum_volume = volume;
            } else {
                sum_volume += volume;
            }
        }
        // 最後の音階はループ内で書き込まれないのでここで書き込む
        self.write_volume(before_note, sum_volume, frame);

        self.phase_prev.copy_from_slice(&self.phase);
    }
}

fn filter_range(a: &[usize], start: usize, end: usize) -> (usize, usize) {
    let left = a.partition_point(|&x| x < start);
    let right = a.partition_point(|&x| x <= end);

    (left, right)
}
