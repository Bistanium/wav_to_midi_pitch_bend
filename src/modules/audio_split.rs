use super::math::hann_window;
use super::math::pad;

pub struct AudioSplit {
    win: Vec<f64>,
    win_size: usize,
    hop_size: usize,
    pad_multiple: usize,
}

impl AudioSplit {
    pub fn new(win_size: usize, hop_size: usize, pad_multiple: usize) -> Self {
        let win = hann_window(win_size);
        Self {
            win,
            win_size,
            hop_size,
            pad_multiple,
        }
    }

    // オーディオ分割
    pub fn audio_split(&self, data: &[f64], i: usize) -> Vec<f64> {
        let pad_size = self.win_size * (self.pad_multiple - 1);
        // 複数窓間の同期に必要
        let center = (self.win_size / 2) as i32;
        let start = (i * self.hop_size) as i32 - center;
        let end = start + self.win_size as i32;
        // 範囲外を含むデータは捨てる
        let segment = if start < 0 || data.len() < end as usize {
            vec![0.0; self.win_size]
        } else {
            data[start as usize..end as usize]
                .iter()
                .zip(self.win.iter())
                .map(|(x, w)| x * w)
                .collect::<Vec<f64>>()
        };
        let mut padded_segment = pad(&segment, 0, pad_size as i32);
        padded_segment.rotate_left(center as usize);

        padded_segment
    }
}
