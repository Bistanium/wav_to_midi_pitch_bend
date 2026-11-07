use super::math::{hann_window, pad};

pub struct AudioSplit {
    win: Vec<f64>,
    win_size: usize,
    hop_size: usize,
    pad_size: usize,
    center: usize,
    current_frame: usize,
}

impl AudioSplit {
    pub fn new(win_size: usize, hop_size: usize, pad_multiple: usize) -> Self {
        let win = hann_window(win_size);

        let pad_size = win_size * (pad_multiple - 1);
        let center = win_size / 2;
        Self {
            win,
            win_size,
            hop_size,
            pad_size,
            center,
            current_frame: 0,
        }
    }

    // オーディオ分割
    pub fn audio_split(&mut self, data: &[f64]) -> Vec<f64> {
        // 複数窓間の同期に必要
        let start = (self.current_frame * self.hop_size) as i32 - self.center as i32;
        let end = start + self.win_size as i32;
        self.current_frame += 1;
        // 範囲外を含むデータは捨てる
        let segment = if start < 0 || (data.len() as i32) < end {
            vec![0.0; self.win_size]
        } else {
            data[start as usize..end as usize]
                .iter()
                .zip(self.win.iter())
                .map(|(x, w)| x * w)
                .collect()
        };
        
        pad(segment, 0, self.pad_size as i32)
    }
}

