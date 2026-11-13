use super::math::hann_window;

pub struct AudioSplit {
    buffer: Vec<f64>,
    win: Vec<f64>,
    win_size: usize,
    hop_size: usize,
    center: isize,
}

impl AudioSplit {
    pub fn new(win_size: usize, hop_size: usize, pad_multiple: usize) -> Self {
        let win = hann_window(win_size);

        let center = (win_size / 2) as isize;
        Self {
            buffer: vec![0.0; win_size * pad_multiple],
            win,
            win_size,
            hop_size,
            center,
        }
    }

    // オーディオ分割
    pub fn audio_split(&mut self, data: &[f64], frame: usize) -> &[f64] {
        // 複数窓間の同期に必要
        let pos = isize::try_from(
            frame.checked_mul(self.hop_size)
                .expect("usize multiplication overflow"),
        )
        .expect("isize conversion overflow");
        let start = pos - self.center;
        let end = start
            .checked_add(self.win_size as isize)
            .expect("isize addition overflow");
        let end = end as usize;
        // 範囲外を含むデータは捨てる(事前にパディングしてあるため)
        if 0 <= start && end <= data.len() {
            let start = start as usize;
            for ((b, d), w) in self.buffer[..self.win_size]
                .iter_mut()
                .zip(data[start..end].iter())
                .zip(self.win.iter())
            {
                *b = d * w;
            }
        } else {
            self.buffer.fill(0.0);
        }

        &self.buffer
    }
}
