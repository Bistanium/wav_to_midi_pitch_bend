use super::math::hann_window;

pub struct AudioSplit {
    buffer: Vec<f64>,
    win: Vec<f64>,
    win_size: usize,
    hop_size: usize,
    center: usize,
}

impl AudioSplit {
    pub fn new(win_size: usize, hop_size: usize, pad_multiple: usize) -> Self {
        let win = hann_window(win_size);

        let center = win_size / 2;
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
        // frame * hop_sizeがusizeの範囲を超えるかどうか
        let pos = frame.checked_mul(self.hop_size);
        let pos = match pos {
            Some(p) => p,
            None => {
                eprintln!("usize multiplication overflow");
                self.buffer.fill(0.0);
                return &self.buffer;
            }
        };

        // 範囲外を含むデータは捨てる(事前にパディングしてあるため)
        if pos < self.center {
            self.buffer.fill(0.0);
            return &self.buffer;
        }

        // centerは複数窓間の同期に必要
        let start = pos - self.center;
        let end = start.checked_add(self.win_size);
        let end = match end {
            Some(e) => e,
            None => {
                eprintln!("usize addition overflow");
                self.buffer.fill(0.0);
                return &self.buffer;
            }
        };

        // 範囲外を含むデータは捨てる
        if end > data.len() {
            self.buffer.fill(0.0);
            return &self.buffer;
        }

        let buffer = &mut self.buffer[..self.win_size];
        let data = &data[start..end];
        for ((b, d), w) in buffer.iter_mut().zip(data).zip(self.win.iter()) {
            *b = d * w;
        }

        &self.buffer
    }
}
