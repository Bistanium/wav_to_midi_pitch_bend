use super::math::linspace_int;

pub struct Regions {
    peaks: Vec<usize>,
    borders: Vec<usize>,
    regions: Vec<[usize; 2]>,
    none_region: Vec<[usize; 2]>,
    none_peak_indice: Vec<usize>,
    n_array: Vec<usize>,
    win_size: usize,
}

impl Regions {
    pub fn new(win_size: usize, n_min: f64, n_max: f64) -> Self {
        let n_array = linspace_int(n_min, n_max, win_size);
        Self {
            peaks: Vec::with_capacity(win_size),
            borders: vec![0; win_size + 1],
            regions: Vec::with_capacity(win_size),
            none_region: vec![[0, 0]],
            none_peak_indice: vec![0],
            n_array,
            win_size,
        }
    }

    fn detect_peaks(&mut self, data: &[f64]) {
        self.peaks.clear();
        for i in 0..self.win_size {
            let n = self.n_array[i];

            if i < n || i + n + 1 > self.win_size {
                continue;
            }

            let center = data[i];

            if center < 1e-6 {
                continue;
            }

            // 振幅にマイナスはないので0.0で問題ない
            let mut left_max = 0.0;
            for v in &data[(i - n)..i] {
                if *v > left_max {
                    left_max = *v;
                }
            }

            let mut right_max = 0.0;
            for v in &data[(i + 1)..(i + n + 1)] {
                if *v > right_max {
                    right_max = *v;
                }
            }

            if center > left_max && center > right_max {
                self.peaks.push(i);
            }
        }
    }

    pub fn divide_into_regions(&mut self, data: &[f64]) -> (&[[usize; 2]], &[usize]) {
        self.detect_peaks(data);

        let num_peaks = self.peaks.len();
        if num_peaks == 0 {
            return (&self.none_region, &self.none_peak_indice);
        }

        self.borders[0] = 0;
        self.borders[num_peaks] = self.win_size;

        for i in 0..(num_peaks - 1) {
            let i1 = self.peaks[i];
            let i2 = self.peaks[i + 1];

            // 区間内の最小値のインデックスを得る
            let mut min_idx = i1;
            let mut min_val = data[i1];
            for (j, v) in data[(i1 + 1)..i2].iter().enumerate() {
                if *v < min_val {
                    min_val = *v;
                    min_idx = j + i1 + 1;
                }
            }
            self.borders[i + 1] = min_idx;
        }

        self.regions.clear();
        for i in 0..num_peaks {
            self.regions.push([self.borders[i], self.borders[i + 1]]);
        }

        (&self.regions, &self.peaks)
    }
}
