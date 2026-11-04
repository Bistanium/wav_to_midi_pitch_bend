use super::math::linspace_int;

pub struct Regions {
    peaks: Vec<usize>,
    borders: Vec<usize>,
    regions: Vec<[usize; 2]>,

    none_region: Vec<[usize; 2]>,
    none_peak_indice: Vec<usize>,
}

impl Regions {
    pub fn new(max_len: usize) -> Self {
        Self {
            peaks: Vec::with_capacity(max_len),
            borders: vec![0; max_len + 1],
            regions: Vec::with_capacity(max_len),

            none_region: vec![[0, 0]],
            none_peak_indice: vec![0],
        }
    }

    fn detect_peaks(&mut self, data: &[f64], n_array: Vec<usize>) {
        self.peaks.clear();
        let len = data.len();
        for i in 0..len {
            let n = n_array[i];

            if i < n || i + n + 1 > len {
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

    pub fn divide_into_regions(
        &mut self,
        data: &[f64],
        n_min: f64,
        n_max: f64,
    ) -> (Vec<[usize; 2]>, Vec<usize>) {
        let len = data.len();

        let n_array = linspace_int(n_min, n_max, len);

        self.detect_peaks(data, n_array);

        let num_peaks = self.peaks.len();

        if num_peaks == 0 {
            return (self.none_region.to_vec(), self.none_peak_indice.to_vec());
        }

        self.borders[0] = 0;
        self.borders[num_peaks] = len;

        for i in 0..(num_peaks - 1) {
            let i1 = self.peaks[i];
            let i2 = self.peaks[i + 1];

            let mut min_idx = i1;
            let mut min_val = data[i1];

            for j in (i1 + 1)..i2 {
                let v = data[j];
                if v < min_val {
                    min_val = v;
                    min_idx = j;
                }
            }

            self.borders[i + 1] = min_idx;
        }

        self.regions.clear();
        for i in 0..num_peaks {
            self.regions.push([self.borders[i], self.borders[i + 1]]);
        }

        (self.regions.to_vec(), self.peaks.to_vec())
    }
}
