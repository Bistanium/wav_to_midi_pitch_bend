use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

pub struct Fft {
    fft: Arc<dyn RealToComplex<f64>>,
    buffer: Vec<f64>,
    c_buffer: Vec<Complex<f64>>,
}

impl Fft {
    pub fn new(win_size: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(win_size);
        let half_win_size = win_size / 2 + 1;

        Self {
            fft,
            buffer: vec![0.0; win_size],
            c_buffer: vec![Complex::new(0.0, 0.0); half_win_size],
        }
    }

    pub fn fft(&mut self, input: &[f64]) -> &[Complex<f64>] {
        self.buffer.copy_from_slice(input);

        self.fft
            .process(&mut self.buffer, &mut self.c_buffer)
            .expect("fft failed");

        &self.c_buffer
    }
}
