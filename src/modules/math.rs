use rustfft::num_complex::Complex;
use std::f64::consts::PI;

pub fn hann_window(size: usize) -> Vec<f64> {
    let hann_base = 2.0 * PI / (size - 1) as f64;
    (0..size)
        .map(|n| 0.5 * (1.0 - (hann_base * n as f64).cos()))
        .collect()
}

pub fn wrap_to_pi(phase: f64) -> f64 {
    (phase + PI).rem_euclid(2.0 * PI) - PI
}

pub fn linspace_int(start: f64, end: f64, num: usize) -> Vec<usize> {
    let step = (end - start) / (num - 1) as f64;
    (0..num)
        .map(|i| (start + step * i as f64 + 0.5) as usize)
        .collect()
}

pub fn to_mag_phase(spectrum: &[Complex<f64>], mag: &mut [f64], phase: &mut [f64]) {
    for ((s, m), p) in spectrum.iter().zip(mag.iter_mut()).zip(phase.iter_mut()) {
        (*m, *p) = Complex::to_polar(*s);
    }
}

fn pad_zero(data: &[f64], f_amount: usize, b_amount: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len() + f_amount + b_amount);
    // 前に0を追加
    out.extend(std::iter::repeat_n(0.0, f_amount));
    out.extend_from_slice(data);
    // 後ろに0を追加
    out.extend(std::iter::repeat_n(0.0, b_amount));

    out
}

pub fn pad(data: &[f64], front_amount: i32, back_amount: i32) -> Vec<f64> {
    if front_amount >= 0 && back_amount >= 0 {
        pad_zero(data, front_amount as usize, back_amount as usize)
    } else if front_amount >= 0 && back_amount < 0 {
        pad_zero(
            &data[0..(data.len() - (-back_amount) as usize)],
            front_amount as usize,
            0,
        )
    } else if front_amount < 0 && back_amount >= 0 {
        pad_zero(
            &data[((-front_amount) as usize)..data.len()],
            0,
            back_amount as usize,
        )
    } else {
        data[((-front_amount) as usize)..(data.len() - (-back_amount) as usize)].to_vec()
    }
}
