use std::f64::consts::PI;

use rustfft::num_complex::Complex;

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
        .map(|i| (start + step * i as f64).round() as usize)
        .collect()
}

pub fn to_mag_phase(spectrum: &[Complex<f64>], mag: &mut [f64], phase: &mut [f64]) {
    for ((s, m), p) in spectrum.iter().zip(mag.iter_mut()).zip(phase.iter_mut()) {
        (*m, *p) = Complex::to_polar(*s);
    }
}

pub fn pad(mut data: Vec<f64>, front_amount: i32, back_amount: i32) -> Vec<f64> {
    // 減らす量が元のデータの長さよりも大きくなるかどうか
    let front_remove = if front_amount < 0 {
        front_amount.unsigned_abs() as usize
    } else {
        0
    };
    let back_remove = if back_amount < 0 {
        back_amount.unsigned_abs() as usize
    } else {
        0
    };
    if data.len() <= front_remove + back_remove {
        eprintln!("Padding is not performed, as the original data would be lost.");
        return data;
    }

    if front_amount > 0 {
        data.splice(0..0, std::iter::repeat_n(0.0, front_amount as usize));
    } else if front_amount < 0 {
        data.drain(..front_remove);
    }

    if back_amount > 0 {
        data.resize(data.len() + back_amount as usize, 0.0);
    } else if back_amount < 0 {
        data.truncate(data.len() - back_remove);
    }

    data
}

pub fn filter_range(data: &[usize], start: usize, end: usize) -> (usize, usize) {
    let left = data.partition_point(|&x| x < start);
    let right = data.partition_point(|&x| x <= end);

    (left, right)
}
