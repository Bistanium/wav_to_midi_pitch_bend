mod modules;
use crate::modules::midi_from_wav::midi_from_wav;
use crate::modules::preparation::preparation;
use crate::modules::settings::Settings;
use crate::modules::wav::load_wav;
use std::path::Path;

fn get_mono(left: Vec<f64>, right: Option<Vec<f64>>) -> Vec<f64> {
    match right {
        Some(s) => left.iter().zip(s.iter()).map(|(l, r)| l + r).collect(),
        None => left,
    }
}

fn show_output_info(output_name: &str) {
    let dir = Path::new(&output_name)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or("ERROR: DIRECTORY");
    let filename = Path::new(&output_name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("ERROR: FILE_NAME");
    if !dir.is_empty() {
        println!("ディレクトリ: {}", dir);
    }
    println!("ファイル名: {}", filename);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 設定ファイル
    let settings = Settings::load("settings.txt");

    // ダイアログ
    let (input_name, output_name) = preparation(&settings)?;

    // 出力ファイル名表示
    show_output_info(&output_name);

    // WAV読み込み
    let (left, right, sample_rate) = load_wav(input_name)?;
    let mono = get_mono(left, right);

    // WAVをMIDIに変換して書き出す
    midi_from_wav(mono, sample_rate, output_name, &settings);

    Ok(())
}
