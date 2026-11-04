mod modules;
use crate::modules::midi_from_wav::midi_from_wav;
use crate::modules::preparation::preparation;
use crate::modules::setting::Settings;
use crate::modules::wav::load_wav;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 設定ファイル
    let settings = Settings::load("setting.txt");

    let (input_name, output_name) = preparation(&settings)?;

    // 出力ファイル名表示
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

    // WAV読み込み
    let (left, right, sample_rate) = load_wav(input_name)?;
    let mono = if let Some(s) = right {
        left.iter().zip(s.iter()).map(|(l, r)| l + r).collect()
    } else {
        left
    };

    // WAVをMIDIに変換して書き出す
    midi_from_wav(&mono, sample_rate, output_name, &settings);

    Ok(())
}
