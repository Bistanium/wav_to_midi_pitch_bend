use rfd::FileDialog;
use std::path::Path;

use crate::modules::settings::Settings;

pub fn preparation(setting: &Settings) -> Result<(String, String), Box<dyn std::error::Error>> {
    let file = FileDialog::new()
        .add_filter("WAV audio", &["wav"])
        .pick_file();

    let input_name = match file {
        // ウィンドウを閉じたとき
        None => {
            std::process::exit(0);
        }
        Some(path) => path.to_string_lossy().to_string(),
    };
    let path = Path::new(&input_name);

    // ディレクトリ
    let dir = path.parent().and_then(|p| p.to_str()).unwrap_or("");
    // 拡張子抜きファイル名
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    // 拡張子
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    // 大文字と小文字を区別しない
    if !ext.eq_ignore_ascii_case("wav") {
        return Err("Error: wavファイルを入力してください。".into());
    }

    // ピッチベンドの有無
    let stem = if setting.disable_pitch_bend {
        &format!("{} {}", stem, "[disable pitch bend]")
    } else {
        stem
    };

    // ディレクトリがついてくるときがあるのであったらつける
    let temp_output_name = if dir.is_empty() {
        format!("{}.{}", stem, "mid")
    } else {
        format!("{}\\{}.{}", dir, stem, "mid")
    };

    let mut output_name = temp_output_name.clone();
    let o_path = Path::new(&temp_output_name);

    // 名前に重複があれば (1),(2) のように名前を変えていく
    for i in 0..100 {
        if i > 0 {
            let stem = o_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let ext = o_path.extension().and_then(|s| s.to_str()).unwrap_or("");
            let dir = path.parent().and_then(|p| p.to_str()).unwrap_or("");
            if dir.is_empty() {
                output_name = format!("{} ({}).{}", stem, i, ext);
            } else {
                output_name = format!("{}\\{} ({}).{}", dir, stem, i, ext);
            }
        }
        let o_path = Path::new(&output_name);
        // 同名のファイルがなければ抜ける
        if !o_path.exists() {
            break;
        }
    }

    Ok((input_name, output_name))
}
