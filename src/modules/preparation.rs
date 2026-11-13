use crate::modules::settings::Settings;
use clap::Parser;
use rfd::FileDialog;
use std::path::Path;

#[derive(Parser)]
struct Args {
    #[arg(short, long, value_name = "PATH", help = "input WAV file path")]
    input: String,

    #[arg(short, long, value_name = "PATH", help = "output MIDI file path")]
    output: Option<String>,
}

pub fn preparation(setting: &Settings) -> Result<(String, String), Box<dyn std::error::Error>> {
    let arg_count = std::env::args_os().len();

    let (input_name, output_name) = if arg_count > 1 {
        // コマンドラインから指定したとき
        let args = Args::parse();
        let input_name = args.input;
        let output_name = args.output;

        (input_name, output_name)    
    } else {
        // exeをダブルクリックしたとき
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

        (input_name, None)
    };

    // 入力ファイルのパス
    let path = Path::new(&input_name);

    // ファイルが存在するか
    if !path.exists() {
        return Err("指定されたファイルが見つかりません。".into());
    }

    // ディレクトリ
    let dir = path.parent().and_then(|p| p.to_str()).unwrap_or("");
    // 拡張子抜きファイル名
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    // 拡張子
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    // 大文字と小文字を区別しない
    if !ext.eq_ignore_ascii_case("wav") {
        return Err("WAVファイルを入力してください。".into());
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

    // 出力ファイル名が指定されていればそれを使う
    let temp_output_name = if let Some(output_name) = output_name {
        output_name
    } else {
        temp_output_name
    };

    let mut output_name = temp_output_name.clone();
    let o_path = Path::new(&temp_output_name);

    let dir = o_path.parent().and_then(|p| p.to_str()).unwrap_or("");
    let ext = o_path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if !ext.eq_ignore_ascii_case("mid") {
        return Err("出力はMIDIファイルにしてください。".into());
    }

    // 名前に重複があれば (1),(2) のように名前を変えていく
    for i in 0..100 {
        if i > 0 {
            let stem = o_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
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
