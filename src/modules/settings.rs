use std::fs;
use std::io::ErrorKind;

// ファイルが存在しなかったときに新規作成する設定ファイルの内容
const DEFAULT_CONFIG: &str = "\
    # ピッチベンドを無効にするか ( true / false )
    disable_pitch_bend = false
    # 近い音量のノートを繋げる ( Integers from 0 to 64 inclusive )
    similar_velocity_threshold = 4
    # velocityの最小値 ( Integers from 0 to 127 inclusive )
    minimum_velocity = 4
    # 0埋め倍数 ( Integers from 1 to 16 inclusive )
    pad_multiple = 2
    # 使用するトラックの数 ( Even integers from 2 to 14, inclusive )
    number_of_tracks = 8
";

pub struct Settings {
    pub disable_pitch_bend: bool,
    pub similar_velocity_threshold: usize,
    pub minimum_velocity: usize,
    pub pad_multiple: usize,
    pub number_of_tracks: usize,
}

impl Settings {
    // デフォルト設定
    pub fn default() -> Self {
        Self {
            disable_pitch_bend: false,
            similar_velocity_threshold: 4,
            minimum_velocity: 4,
            pad_multiple: 2,
            number_of_tracks: 8,
        }
    }

    // 設定ファイル読み込み
    pub fn load(path: &str) -> Self {
        // 設定をデフォルト値にする
        let mut settings = Self::default();

        // 設定ファイルの読み込み
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            // ファイルが存在しない場合
            Err(e) if e.kind() == ErrorKind::NotFound => {
                eprintln!("設定ファイルが存在しません。新規作成します: {}", path);

                // タブが不要なので削除する
                let default_config = DEFAULT_CONFIG
                    .lines()
                    .map(|line| line.trim_start())
                    .collect::<Vec<_>>()
                    .join("\n");
                match fs::write(path, default_config) {
                    Ok(()) => {
                        println!("設定ファイルを作成しました");
                    }
                    Err(e) => {
                        eprintln!("設定ファイルを作成できません: {}", e);
                    }
                }

                settings.apply_constraints();
                return settings;
            }
            // ファイルは存在するが読み込めない場合
            Err(e) => {
                eprintln!(
                    "設定ファイルを読み込めません。デフォルト設定を使用します: {}",
                    e
                );

                settings.apply_constraints();
                return settings;
            }
        };

        // 設定ファイルを1行ずつ処理する
        for line in content.lines() {
            let line = line.split('#').next().unwrap().trim();

            if line.is_empty() {
                continue;
            }

            let (key, value) = match line.split_once('=') {
                Some(v) => v,
                None => {
                    eprintln!("{}: 不正な形式です", line);
                    continue;
                }
            };

            // キーと値の前後にある空白を取り除く
            let key = key.trim();
            let value = value.trim();

            // 値のチェックをする
            match key {
                "disable_pitch_bend" => match value {
                    "true" => settings.disable_pitch_bend = true,
                    "false" => settings.disable_pitch_bend = false,
                    _ => eprintln!("{}: 不正な値です", key),
                },

                "similar_velocity_threshold" => match value.parse::<usize>() {
                    Ok(v) if check(v, 0, 64) => {
                        settings.similar_velocity_threshold = v;
                    }
                    Ok(_) => {
                        eprintln!("{}: 不正な値です", key);
                    }
                    Err(_) => eprintln!("{}: 不正な値です", key),
                },

                "minimum_velocity" => match value.parse::<usize>() {
                    Ok(v) if check(v, 0, 128) => {
                        settings.minimum_velocity = v;
                    }
                    Ok(_) => {
                        eprintln!("{}: 不正な値です", key);
                    }
                    Err(_) => eprintln!("{}: 不正な値です", key),
                },

                "pad_multiple" => match value.parse::<usize>() {
                    Ok(v) if check(v, 1, 16) => {
                        settings.pad_multiple = v;
                    }
                    Ok(_) => {
                        eprintln!("{}: 不正な値です", key);
                    }
                    Err(_) => eprintln!("{}: 不正な値です", key),
                },

                "number_of_tracks" => match value.parse::<usize>() {
                    Ok(v) if check(v, 2, 14) && is_even(v) => {
                        settings.number_of_tracks = v;
                    }
                    Ok(_) => {
                        eprintln!("{}: 不正な値です", key);
                    }
                    Err(_) => eprintln!("{}: 不正な値です", key),
                },

                _ => eprintln!("{}: 未定義のキーです", key),
            }
        }

        settings.apply_constraints();

        settings
    }

    fn apply_constraints(&mut self) {
        if self.disable_pitch_bend {
            self.number_of_tracks = 1;
        }
    }
}

fn check(value: usize, start: usize, end: usize) -> bool {
    (start..=end).contains(&value)
}

fn is_even(value: usize) -> bool {
    value % 2 == 0
}
