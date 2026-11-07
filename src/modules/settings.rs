use std::fs;
use std::io::ErrorKind;

// ファイルが存在しなかったときに新規作成する設定ファイルの内容
const DEFAULT_CONFIG: &str = "\
    # ピッチベンドを無効にするか (true / false)
    disable_pitch_bend = false

    # 近い音量のノートを繋げる (Integers from 0 to 64 inclusive)
    similar_velocity_threshold = 4
    # velocityの最小値 (Integers from 0 to 127 inclusive)
    minimum_velocity = 2

    # 最終的な音量を増減させる (Numbers from 0.0 to 2.0 inclusive)
    gain = 1
    # 下限をノート番号で指定 (Integers from 12 to 128 inclusive)
    start_note = 36
    # 上限をノート番号で指定 (Integers from 12 to 128 inclusive)
    end_note = 128

    # 0埋め倍数 (Integers from 1 to 8 inclusive)
    pad_multiple = 2
    # 使用するトラックの数 (Even integers from 2 to 14, inclusive)
    number_of_tracks = 8
";

pub struct Settings {
    pub disable_pitch_bend: bool,
    pub similar_velocity_threshold: u8,
    pub minimum_velocity: u8,
    pub gain: f64,
    pub start_note: u8,
    pub end_note: u8,
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
            gain: 1.0,
            start_note: 36,
            end_note: 128,
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
                eprintln!("設定ファイルが存在しません: {}", path);

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

        let mut start_note = settings.start_note;
        let mut end_note = settings.end_note;
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
                    _ => error(key),
                },

                "similar_velocity_threshold" => match value.parse::<u8>() {
                    Ok(v) if check(v, 0, 64) => {
                        settings.similar_velocity_threshold = v;
                    }
                    Ok(_) => {
                        error(key);
                    }
                    Err(_) => error(key),
                },

                "minimum_velocity" => match value.parse::<u8>() {
                    Ok(v) if check(v, 0, 127) => {
                        settings.minimum_velocity = v;
                    }
                    Ok(_) => {
                        error(key);
                    }
                    Err(_) => error(key),
                },

                "gain" => match value.parse::<f64>() {
                    Ok(v) if check(v, 0.0, 2.0) => {
                        settings.gain = v;
                    }
                    Ok(_) => {
                        error(key);
                    }
                    Err(_) => error(key),
                },

                "start_note" => match value.parse::<u8>() {
                    Ok(v) if check(v, 12, 128) => {
                        start_note = v;
                    }
                    Ok(_) => {
                        error(key);
                    }
                    Err(_) => error(key),
                },

                "end_note" => match value.parse::<u8>() {
                    Ok(v) if check(v, 12, 128) => {
                        end_note = v;
                    }
                    Ok(_) => {
                        error(key);
                    }
                    Err(_) => error(key),
                },

                "pad_multiple" => match value.parse::<usize>() {
                    Ok(v) if check(v, 1, 8) => {
                        settings.pad_multiple = v;
                    }
                    Ok(_) => {
                        error(key);
                    }
                    Err(_) => error(key),
                },

                "number_of_tracks" => match value.parse::<usize>() {
                    Ok(v) if check(v, 2, 14) && is_even(v) => {
                        settings.number_of_tracks = v;
                    }
                    Ok(_) => {
                        error(key);
                    }
                    Err(_) => error(key),
                },

                _ => eprintln!("{}: 未定義のキーです", key),
            }
        }

        settings.check_notes_range(start_note, end_note);
        settings.apply_constraints();

        settings
    }

    fn apply_constraints(&mut self) {
        if self.disable_pitch_bend {
            self.number_of_tracks = 1;
        }
    }

    fn check_notes_range(&mut self, start_note: u8, end_note: u8) {
        if start_note <= end_note {
            self.start_note = start_note;
            self.end_note = end_note
        } else {
            eprintln!("start_note, end_note: 不正な値です。デフォルト設定を使用します。",);
        }
    }
}

fn check<T: PartialOrd>(value: T, start: T, end: T) -> bool {
    (start..=end).contains(&value)
}

fn is_even(value: usize) -> bool {
    value.is_multiple_of(2)
}

fn error(key: &str) {
    eprintln!("{}: 不正な値です。デフォルト設定を使用します。", key);
}
