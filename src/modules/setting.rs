use std::fs;

#[derive(Debug, Clone)]
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
            Err(e) => {
                eprintln!("{}", e);
                eprintln!("デフォルト設定を利用します");
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
