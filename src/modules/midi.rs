use midly::{
    Format, Header, MetaMessage, MidiMessage, PitchBend, Smf, Timing, TrackEvent, TrackEventKind,
    num::{u4, u7, u15, u24, u28},
};

use crate::modules::settings::Settings;

use super::midi_from_wav::Track;

pub struct Midi {
    tracks: Vec<Vec<TrackEvent<'static>>>,
    pitch_bend_values: Vec<i16>,
    number_of_tracks: usize,
    time: u32,
    normalize: f64,
    settings: Settings,
}

impl Midi {
    pub fn new(settings: Settings) -> Self {
        let number_of_tracks = settings.number_of_tracks;
        let mut tracks = vec![vec![]; number_of_tracks];

        // 拍子
        tracks[0].push(TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::TimeSignature(4, 2, 24, 8)),
        });
        // 240BPM
        // 6*10^7/BPM
        tracks[0].push(TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(250_000))),
        });

        let pitch_bend_values = generate_bend_values(number_of_tracks);

        // 高域の値を用いて以下の式で計算可能
        // (1920.0 * hop_size as f64 / sr as f64).round() as u32
        let time = 12;

        Self {
            tracks,
            pitch_bend_values,
            number_of_tracks,
            time,
            normalize: 1.0,
            settings,
        }
    }

    pub fn export(&mut self, name: String) {
        // 最後にnote_offメッセージがない可能性があるため
        for i in 0..self.number_of_tracks {
            let ch: usize = if i < 9 { i } else { i + 1 };
            self.push("note_off", ch, 0, 0, self.time);
            for j in 1..128 {
                self.push("note_off", ch, j, 0, 0);
            }
        }
        // MIDIトラック終了
        let event = TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        };
        for track in &mut self.tracks {
            track.push(event);
        }

        let smf = Smf {
            header: Header {
                format: Format::Parallel,
                // 時間の基準
                timing: Timing::Metrical(u15::new(480)),
            },
            tracks: self.tracks.clone(),
        };

        smf.save(name).expect("Failed to export MIDI");
    }

    fn push(&mut self, message: &str, channel: usize, note: usize, velocity: u32, time: u32) {
        // 入力値チェック
        let channel = channel.clamp(0, 15) as u8;
        let note = note.clamp(0, 127) as u8;
        let velocity = velocity.clamp(0, 127) as u8;
        // 2^28-1
        let time = time.clamp(0, 268_435_455);

        // channelは10番目のチャンネル回避で10番目以降が+1されているため
        let i = if channel < 9 { channel } else { channel - 1 };
        let i = i as usize;

        let event = match message {
            "note_on" => TrackEvent {
                delta: u28::new(0),
                kind: TrackEventKind::Midi {
                    channel: u4::new(channel),
                    message: MidiMessage::NoteOn {
                        key: u7::new(note),
                        vel: u7::new(velocity),
                    },
                },
            },
            "note_off" => TrackEvent {
                delta: u28::new(time),
                kind: TrackEventKind::Midi {
                    channel: u4::new(channel),
                    message: MidiMessage::NoteOff {
                        key: u7::new(note),
                        vel: u7::new(0),
                    },
                },
            },
            "pitchwheel" => TrackEvent {
                delta: u28::new(0),
                kind: TrackEventKind::Midi {
                    channel: u4::new(channel),
                    message: MidiMessage::PitchBend {
                        bend: PitchBend::from_int(self.pitch_bend_values[i]),
                    },
                },
            },
            // messageがプログラム内部で固定のため
            _ => unreachable!(),
        };

        self.tracks[i].push(event);
    }

    fn manage_notes(
        &mut self,
        track: &mut Track,
        frame: usize,
        append_pitch_bend: bool,
        next_time: bool,
    ) {
        let min_velocity = self.settings.minimum_velocity as u32;
        let similar_velocity_threshold = self.settings.similar_velocity_threshold as u32;
        let disable_pitch_bend = self.settings.disable_pitch_bend;
        let gain = self.settings.gain;

        let start_note = track.start_note;
        let end_note = track.end_note;

        for i in 0..self.number_of_tracks {
            let ch: usize = if i < 9 { i } else { i + 1 };
            if append_pitch_bend && !disable_pitch_bend {
                self.push("pitchwheel", ch, 0, 0, 0);
            }
            for j in start_note..end_note {
                let idx = i * (end_note - start_note) + j - start_note;

                let before_velocity = track.temp_track[frame - 1][idx];
                let before_velocity = before_velocity.round() as u32;

                let current_velocity =
                    (track.temp_track[frame][idx] * self.normalize * gain).sqrt();
                // 次のフレームで流用するので更新しておく
                track.temp_track[frame][idx] = current_velocity;
                let current_velocity = current_velocity.round() as u32;

                // 前回音がなかったとき
                if before_velocity == 0 {
                    // 今回は音があったとき
                    if min_velocity < current_velocity {
                        self.push("note_on", ch, j, current_velocity, 0);
                    } else {
                        track.temp_track[frame][idx] = 0.0;
                    }
                } else {
                    //　今回の音が前回よりも小さくなりすぎたとき
                    let is_smaller =
                        current_velocity + similar_velocity_threshold < before_velocity;
                    // 今回の音が前回よりも大きくなりすぎたとき
                    let is_biger = before_velocity + similar_velocity_threshold < current_velocity;
                    if is_smaller || is_biger || current_velocity <= min_velocity {
                        self.push("note_off", ch, j, 0, 0);
                        if min_velocity < current_velocity {
                            self.push("note_on", ch, j, current_velocity, 0);
                        } else {
                            track.temp_track[frame][idx] = 0.0;
                        }
                    } else {
                        let velocity = track.temp_track[frame - 1][idx];
                        track.temp_track[frame][idx] = velocity;
                    }
                }
            }

            if next_time {
                self.push("note_off", ch, 0, 0, self.time);
            }
        }
    }

    pub fn normalize_velocity(&mut self, track: &mut [Track], segments_data_range: usize) {
        // それぞれから最大値を得る
        let max_velocity = get_max_velocity(&track[0])
            .max(get_max_velocity(&track[1]))
            .max(get_max_velocity(&track[2]))
            .max(get_max_velocity(&track[3]));

        // 正規化係数
        self.normalize = 127.0 * 127.0 / max_velocity;

        // Velocity変化等考慮してMIDIへ書き込む
        for i in 8..segments_data_range {
            if i % 8 == 0 {
                let j = i / 8;
                self.manage_notes(&mut track[0], j, false, false);
            }
            if i % 4 == 0 {
                let j = i / 4;
                self.manage_notes(&mut track[1], j, false, false);
            }
            if i % 2 == 0 {
                let j = i / 2;
                self.manage_notes(&mut track[2], j, true, false);
            }
            self.manage_notes(&mut track[3], i, false, true);
        }
    }
}

fn get_max_velocity(track: &Track) -> f64 {
    track
        .temp_track
        .iter()
        .flatten()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(1.0)
}

fn generate_bend_values(nums: usize) -> Vec<i16> {
    assert!(nums > 0 && nums % 2 == 0);

    let half = nums / 2;

    let positive_values = (0..half)
        .map(|i| (i as f64 * 2048.0 / half as f64).round() as i16);

    let negative_values = (0..half)
        .map(|i| (i as f64 * 2048.0 / half as f64).round() as i16 - 2048);

    positive_values.chain(negative_values).collect()
}