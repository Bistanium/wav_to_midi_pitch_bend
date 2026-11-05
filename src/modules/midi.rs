use super::fft2midi::Fft2midi;
use crate::modules::settings::Settings;
use midly::{
    Format, Header, MetaMessage, MidiMessage, PitchBend, Smf, Timing, TrackEvent, TrackEventKind,
    num::{u4, u7, u15, u24, u28},
};

pub struct Midi {
    tracks: Vec<Vec<TrackEvent<'static>>>,
    pitch_bend_values: Vec<i16>,
    number_of_tracks: usize,
    time: u32,
}

impl Midi {
    pub fn new(settings: &Settings) -> Self {
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

        let step = 4096 / number_of_tracks;
        let pitch_bend_values = generate_bend_values(step);

        // 高周波帯の値を用いて以下の式で計算可能
        // (1920.0 * sec * hop_size as f64 / win_size as f64 + 0.5) as u32
        let time = 12;

        Self {
            tracks,
            pitch_bend_values,
            number_of_tracks,
            time,
        }
    }

    pub fn export(&mut self, name: String) {
        // 最後にnote_offメッセージがない可能性があるため
        for i in 0..self.number_of_tracks as u8 {
            self.push("note_off", i, 0, 0, self.time);
            for j in 1..128 {
                self.push("note_off", i, j, 0, 0);
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

    fn push(&mut self, message: &str, channel: u8, note: u8, velocity: u8, time: u32) {
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
                        bend: PitchBend::from_int(self.pitch_bend_values[channel as usize]),
                    },
                },
            },
            // messageがプログラム内部で固定のため
            &_ => unreachable!(),
        };
        self.tracks[channel as usize].push(event);
    }

    fn manage_notes(
        &mut self,
        f2m: &mut Fft2midi,
        frame: usize,
        normalize_value: f64,
        settings: &Settings,
        append_pitch_bend: bool,
        next_time: bool,
    ) {
        let min_velocity = settings.minimum_velocity as u8;
        let similar_velocity_threshold = settings.similar_velocity_threshold as u8;
        let disable_pitch_bend = settings.disable_pitch_bend;

        let (start_note, end_note) = f2m.get_start_end_note();
        let note_nums = end_note - start_note;
        let (start_note_u8, end_note_u8) = (start_note as u8, end_note as u8);

        for i in 0..self.number_of_tracks as u8 {
            if append_pitch_bend && !disable_pitch_bend {
                self.push("pitchwheel", i, 0, 0, 0);
            }
            for j in start_note_u8..end_note_u8 {
                let idx = i as usize * note_nums + j as usize - start_note;
                let before_volume =
                    (f2m.get_temp_track()[frame - 1][idx] * normalize_value + 0.5) as u8;
                let current_volume =
                    (f2m.get_temp_track()[frame][idx] * normalize_value + 0.5) as u8;

                // 前回音がなかったとき
                if before_volume == 0 {
                    // 今回は音があったとき
                    if current_volume > min_velocity {
                        self.push("note_on", i, j, current_volume, 0);
                    } else {
                        f2m.set_temp_track(frame, idx, 0.0);
                    }
                } else {
                    let is_smaller = current_volume + similar_velocity_threshold < before_volume;
                    let is_biger = before_volume + similar_velocity_threshold < current_volume;
                    if is_smaller || is_biger || current_volume <= min_velocity {
                        self.push("note_off", i, j, 0, 0);
                        if current_volume > min_velocity {
                            self.push("note_on", i, j, current_volume, 0);
                        } else {
                            f2m.set_temp_track(frame, idx, 0.0);
                        }
                    } else {
                        f2m.set_temp_track(frame, idx, f2m.get_temp_track()[frame - 1][idx]);
                    }
                }
            }

            if next_time {
                self.push("note_off", i, 0, 0, self.time);
            }
        }
    }

    pub fn normalize_velocity(
        &mut self,
        mut low_f2m: Fft2midi,
        mut mid_f2m: Fft2midi,
        mut high_f2m: Fft2midi,
        segments_data_range: usize,
        settings: &Settings,
    ) {
        // それぞれの最大値を得る
        let max_velocity_low = low_f2m
            .get_temp_track()
            .iter()
            .flatten()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.0);
        let max_velocity_mid = mid_f2m
            .get_temp_track()
            .iter()
            .flatten()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.0);
        let max_velocity_high = high_f2m
            .get_temp_track()
            .iter()
            .flatten()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.0);
        let max_velocity = max_velocity_low
            .max(max_velocity_mid)
            .max(max_velocity_high);

        // 正規化係数
        let normalize = 127.0 / max_velocity;

        // Velocity変化等考慮してMIDIへ書き込む
        for i in 4..segments_data_range {
            if i % 4 == 0 {
                let j = i / 4;
                self.manage_notes(&mut low_f2m, j, normalize, settings, false, false);
            }
            if i % 2 == 0 {
                let j = i / 2;
                self.manage_notes(&mut mid_f2m, j, normalize, settings, true, false);
            }
            self.manage_notes(&mut high_f2m, i, normalize, settings, false, true);
        }
    }
}

fn generate_bend_values(step: usize) -> Vec<i16> {
    let positive_values = (0..2048).step_by(step).map(|x| x as i16);
    let negative_values = (0..2048).step_by(step).map(|x| x as i16 - 2048);
    positive_values.chain(negative_values).collect()
}
