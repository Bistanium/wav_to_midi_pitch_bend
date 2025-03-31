from dataclasses import dataclass
from math import log10, modf, sqrt
from pathlib import Path
import sys
import tkinter
import tkinter.filedialog
import wave

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
import scipy
from tqdm import tqdm


# 設定ゾーン  不正な入力のチェックをしていないので変更には気をつける
class Settings:
    # ピッチベンドを無効にするか
    disable_pitch_bend = False
    # 近い音量のノートを繋げる, 大きくすると軽量になる, -1で無効化
    similar_velocity_threshold = 4
    # 3以上推奨, 大きくすると軽量になる
    minimum_velocity = 6
    # カイザー窓のβ値  β=πα
    window_beta = 8
    # %を表記しない
    overlap_rate = 50


class About_tracks:
    if Settings.disable_pitch_bend:
        note_digit = 1
        number_of_tracks = 1
    else:
        note_digit = 10
        number_of_tracks = 10


@dataclass(slots=True)
class NoteMessage:
    type: str
    note: int
    velocity: float
    channel: int


# midi化関数
def fft2midi(F, before_volume_list, fs, N, start_note, end_note, append_pitch_bend, next_time):

    TWELFTH_ROOT_OF_2 = 1.059463094359295264561825294946
    TWELVE_OVER_LOG10_2 = 39.86313713864834817444383315387
    LOG10_440_TIMES_TOL_MINUS_69 = 36.37631656229591524883618971458

    current_volume_list = [0] * 1280
    number_of_tracks = About_tracks.number_of_tracks
    tracks = [[] for _ in range(number_of_tracks)]
    sec = N / fs
    before_note = 0.0
    sum_volume = 0.0
    # 1.05~1.1乗ぐらいがいい音に聞こえる。小さくすると高音が刺さる
    volumes = (np.abs(F) / N * 2) ** 1.05

    range_start = int(sec * 440 * TWELFTH_ROOT_OF_2 ** (start_note - 69 - 1))
    range_end = int(sec * 440 * TWELFTH_ROOT_OF_2 ** (end_note - 69 + 1))

    # note_digitは使うトラックの数ともいえる
    note_digit = About_tracks.note_digit

    for i in range(range_start, range_end):

        volume = volumes[i]
        if volume < 3:
            continue

        # ノート番号計算, i/secは周波数
        note = log10(i / sec) * TWELVE_OVER_LOG10_2 - LOG10_440_TIMES_TOL_MINUS_69
        round_1_note = int(note * note_digit + 0.5) / note_digit

        # 音が変わったら前の音階をmidiに打ち込む
        if before_note != round_1_note:
            round_0_note = int(before_note + 0.5)
            if round_0_note <= 127:
                midi_note_decimal, midi_note_int = modf(before_note)
                track_num = int(midi_note_decimal * 10 + 0.5)
                # なぜ平方根が必要かはわからない
                current_volume_list[track_num * 128 + round_0_note] = sqrt(sum_volume)

            before_note = round_1_note
            sum_volume = volume
        else:
            sum_volume += volume


    similar_velocity_threshold = Settings.similar_velocity_threshold
    disable_pitch_bend = Settings.disable_pitch_bend
    for i, track in enumerate(tracks):
        # 0から数えて9つ目は打楽器
        ch = i if i != 9 else 10
        if append_pitch_bend and not disable_pitch_bend:
            track.append(NoteMessage(type="pitchwheel", note=-1, velocity=-1, channel=ch))
        for j in range(start_note, end_note):
            note_idx = i * 128 + j
            before_volume = before_volume_list[note_idx]
            current_volume = current_volume_list[note_idx]

            is_below_min = current_volume <= 2
            # 前回音があったとき
            if before_volume != 0:
                is_small = current_volume < before_volume - similar_velocity_threshold 
                is_big = before_volume + similar_velocity_threshold < current_volume
                # 音量変化が指定した値より大きいとき
                if is_small or is_big or is_below_min:
                    track.append(NoteMessage(type="note_off", note=j, velocity=-1, channel=ch))
                    if not is_below_min:
                        track.append(NoteMessage(type="note_on", note=j, velocity=current_volume, channel=ch))
                    else:
                        current_volume_list[note_idx] = 0
                else:
                    current_volume_list[note_idx] = before_volume
            # 閾値超過のとき
            elif not is_below_min:
                track.append(NoteMessage(type="note_on", note=j, velocity=current_volume, channel=ch))
            else:
                current_volume_list[note_idx] = 0

        if next_time:
            track.append(NoteMessage(type="next_time", note=-1, velocity=-1, channel=ch))

    return current_volume_list, tracks


def choose_wav_file():
    # ファイル選択
    while True:
        fTyp = [("Audio File", ".wav"), ("wav", ".wav")]
        input_name = tkinter.filedialog.askopenfilename(filetypes = fTyp)
        input_path_obj = Path(input_name)

        if input_name:
            extension = input_path_obj.suffix
            if extension == ".wav":
                break
            else:
                sys.exit("ファイルが正しくありません")
        else:
            sys.exit()

    # ファイル名重複チェック
    output_path_obj = input_path_obj.with_suffix(".mid")
    # ピッチベンドの有無
    if Settings.disable_pitch_bend:
        output_path_obj = input_path_obj.with_name(f"{input_path_obj.stem} [disable pitch bend].mid")
    if output_path_obj.exists():
        base_name = output_path_obj.stem
        for i in range(1, 1000):
            output_path_obj = output_path_obj.with_name(f"{base_name} ({i}).mid")
            if not output_path_obj.exists():
                break
        else:
            sys.exit("ファイル名が重複しすぎているため作成できません")

    return input_path_obj, output_path_obj


# Wave読み込み
def read_wav(file_path):
    with wave.open(str(file_path)) as wf:
        # 全部読み込む
        buf = wf.readframes(-1) 

        # 16bitのときのみ
        if wf.getsampwidth() == 2:
            data = np.frombuffer(buf, dtype=np.int16)
            data = (data / 32768).astype(np.float32)
            fs = wf.getframerate()
        else:
            sys.exit("ビット深度が16bit以外です")

        if wf.getnchannels() == 2:
            mono_data = (data[::2] + data[1::2]) / 2.0
        else:
            mono_data = data

    return mono_data, fs


def definition_midi():
    # midi定義
    number_of_tracks = About_tracks.number_of_tracks
    mid = MidiFile()
    tracks = [MidiTrack() for _ in range(number_of_tracks)]
    mid.tracks.extend(tracks)
    tracks[0].append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(240)))
    tracks[0].append(MetaMessage(
        "time_signature", numerator=4, denominator=4,
        clocks_per_click=24, notated_32nd_notes_per_beat=8
    ))
    # 音色変更用
    #for i, track in enumerate(tracks):
    #    ch = i if i != 9 else 10
    #    # バンク  value=にMSB,LSBの順で
    #    track.append(Message("control_change", channel=ch, control=0, value=0))
    #    track.append(Message("control_change", channel=ch, control=32, value=0))
    #    # program=に0から数えた音色の番号
    #    track.append(Message("program_change", channel=ch, program=0))

    return mid, tracks


# リサンプリング
def resampling(data, fs, target_fs, amp):
    data = data / amp
    # リサンプリング
    data = scipy.signal.resample_poly(data, target_fs, fs)

    return data


def resampling_3(data, fs, new_fs_list):
    resampled_data_list = []
    amp = max(max(data), -(min(data) + 1)) * 1.05
    for i in range(3):
        resampled_data_list.append(
            resampling(data, fs, new_fs_list[i], amp)
        )

    return resampled_data_list


# オーディオ分割
def audio_split(data, win_size):
    len_data = len(data)
    # symはFalseの方がスペクトラム解析に良いらしい
    win = scipy.signal.windows.kaiser(win_size, Settings.window_beta, sym=False)
    win = win.astype(np.float32)

    # 50%→2, 75%→4に変換
    overlap = int(100 / (100 - Settings.overlap_rate) + 0.5)

    step_size = win_size // overlap
    num_segments = (len_data - win_size) // step_size + 1

    indices = np.arange(0, num_segments * step_size, step_size)
    segments_data = np.zeros((num_segments + 1, win_size), dtype=np.int16)

    for idx, start in enumerate(indices):
        end = start + win_size
        segment = data[start:end] * win
        # 16bit化処理
        segment *= 32767
        np.round(segment, out=segment)
        np.clip(segment, -32768, 32767, out=segment)
        segments_data[idx, :] = segment

    return segments_data


def audio_split_3(resampled_datas, window_size_low_list):
    segment_data_list = []
    for i in range(3):
        segment_data_list.append(
            audio_split(resampled_datas[i], window_size_low_list[i])
        )

    return segment_data_list


def normalize_velocity(tracks):
    # すべてのトラックのvelocityの最大値を求める
    velocities = np.array([
        note_event.velocity
        for track in tracks
        for note_event in track
    ])
    max_velocity = velocities.max(initial=1)

    # 正規化係数を計算
    normalize = 127 / max_velocity

    # 閾値以下のノート番号を記録するリスト
    removed_notes = set()
    # これ以下のvelocityは無視される
    min_velocity = Settings.minimum_velocity
    # velocityを正規化し、閾値以下の音量のnoteイベントの削除処理
    for track in tracks:
        new_track = []
        for note_event in track:
            if note_event.type == "note_on":
                new_velocity = int(note_event.velocity * normalize + 0.5)
                if new_velocity <= min_velocity:
                    removed_notes.add((note_event.note, note_event.channel))
                    continue
                # velocityを更新
                note_event.velocity = new_velocity
            elif note_event.type == "note_off":
                note = note_event.note
                channnel = note_event.channel
                if (note, channnel) in removed_notes:
                    removed_notes.remove((note, channnel))
                    continue
            new_track.append(note_event)
        # 元のトラック更新
        track[:] = new_track

    return tracks


def create_all_note_messages(N, fs):
    """
    1. ノート番号が0から、velocityが0から127まで、ノート番号分作り、それをさらにチャンネル分(9を除く)作る
    2. ノート番号が0から127まで、チャンネル分(9を除く)作る
    3. note_offイベントのtimeをチャンネル分(9を除く)作る
    4. ピッチベンドをチャンネル分(9を除く)作る
    """
    # 1.(163,840) + 2.(1,280) + 3.(10) + 4.(10) = 165,140
    all_note_messages_list = np.zeros(165140, dtype=object)

    # 1
    start_idx = 0
    for i in range(10): # channel
        ch = i if i != 9 else 10
        for j in range(128): # note number
            for k in range(128): # velocity
                all_note_messages_list[16384 * i + 128 * j + k] = (
                    Message("note_on", note=j, velocity=k, channel=ch)
                )
    start_idx += 163840

    # 2
    for i in range(10): # channel
        ch = i if i != 9 else 10
        for j in range(128): # note number
            all_note_messages_list[128 * i + j + start_idx] = (
                Message("note_off", note=j, channel=ch)
            )
    start_idx += 1280

    # 3
    sec = N / fs
    overlap = int(100 / (100 - Settings.overlap_rate) + 0.5)
    t = int(240 * sec / overlap + 0.5)
    for i in range(10): # channel
        ch = i if i != 9 else 10
        all_note_messages_list[i + start_idx] = (
            Message("note_off", channel=ch, time=t)
        )
    start_idx += 10

    # 4
    for i in range(10): # channel
        ch = i if i != 9 else 10
        bend_values = [0, 410, 819, 1229, 1638, -2048, -1638, -1229, -819, -410]
        all_note_messages_list[i + start_idx] = (
            Message("pitchwheel", channel=ch, pitch=bend_values[i])
        )
    start_idx += 10

    return all_note_messages_list


def append_tracks(tracks, temp_track, all_note_messages_list):
    for track in temp_track:
        for note_event in track:
            track_num = min(note_event.channel, 9)
            if note_event.type == "note_on":
                tracks[track_num].append(
                    all_note_messages_list[16384 * track_num + 128 * note_event.note + note_event.velocity]
                )
            elif note_event.type == "note_off":
                tracks[track_num].append(
                    all_note_messages_list[128 * track_num + note_event.note + 163840]
                )
            elif note_event.type == "next_time":
                tracks[track_num].append(
                    all_note_messages_list[track_num + 165120]
                )
            elif note_event.type == "pitchwheel":
                tracks[track_num].append(
                    all_note_messages_list[track_num + 165130]
                )


def main():

    # Wavファイル選択
    input_path_obj, output_path_obj = choose_wav_file()

    print("\ninput:", input_path_obj.name)
    print("output:", output_path_obj.name)

    # Wav読み込み
    data, fs = read_wav(input_path_obj)

    print("\n再サンプリング&データ分割中…… (1/5)\n")

    # リサンプリング
    new_fs_high = 40960
    new_fs_middle = 10240
    new_fs_low = 640
    new_fs_list = (new_fs_high, new_fs_middle, new_fs_low)
    resampled_data_list = resampling_3(data, fs, new_fs_list)

    # オーディオ分割
    window_size_high = 4096
    window_size_middle = 2048  # 時間分解能がhighの1/2
    window_size_low = 256      # 時間分解能がhighの1/4
    window_size_low_list = (window_size_high, window_size_middle, window_size_low)
    segment_data_list = audio_split_3(resampled_data_list, window_size_low_list)
    segments_data_high, segments_data_middle, segments_data_low = segment_data_list

    del data, resampled_data_list, segment_data_list

    # ここから変換の準備
    before_volume_list_high = [0] * 1280
    before_volume_list_middle = [0] * 1280
    before_volume_list_low = [0] * 1280

    # loopの長さ
    segments_data_range = len(segments_data_low) * 4

    # FFT&midiデータ化
    temp_track = []
    for i in tqdm(range(0, segments_data_range), desc="Convert to MIDI (2/5)"):
        # 低音用
        if i % 4 == 0:
            ffted_data_low = scipy.fft.fft(segments_data_low[i // 4])
            before_volume_list_low, part_of_tracks = fft2midi(
                F=ffted_data_low, before_volume_list=before_volume_list_low,
                fs=new_fs_low, N=window_size_low,
                start_note=36, end_note=60,
                append_pitch_bend=False, next_time=False
            )
            temp_track.extend(part_of_tracks)

        # 中音用
        if i % 2 == 0:
            ffted_data_middle = scipy.fft.fft(segments_data_middle[i // 2])
            before_volume_list_middle, part_of_tracks = fft2midi(
                F=ffted_data_middle, before_volume_list=before_volume_list_middle,
                fs=new_fs_middle, N=window_size_middle,
                start_note=60, end_note=108,
                append_pitch_bend=True, next_time=False
            )
            temp_track.extend(part_of_tracks)

        # 高音用
        ffted_data_high = scipy.fft.fft(segments_data_high[i])
        before_volume_list_high, part_of_tracks = fft2midi(
            F=ffted_data_high, before_volume_list=before_volume_list_high,
            fs=new_fs_high, N=window_size_high,
            start_note=108, end_note=128,
            append_pitch_bend=False, next_time=True
        )
        temp_track.extend(part_of_tracks)

    del segments_data_high, segments_data_middle, segments_data_low

    # midi定義
    mid, tracks = definition_midi()

    print("\n正規化処理中…… (3/5)")

    normalized_tracks = normalize_velocity(temp_track)

    print("\nMIDIメッセージ生成中…… (4/5)")

    all_note_messages_list = create_all_note_messages(window_size_high, new_fs_high)

    append_tracks(tracks, normalized_tracks, all_note_messages_list)

    del temp_track, normalized_tracks, all_note_messages_list

    print("\nMIDI保存中…… (5/5)")

    mid.save(output_path_obj)


if __name__ == "__main__":
    main()
