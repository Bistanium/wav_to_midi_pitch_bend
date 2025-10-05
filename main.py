from dataclasses import dataclass
from math import modf, sqrt
from pathlib import Path
import sys
import tkinter
import tkinter.filedialog

import mido
import numpy as np
import resampy
import scipy
import soundfile as sf
from tqdm import tqdm


# 設定ゾーン  不正な入力のチェックはしていないから注意
class Settings:
    # ピッチベンドを無効にするか
    disable_pitch_bend = False
    # 近い音量のノートを繋げる, 大きくすると軽量になる, -1で無効化
    similar_velocity_threshold = 4
    # 2以上推奨, 大きくすると軽量になる
    minimum_velocity = 4
    # %を表記しない
    overlap_rate = 75
    # 0埋め倍数
    pad_multiple = 2


class AboutTracks:
    note_digit = 10
    number_of_tracks = 10
    @classmethod
    def update_settings(cls):
        cls.note_digit = 1
        cls.number_of_tracks = 1


@dataclass(slots=True)
class NoteMessage:
    type: str
    note: int
    velocity: float
    channel: int


def WrapToPi(phases):
    PI = np.pi
    return (phases + PI) % (2 * PI) - PI


def estimate_frequency(theta_prev, theta, N, pad_N, fs):
    overlap = int(100 / (100 - Settings.overlap_rate) + 0.5)
    # フレーム間隔
    delta_t = int(N / overlap + 0.5)
    # 各周波数ビンに対応する角周波数
    omega = 2 * np.pi * np.arange(0, pad_N) / pad_N
    # フレーム間の周波数の位相の変化と対応する周波数のビンから計算される位相の変化との差
    delta_phi = WrapToPi(theta - theta_prev - omega * delta_t)
    # omegaを補正して位相の変化がより正確になるようにする
    true_omega = omega + delta_phi / delta_t
    # omegaの式より真の周波数ビン(np.arrangeの値)を得る
    f_bin = true_omega * pad_N / (2 * np.pi)
    # 周波数ビンを周波数に変換
    freq = f_bin / pad_N * fs
    # 対数がエラーにならないように
    np.maximum(freq, 1, out=freq)
    # ノート番号を求める用
    log10_f_n = np.log10(freq)

    return log10_f_n


# midi化関数
def fft2midi(angel_F_prev, F, before_volume_list, fs, N, start_note, end_note, append_pitch_bend, next_time):

    TWELFTH_ROOT_OF_2 = 1.059463094359295264561825294946
    TWELVE_OVER_LOG10_2 = 39.86313713864834817444383315387
    LOG10_440_TIMES_TOL_MINUS_69 = 36.37631656229591524883618971458

    current_volume_list = [0] * 1280
    number_of_tracks = AboutTracks.number_of_tracks
    tracks = [[] for _ in range(number_of_tracks)]
    pad_N = N * Settings.pad_multiple
    sec = pad_N / fs
    before_note = 0.0
    sum_volume = 0.0
    volumes = np.abs(F) / pad_N

    # 位相差からより正確な周波数(ノート番号)を得る
    angle_F = np.angle(F)
    truth_notes = estimate_frequency(angel_F_prev, angle_F, N, pad_N, fs)

    # 境界の±1ぐらいから開始してmidiノート周辺に対応するビンの振幅を拾えるようにする
    range_start = int(sec * 440 * TWELFTH_ROOT_OF_2 ** (start_note - 69 - 1))
    range_end = int(sec * 440 * TWELFTH_ROOT_OF_2 ** (end_note - 69 + 1))

    # note_digitは使うトラックの数ともいえる
    note_digit = AboutTracks.note_digit

    for i in range(range_start, range_end):
        volume = volumes[i]
        # 音量が小さすぎるときは無視する
        if volume < 2:
            continue

        # ノート番号計算
        note = truth_notes[i] * TWELVE_OVER_LOG10_2 - LOG10_440_TIMES_TOL_MINUS_69
        round_1_note = int(note * note_digit + 0.5) / note_digit

        # 音が変わったら前の音階をmidiに打ち込む
        if before_note != round_1_note:
            round_0_note = int(before_note + 0.5)
            if round_0_note <= 127:
                midi_note_decimal, _ = modf(before_note)
                track_num = int(midi_note_decimal * 10 + 0.5)
                # 平方根をとると音がいい感じになる
                current_volume_list[track_num * 128 + round_0_note] = sqrt(sum_volume)

            before_note = round_1_note
            sum_volume = volume
        else:
            sum_volume += volume


    similar_velocity_threshold = Settings.similar_velocity_threshold
    disable_pitch_bend = Settings.disable_pitch_bend
    for i, track in enumerate(tracks):
        # 0から数えて9つ目は打楽器のため
        ch = i if i != 9 else 10
        if append_pitch_bend and not disable_pitch_bend:
            track.append(NoteMessage(type="pitchwheel", note=-1, velocity=-1, channel=ch))
        for j in range(start_note, end_note):
            note_idx = i * 128 + j
            before_volume = before_volume_list[note_idx]
            current_volume = current_volume_list[note_idx]

            # 定数部分が0以上でないとmidiが壊れる
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

        # 別関数として分けるか考え中
        if next_time:
            track.append(NoteMessage(type="next_time", note=-1, velocity=-1, channel=ch))

    return current_volume_list, tracks, angle_F


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
        for i in range(1, 100):
            output_path_obj = output_path_obj.with_name(f"{base_name} ({i}).mid")
            if not output_path_obj.exists():
                break
        else:
            sys.exit("ファイル名が重複しすぎているため作成できません")

    return input_path_obj, output_path_obj


# Wave読み込み
def read_wav(file_path):
    # サウンドファイルを読み込む
    data, samplerate = sf.read(file_path, dtype=np.float32)

    # ステレオの場合、チャンネルを合成
    if data.ndim == 2:
        mono_data =(data[:, 0] + data[:, 1]) / 2.0
    else:
        mono_data = data

    return mono_data, samplerate


def definition_midi():
    # midi定義
    number_of_tracks = AboutTracks.number_of_tracks
    mid = mido.MidiFile()
    tracks = [mido.MidiTrack() for _ in range(number_of_tracks)]
    mid.tracks.extend(tracks)
    tracks[0].append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(240)))
    tracks[0].append(mido.MetaMessage(
        "time_signature", numerator=4, denominator=4,
        clocks_per_click=24, notated_32nd_notes_per_beat=8
    ))

    return mid, tracks


# リサンプリング
def resampling(data, fs, target_fs, amp):
    data = data / amp
    # リサンプリング
    return resampy.resample(data, sr_orig=fs, sr_new=target_fs, filter='sinc_window', num_zeros=32)


def resampling_3(data, fs, new_fs_list):
    resampled_data_list = []
    # リサンプリングによって-1~1に波形が収まらない可能性を考慮
    # minの方は+1をしておかないとエラーになる場合がある
    amp = max(max(data), -(min(data) + 1)) * 1.05
    # ループは波形の分割数
    for i in range(3):
        resampled_data_list.append(
            resampling(data, fs, new_fs_list[i], amp)
        )

    return resampled_data_list


def convert_16_bit(data):
    # dataは-1~1と仮定
    data *= 32767
    np.round(data, out=data)
    np.clip(data, -32768, 32767, out=data)

    return data.astype(np.int16)

    
# オーディオ分割
def audio_split(data, win_size):
    padded_data = data
    len_data = len(padded_data)
    # symはFalseの方がスペクトラム解析に良いらしい
    win = scipy.signal.windows.hann(win_size, sym=False).astype(np.float32)

    # 50%→2, 75%→4に変換
    overlap = int(100 / (100 - Settings.overlap_rate) + 0.5)

    step_size = win_size // overlap
    num_segments = (len_data - win_size) // step_size + 1

    indices = np.arange(0, num_segments * step_size, step_size)
    segments_data = np.zeros((num_segments + 1, win_size * Settings.pad_multiple), dtype=np.int16)

    pad_size = win_size * (Settings.pad_multiple - 1)
    for idx, start in enumerate(indices):
        end = start + win_size
        segment = padded_data[start:end] * win
        padded_segment = np.pad(segment, (0, pad_size), mode='constant', constant_values=0)
        segments_data[idx, :] = convert_16_bit(padded_segment)

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

    # 正規化係数
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
                if new_velocity > 127:
                    new_velocity = 127
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
                    mido.Message("note_on", note=j, velocity=k, channel=ch)
                )
    start_idx += 163840

    # 2
    for i in range(10): # channel
        ch = i if i != 9 else 10
        for j in range(128): # note number
            all_note_messages_list[128 * i + j + start_idx] = (
                mido.Message("note_off", note=j, channel=ch)
            )
    start_idx += 1280

    # 3
    sec = N / fs
    overlap = int(100 / (100 - Settings.overlap_rate) + 0.5)
    t = int(960 * sec / overlap + 0.5)
    for i in range(10): # channel
        ch = i if i != 9 else 10
        all_note_messages_list[i + start_idx] = (
            mido.Message("note_off", channel=ch, time=t)
        )
    start_idx += 10

    # 4
    for i in range(10): # channel
        ch = i if i != 9 else 10
        bend_values = [0, 410, 819, 1229, 1638, -2048, -1638, -1229, -819, -410]
        all_note_messages_list[i + start_idx] = (
            mido.Message("pitchwheel", channel=ch, pitch=bend_values[i])
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
    # ビッチベンドを使うかによって設定を変更
    if Settings.disable_pitch_bend:
        AboutTracks.update_settings()

    # Wavファイル選択
    input_path_obj, output_path_obj = choose_wav_file()

    print("\ninput:", input_path_obj.name)
    print("output:", output_path_obj.name)

    # Wav読み込み
    data, fs = read_wav(input_path_obj)

    # 最後の方まで音があるとmidiで音が残ったままになるため
    new_data = np.pad(data, (4096, 4096), mode='constant', constant_values=0)

    print("\n再サンプリング&データ分割中…… (1/5)\n")

    # リサンプリング
    new_fs_high = 40960
    new_fs_mid = 10240
    new_fs_low = 640
    new_fs_list = (new_fs_high, new_fs_mid, new_fs_low)
    resampled_data_list = resampling_3(new_data, fs, new_fs_list)

    # オーディオ分割
    window_size_high = 2048
    window_size_mid = 1024 # 時間分解能がhighの1/2
    window_size_low = 128      # 時間分解能がhighの1/4
    window_size_list = (window_size_high, window_size_mid, window_size_low)
    segment_data_list = audio_split_3(resampled_data_list, window_size_list)
    segments_data_high, segments_data_mid, segments_data_low = segment_data_list

    del data, resampled_data_list, segment_data_list

    # 変換の準備
    before_volume_list_high = [0] * 1280
    before_volume_list_mid = [0] * 1280
    before_volume_list_low = [0] * 1280
    # loopの長さ
    segments_data_range = len(segments_data_low) * 4
    
    # それぞれループのインデックスが4のときの一つ前の配列を指定
    angle_F_low = np.angle(scipy.fft.fft(segments_data_low[0]))
    angle_F_mid = np.angle(scipy.fft.fft(segments_data_mid[1]))
    angle_F_high = np.angle(scipy.fft.fft(segments_data_high[3]))

    # 保存用
    temp_tracks = []
    for i in tqdm(range(4, segments_data_range), desc="Convert to MIDI (2/5)"):
        # 低音用
        if i % 4 == 0:
            ffted_data_low = scipy.fft.fft(segments_data_low[i // 4])
            before_volume_list_low, part_of_tracks, angle_F_low = fft2midi(
                angel_F_prev=angle_F_low, F=ffted_data_low,
                before_volume_list=before_volume_list_low,
                fs=new_fs_low, N=window_size_low,
                start_note=36, end_note=60,
                append_pitch_bend=False, next_time=False
            )
            temp_tracks.extend(part_of_tracks)

        # 中音用
        if i % 2 == 0:
            ffted_data_mid = scipy.fft.fft(segments_data_mid[i // 2])
            before_volume_list_mid, part_of_tracks, angle_F_mid = fft2midi(
                angel_F_prev=angle_F_mid, F=ffted_data_mid,
                before_volume_list=before_volume_list_mid,
                fs=new_fs_mid, N=window_size_mid,
                start_note=60, end_note=108,
                append_pitch_bend=True, next_time=False
            )
            temp_tracks.extend(part_of_tracks)

        # 高音用
        ffted_data_high = scipy.fft.fft(segments_data_high[i])
        before_volume_list_high, part_of_tracks, angle_F_high = fft2midi(
            angel_F_prev=angle_F_high, F=ffted_data_high,
            before_volume_list=before_volume_list_high,
            fs=new_fs_high, N=window_size_high,
            start_note=108, end_note=128,
            append_pitch_bend=False, next_time=True
        )
        temp_tracks.extend(part_of_tracks)

    del segments_data_high, segments_data_mid, segments_data_low

    # midi定義
    mid, tracks = definition_midi()

    print("\n正規化処理中…… (3/5)")

    normalized_tracks = normalize_velocity(temp_tracks)

    print("\nMIDIメッセージ生成中…… (4/5)")

    all_note_messages_list = create_all_note_messages(window_size_high, new_fs_high)

    append_tracks(tracks, normalized_tracks, all_note_messages_list)

    del temp_tracks, normalized_tracks, all_note_messages_list

    print("\nMIDI保存中…… (5/5)")

    mid.save(output_path_obj)


if __name__ == "__main__":
    main()
