import gc
from math import log10, modf, sqrt
from pathlib import Path
import sys
import tkinter, tkinter.filedialog
import wave

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
from numpy.fft import fft
from scipy.signal import resample_poly
from tqdm import tqdm

# midi化関数
def data2midi(F, before_volume_list, fs, N, start_note, end_note, use_bend, next_tick):
    sec = N / fs
    min_vol = 6
    min_vol_2 = min_vol * 2
    before_midi_note, max_volume = 0, 0
    current_volume_list = [[0 for _ in range(128)] for _ in range(10)]

    volumes = (np.abs(F) / N * 4) ** 1.1

    range_start = int(sec * 440 * 1.0594630943593 ** (start_note - 69) * 0.9)
    range_end = int(sec * 440 * 1.0594630943593 ** (end_note - 69) * 1.1)

    for i in range(range_start, range_end):

        volume = volumes[i]
        if volume < min_vol_2 and i != range_end - 1:
            continue

        # ノート番号計算
        midi_note = 69 + log10(i / sec / 440) / 0.025085832971998433
        round_1_midi_note = int(midi_note * 10 + 0.5) / 10

        if before_midi_note != round_1_midi_note: # 音が変わったら前の音階をmidiに打ち込む
            midi_note_syosu, midi_note_seisu = modf(before_midi_note) # 整数部分と小数部分の分離
            track_num = int(midi_note_syosu * 10 + 0.5)

            # 音量調整
            max_volume = sqrt(max_volume) * 0.55
            max_volume = max_volume if max_volume <= 127 else 127

            round_0_midi_note = int(before_midi_note + 0.5)
            round_0_volume = int(max_volume + 0.5)

            if round_0_midi_note <= 127:
                current_volume_list[track_num][round_0_midi_note] = round_0_volume

            before_midi_note = round_1_midi_note
            max_volume = volume

        else:
            max_volume += volume

    sim = 2 # 2~4が良い, -1でノートを繋げる機能無効化
    bend_values = [0, 410, 819, 1229, 1638, -2048, -1638, -1229, -819, -410]
    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        if use_bend:
            track.append(Message('pitchwheel', channel=ch, pitch=bend_values[i]))
        before_volume_list_i = before_volume_list[i]
        current_volume_list_i = current_volume_list[i]
        for j in range(start_note, end_note):
            before_volume_list_i_j = before_volume_list_i[j]
            current_volume_list_i_j = current_volume_list_i[j]
            if before_volume_list_i_j != 0:
                # 音量変化が指定値より大きいor閾値以下のとき
                if (current_volume_list_i_j < before_volume_list_i_j - sim or 
                    before_volume_list_i_j + sim < current_volume_list_i_j or 
                    current_volume_list_i_j < min_vol):
                    track.append(Message('note_off', note=j, channel=ch))
                    if current_volume_list_i_j > min_vol:
                        track.append(Message('note_on', note=j, velocity=current_volume_list_i_j, channel=ch))
                    else:
                        current_volume_list_i[j] = 0
                else:
                    current_volume_list_i[j] = before_volume_list_i_j
            else: # 前の音がなかったとき
                if current_volume_list_i_j > min_vol:
                    track.append(Message('note_on', note=j, velocity=current_volume_list_i_j, channel=ch))
                else:
                    current_volume_list_i[j] = 0
        current_volume_list[i] = current_volume_list_i

        if next_tick:
            track.append(Message('note_off', channel=ch, time=int(60 * sec + 0.5)))

    return current_volume_list


# Wave読み込み
def read_wav(file_path):
    wf = wave.open(file_path, "rb")
    buf = wf.readframes(-1) # 全部読み込む

    # 16bitのときのみ
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype=np.int16)
        fs = wf.getframerate()
    else:
        sys.exit("ビット深度が16bit以外です")

    if wf.getnchannels() == 2:
        mono_data = (data[::2] + data[1::2]) / 2
    else:
        mono_data = data
    wf.close()

    return mono_data, fs


# データ分割
def audio_split(data, win_size, overlap=2):
    len_data = len(data)
    win = np.hanning(win_size)

    step_size = win_size // overlap
    num_segments = (len_data - win_size) // step_size + 1

    # 各セグメントの開始インデックス
    indices = np.arange(0, num_segments * step_size, step_size)
    splited_data = np.zeros((num_segments, win_size), dtype=np.int16)

    for idx, start in enumerate(indices):
        end = start + win_size
        segment = data[start:end]
        win_segment = segment * win
        splited_data[idx, :] = win_segment

    return splited_data


def return_amp(data):
    data_max_val = max(np.max(data), -(np.min(data) + 1))
    amp = data_max_val * 1.1

    return amp


def change_sampling_rate(data, fs, target_fs, amp):
    data = data / amp
    changed_data = resample_poly(data, target_fs, fs)
    changed_data = changed_data * 32767
    changed_data = np.clip(changed_data, -32768, 32767)
    changed_data = changed_data.astype(np.int16)

    return changed_data


if __name__ == '__main__':

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
    if output_path_obj.exists():
        base_name = input_path_obj.stem
        for i in range(1, 1000):
            output_path_obj = output_path_obj.with_name(f"{base_name} ({i}).mid")
            if not output_path_obj.exists():
                break
        else:
            sys.exit("ファイル名が重複しすぎているため作成できません")

    # midi定義
    mid = MidiFile()
    tracks = [MidiTrack() for _ in range(10)]
    mid.tracks.extend(tracks)
    tracks[0].append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(240)))
    tracks[0].append(MetaMessage(
        'time_signature', numerator=4, denominator=4,
        clocks_per_click=24, notated_32nd_notes_per_beat=8
    ))

    # Wav読み込み
    data, fs = read_wav(input_name)

    print("\n再サンプリング&データ分割開始")

    # 再サンプリング
    new_fs_high = 40960
    new_fs_middle = 10240
    new_fs_low = 640

    amp = return_amp(data)
    samped_data_high = change_sampling_rate(data, fs, new_fs_high, amp)
    samped_data_middle = change_sampling_rate(data, fs, new_fs_middle, amp)
    samped_data_low = change_sampling_rate(data, fs, new_fs_low, amp)

    # データ分割
    window_size_high = 8192
    window_size_middle = 4096
    window_size_low = 512

    splited_data_high = audio_split(samped_data_high, window_size_high)
    splited_data_middle = audio_split(samped_data_middle, window_size_middle)
    splited_data_low = audio_split(samped_data_low, window_size_low)

    del data, samped_data_high, samped_data_middle, samped_data_low
    gc.collect()

    print("再サンプリング&データ分割完了\n")

    before_volume_list_high = [[0 for _ in range(128)] for _ in range(10)]
    before_volume_list_middle = [[0 for _ in range(128)] for _ in range(10)]
    before_volume_list_low = [[0 for _ in range(128)] for _ in range(10)]

    # データの長さ
    len_splited_data_high = len(splited_data_high)
    len_splited_data_middle = len(splited_data_middle)
    len_splited_data_low = len(splited_data_low)

    # 評価用の長さ
    range_data_middle = len_splited_data_middle * 2
    range_data_low = len_splited_data_low * 4

    # FFT&midi化
    for i in tqdm(range(0, len_splited_data_high), desc='Convert to MIDI'):
        # 低音用
        if i < range_data_low and i % 4 == 0:
            ffted_data_low = fft(splited_data_low[i // 4])
            before_volume_list_low = data2midi(
                F=ffted_data_low, before_volume_list=before_volume_list_low,
                fs=new_fs_low, N=window_size_low,
                start_note=36, end_note=60,
                use_bend=False, next_tick=False
            )

        # 中音用
        if i < range_data_middle and i % 2 == 0:
            ffted_data = fft(splited_data_middle[i // 2])
            before_volume_list_middle = data2midi(
                F=ffted_data, before_volume_list=before_volume_list_middle,
                fs=new_fs_middle, N=window_size_middle,
                start_note=60, end_note=108,
                use_bend=True, next_tick=False
            )

        # 高音用
        ffted_data_high = fft(splited_data_high[i])
        before_volume_list_high = data2midi(
            F=ffted_data_high, before_volume_list=before_volume_list_high,
            fs=new_fs_high, N=window_size_high,
            start_note=108, end_note=128,
            use_bend=False, next_tick=True)

    del splited_data_high, splited_data_middle, splited_data_low
    gc.collect()

    print("\nMIDI保存中...")

    mid.save(output_path_obj)
