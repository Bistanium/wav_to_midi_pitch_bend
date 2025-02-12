import gc
from math import log10, modf, sqrt
from pathlib import Path
import sys
import tkinter, tkinter.filedialog
import wave

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
from scipy.fft import fft
from scipy.signal import resample_poly
from tqdm import tqdm

# midi化関数
def data2midi(F, before_volume_list, fs, N, start_note, end_note, use_pitch_bend, next_tick):
    sec = N / fs
    min_vol = 6
    min_vol_2 = min_vol * 2
    before_midi_note = 0
    sum_volume = 0.0
    current_volume_list = [0] * 1280
    volumes = (np.abs(F) / N * 2) ** 1.1

    range_start = int(sec * 440 * 1.059463094359295 ** (start_note - 69) * 0.9)
    range_end = int(sec * 440 * 1.059463094359295 ** (end_note - 69) * 1.1)

    for i in range(range_start, range_end):

        volume = volumes[i]
        if volume < min_vol_2:
            continue

        # ノート番号計算
        midi_note = log10(i / sec) * 39.863137138648348 - 36.376316562295915
        round_1_midi_note = int(midi_note * 10 + 0.5) / 10

        if before_midi_note != round_1_midi_note: # 音が変わったら前の音階をmidiに打ち込む
            midi_note_syosu, midi_note_seisu = modf(before_midi_note)
            track_num = int(midi_note_syosu * 10 + 0.5)

            # 音量調整
            sqrt_volume = sqrt(sum_volume) * 0.8
            round_0_volume = int(sqrt_volume + 0.5)
            if round_0_volume > 127:
                round_0_volume = 127

            round_0_midi_note = int(before_midi_note + 0.5)
            if round_0_midi_note <= 127:
                current_volume_list[track_num * 128 + round_0_midi_note] = round_0_volume

            before_midi_note = round_1_midi_note
            sum_volume = volume
        else:
            sum_volume += volume

    sim = 4 # -1でノートを繋げる機能無効化
    bend_values = [0, 410, 819, 1229, 1638, -2048, -1638, -1229, -819, -410]
    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        if use_pitch_bend:
            track.append(Message('pitchwheel', channel=ch, pitch=bend_values[i]))

        temp_idx = i * 128
        for j in range(start_note, end_note):
            note_idx = temp_idx + j
            before_volume = before_volume_list[note_idx]
            current_volume = current_volume_list[note_idx]

            is_below_min = current_volume <= min_vol

            if before_volume != 0: # 前回音があったとき
                is_small = current_volume < before_volume - sim 
                is_big = before_volume + sim < current_volume

                if is_small or is_big or is_below_min: # 音量変化が指定値より大きいor閾値未満のとき
                    track.append(Message('note_off', note=j, channel=ch))
                    if not is_below_min:
                        track.append(Message('note_on', note=j, velocity=current_volume, channel=ch))
                    else:
                        current_volume_list[note_idx] = 0
                else:
                    current_volume_list[note_idx] = before_volume

            elif not is_below_min: # 閾値以上のとき
                track.append(Message('note_on', note=j, velocity=current_volume, channel=ch))
            else:
                current_volume_list[note_idx] = 0

        if next_tick:
            track.append(Message('note_off', channel=ch, time=int(60 * sec + 0.5)))

    return current_volume_list


# Wave読み込み
def read_wav(file_path):
    wf = wave.open(str(file_path), "rb")
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

    indices = np.arange(0, num_segments * step_size, step_size)
    segments_data = np.zeros((num_segments + 1, win_size), dtype=np.int16)

    for idx, start in enumerate(indices):
        end = start + win_size
        segment = data[start:end]
        win_segment = segment * win
        segments_data[idx, :] = win_segment

    return segments_data


def resampling(data, fs, target_fs, amp):
    normalize_data = data / amp
    resampled_data = resample_poly(normalize_data, target_fs, fs)
    scaled_data = resampled_data * 32767
    cliped_data = np.clip(scaled_data, -32768, 32767)
    result_data = cliped_data.astype(np.int16)

    return result_data


if __name__ == '__main__':

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
    if output_path_obj.exists():
        base_name = input_path_obj.stem
        for i in range(1, 1000):
            output_path_obj = output_path_obj.with_name(f"{base_name} ({i}).mid")
            if not output_path_obj.exists():
                break
        else:
            sys.exit("ファイル名が重複しすぎているため作成できません")

    print("\ninput:", input_path_obj.name)
    print("output:", output_path_obj.name)
    
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
    data, fs = read_wav(input_path_obj)

    print("\n再サンプリング&データ分割開始")

    # 再サンプリング
    new_fs_high = 40960
    new_fs_middle = 10240
    new_fs_low = 640

    amp = max(np.max(data), -(np.min(data) + 1)) * 1.1
    resampled_data_high = resampling(data, fs, new_fs_high, amp)
    resampled_data_middle = resampling(data, fs, new_fs_middle, amp)
    resampled_data_low = resampling(data, fs, new_fs_low, amp)

    # データ分割
    window_size_high = 8192
    window_size_middle = 4096
    window_size_low = 512

    segments_data_high = audio_split(resampled_data_high, window_size_high)
    segments_data_middle = audio_split(resampled_data_middle, window_size_middle)
    segments_data_low = audio_split(resampled_data_low, window_size_low)

    del data, resampled_data_high, resampled_data_middle, resampled_data_low
    gc.collect()

    print("再サンプリング&データ分割完了\n")

    before_volume_list_high = [0] * 1280
    before_volume_list_middle = [0] * 1280
    before_volume_list_low = [0] * 1280

    # データの長さ
    len_splited_data_high = len(segments_data_high)
    len_splited_data_middle = len(segments_data_middle)
    len_splited_data_low = len(segments_data_low)

    # 評価用の長さ
    range_data_middle = len_splited_data_middle * 2
    range_data_low = len_splited_data_low * 4

    # FFT&midi化
    for i in tqdm(range(0, len_splited_data_high), desc='Convert to MIDI'):
        # 低音用
        if i % 4 == 0 and i < range_data_low:
            ffted_data_low = fft(segments_data_low[i // 4])
            before_volume_list_low = data2midi(
                F=ffted_data_low, before_volume_list=before_volume_list_low,
                fs=new_fs_low, N=window_size_low,
                start_note=36, end_note=60,
                use_pitch_bend=False, next_tick=False
            )

        # 中音用
        if i % 2 == 0 and i < range_data_middle:
            ffted_data_middle = fft(segments_data_middle[i // 2])
            before_volume_list_middle = data2midi(
                F=ffted_data_middle, before_volume_list=before_volume_list_middle,
                fs=new_fs_middle, N=window_size_middle,
                start_note=60, end_note=108,
                use_pitch_bend=True, next_tick=False
            )

        # 高音用
        ffted_data_high = fft(segments_data_high[i])
        before_volume_list_high = data2midi(
            F=ffted_data_high, before_volume_list=before_volume_list_high,
            fs=new_fs_high, N=window_size_high,
            start_note=108, end_note=128,
            use_pitch_bend=False, next_tick=True
        )

    del segments_data_high, segments_data_middle, segments_data_low
    gc.collect()

    print("\nMIDI保存中...")

    mid.save(output_path_obj)
