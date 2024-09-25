import wave
from math import log10, modf, sqrt
import os
import tkinter, tkinter.filedialog
import sys
import numpy as np
from numpy.fft import fft
from scipy.signal import resample_poly
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from tqdm import tqdm
import gc


# midi化関数
def data2midi(F, fs, N, bol, start_note, end_note, use_bend, next_tick, min_vol=4):
    sec = N / fs
    before_midi_note, max_volume = 0, 0
    otolist = [[0 for _ in range(128)] for _ in range(10)]

    # 音量はそれっぽく聴こえるようにした結果こうなった
    volumes = (np.abs(F) / N * 4) ** 0.6

    if use_bend:
        bend_averages = [0 for _ in range(10)]
        bend_average, before_bend_seisu, volume_sum = 0, 0, 0

    range_start = int(sec * 440 * 1.0594630943593 ** (start_note - 69) * 0.9)
    range_end = int(sec * 440 * 1.0594630943593 ** (end_note - 69) * 1.1)
    for i in range(range_start, range_end):
        volume = volumes[i]
        if volume > min_vol:
            # ノート番号計算
            midi_note = 69 + log10(i / sec / 440) / 0.025085832972
            round_1_midi_note = round(midi_note * 10) / 10

            if use_bend:
                # 可変ピッチベンド
                bend_syosu, bend_seisu = modf(midi_note)
                bend_syosu, bend_seisu = round(bend_syosu * 100000000) / 100000000, int(bend_seisu) # (小数第8位まで残す)
                # 39.99→40.01などに切り替わるときの対応
                if before_bend_seisu == bend_seisu - 1:
                    bend_syosu += 1
                else:
                    before_bend_seisu = bend_seisu

                # 音量に応じて重みを付ける
                bend_average += bend_syosu * volume

                volume_sum += volume

            if before_midi_note != round_1_midi_note: # 音が変わったら前の音階をmidiに打ち込む
                if use_bend:
                    # 次の音の値まで入っているから除く
                    if volume_sum == volume:
                        bend_average /= volume
                    else:
                        bend_average = (bend_average - bend_syosu * volume) / (volume_sum - volume)

                midi_note_syosu, midi_note_seisu = modf(before_midi_note) # 整数部分と小数部分の分離
                midi_note_syosu = round(midi_note_syosu * 10) / 10

                # 音量調整
                max_volume *= 0.35
                max_volume = max_volume if max_volume <= 127 else 127

                round_0_midi_note, round_0_volume = int(round(before_midi_note)), int(round(max_volume))

                if round_0_midi_note > 127:
                    continue

                if midi_note_syosu == 0.5:
                    # round half up
                    if before_midi_note - round_0_midi_note > 0:
                        round_0_midi_note = round_0_midi_note + 1        
                        if otolist[5][round_0_midi_note] != 0:
                            continue

                track_num = int(midi_note_syosu * 10) # トラックと対応させる

                otolist[track_num][round_0_midi_note] = round_0_volume

                if use_bend:
                    # 0つ目のトラックのピッチベンド幅の変換
                    if track_num == 0 and bend_average > 0.5:
                        bend_average -= 1
                    if bend_syosu >= 1:
                        bend_syosu -= 1

                    bend_averages_track_num = bend_averages[track_num]
                    if bend_averages_track_num == 0:
                        bend_averages[track_num] = bend_average
                    else:
                        bend_averages[track_num] = (bend_averages_track_num + bend_average) / 2

                    before_bend_seisu, bend_average, volume_sum = bend_seisu, bend_syosu * volume, volume # ピッチベンド用

                before_midi_note, max_volume = round_1_midi_note, volume
            else:
                max_volume = sqrt(max_volume ** 2 + volume ** 2)


    sim = 2 # 2~4が良い, -1でノートを繋げる機能無効化
    if use_bend:
        max_bend = 4096 # 5つ目以上のトラックのピッチベンド幅が0のときに-4096になるのを防ぐ
        bend_values = np.array([409.6 * (bend * 10) if i < 5 else -409.6 * ((1 - bend) * 10) for i, bend in enumerate(bend_averages)])
        bend_values = np.clip(bend_values, -max_bend, max_bend)
        bend_values[np.abs(bend_values) == max_bend] = 0  # 範囲外の値を0に設定
        bend_values = np.round(bend_values).astype(int)
        # bend_values = [0, 410, 819, 1229, 1638, -2048, -1638, -1229, -819, -410] 固定用
    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        if use_bend:
            track.append(Message('pitchwheel', channel=ch, pitch=bend_values[i]))
        before_volume_list = bol[i]
        volume_list = otolist[i]
        for j in range(start_note, end_note):
            before_vol = before_volume_list[j]
            now_vol = volume_list[j]
            if before_vol != 0:
                # 音量変化が指定値より大きいor閾値以下のとき
                if now_vol < before_vol - sim or before_vol + sim < now_vol or now_vol < min_vol or now_vol == 0:
                    track.append(Message('note_off', note=j, channel=ch, time=0))
                    if now_vol > min_vol:
                        track.append(Message('note_on', note=j, velocity=now_vol, channel=ch, time=0))
                else: # now_vol >= before_vol - sim and before_vol + sim >= now_vol
                    otolist[i][j] = before_vol
            else: # 前の音がなかったとき
                if now_vol > min_vol:
                    track.append(Message('note_on', note=j, velocity=now_vol, channel=ch, time=0))

        if next_tick:
            note_time = int(round(120 * sec))
            track.append(Message('note_off', note=0, channel=ch, time=note_time))

    return otolist


# Wave読み込み
def read_wav(file_path):
    wf = wave.open(file_path, "rb")
    buf = wf.readframes(-1) # 全部読み込む

    # 16bitのときのみ10進数化
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype=np.int16)
    else:
        sys.exit("ビット深度が16bit以外です")

    # ステレオの場合左音声のみ
    if wf.getnchannels() == 2:
        mono_data = data[::2]
    else:
        mono_data = data
    wf.close()
    return mono_data


# wavファイルの情報を取得
def info_wav(file_path):
    ret = {}
    wf = wave.open(file_path, "rb")
    ret["ch"] = wf.getnchannels()
    ret["byte"] = wf.getsampwidth()
    ret["fs"] = wf.getframerate()
    ret["N"] = wf.getnframes()
    ret["sec"] = ret["N"] / ret["fs"]
    wf.close()
    return ret


# データ分割
def audio_split(data, win_size, window_func="hamming", overlap=2):
    len_data = len(data)
    if window_func == "hamming":
        win = np.hamming(win_size) * 0.93 # hann窓との比
    else:
        win = np.hanning(win_size)

    step_size = win_size // overlap
    num_segments = (len_data - win_size) // step_size + 1

    # 各セグメントの開始インデックスを計算
    indices = np.arange(0, num_segments * step_size, step_size)

    # 窓サイズ未満のデータの行+4つ分空の行=5
    splited_data = np.zeros((num_segments + 5, win_size), dtype=np.int16)

    for idx, start in enumerate(indices):
        end = start + win_size
        segment = data[start:end]
    
        # ウィンドウをかける
        win_segment = segment * win

        # 結果を保存
        splited_data[idx, :] = win_segment

    # 最後のセグメント処理（残りのデータがある場合）
    remaining = len_data - indices[-1] - win_size
    if remaining > 0:
        # 1を引いてインデックスのずれを合わせる
        segment = data[-remaining - 1:-1]
        if window_func == "hamming":
            win = np.hamming(len(segment)) * 0.93
        else:
            win = np.hanning(len(segment))
        win_segment = segment * win

        # ゼロパディング
        padded_segment = np.pad(win_segment, (0, win_size - len(win_segment)), 'constant')

        # 結果を保存
        splited_data[-5, :] = padded_segment

    return splited_data


def return_amp(data):
    # プラスとマイナスの絶対値が大きい方
    data_max_val = max(np.max(data), -(np.min(data) + 1))

    # 0.9倍して再サンプリング後のデータがin16の上限を超えないようにする
    amp = data_max_val * 0.9

    return amp


def change_samplingrate(data, fs, target_fs, amp):
    # 16bit想定
    data = data / 32768

    changed_data = resample_poly(data, target_fs, fs)

    changed_data = changed_data * amp

    changed_data = np.clip(changed_data, -32768, 32767)

    changed_data = changed_data.astype(np.int16)

    return changed_data


if __name__ == '__main__':

    while True:
        fTyp = [("Audio File", ".wav"), ("wav", ".wav")]
        input_name = tkinter.filedialog.askopenfilename(filetypes = fTyp)

        if input_name:
            k = str(os.path.splitext(input_name)[1]) # kは拡張子の略
            if k == ".wav":
                break
            else:
                sys.exit("ファイルが正しくありません")
        else:
            sys.exit()

    out_midi_name = f"{str(os.path.splitext(input_name)[0])}.mid"
    if os.path.isfile(out_midi_name):
        count = 1
        while True:
            out_midi_name = f"{str(os.path.splitext(input_name)[0])} ({count}).mid"
            if not os.path.isfile(out_midi_name) or count >= 10:
                break
            count += 1

    # midi定義
    mid = MidiFile()
    tracks = [MidiTrack() for _ in range(10)]
    mid.tracks.extend(tracks)

    tracks[0].append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(480)))

    # Wav読み込み
    data = read_wav(input_name)

    # Wavの情報取得
    wi = info_wav(input_name)
    
    # 高音部分のウィンドウサイズ
    window_size = 1024 * 8 

    print("\n再サンプリング&データ分割開始")

    # 再サンプリング
    new_fs_high = 40960
    new_fs = 10240
    new_fs_low = 640

    amp = return_amp(data)
    samped_data_high = change_samplingrate(data, wi["fs"], new_fs_high, amp)
    samped_data = change_samplingrate(data, wi["fs"], new_fs, amp)
    samped_data_low = change_samplingrate(data, wi["fs"], new_fs_low, amp)

    # データ分割
    splited_data_high= audio_split(samped_data_high, window_size, window_func="hann")
    splited_data = audio_split(samped_data, window_size // 2, window_func="hamming")
    splited_data_low = audio_split(samped_data_low, window_size // 16, window_func="hamming")

    print("再サンプリング&データ分割完了\n")

    del data, samped_data_high, samped_data, samped_data_low
    gc.collect()

    before_otolist_high = [[0 for _ in range(128)] for _ in range(10)]
    before_otolist = [[0 for _ in range(128)] for _ in range(10)]
    before_otolist_low = [[0 for _ in range(128)] for _ in range(10)]

    len_data_high = len(splited_data_high)
    len_data = len(splited_data)
    len_data_low = len(splited_data_low)

    range_data = len_data * 2
    range_data_low = len_data_low * 4

    # FFT&midi化
    for i in tqdm(range(0, len_data_high), desc='Convert to MIDI'):
        # 低音用
        if i % 4 == 0 and i < range_data_low:
            ffted_data_low = fft(splited_data_low[i // 4])
            before_otolist_low = data2midi(ffted_data_low, new_fs_low, len(ffted_data_low), before_otolist_low, start_note=24, end_note=48, use_bend=False, next_tick=False)

        # 中音用
        if i % 2 == 0 and i < range_data:
            ffted_data = fft(splited_data[i // 2])
            before_otolist = data2midi(ffted_data, new_fs, len(ffted_data), before_otolist, start_note=48, end_note=96, use_bend=True, next_tick=False)

        # 高音用
        ffted_data_high = fft(splited_data_high[i])
        before_otolist_high = data2midi(ffted_data_high, new_fs_high, len(ffted_data_high), before_otolist_high, start_note=96, end_note=128, use_bend=False, next_tick=True)

    del splited_data_high, splited_data, splited_data_low
    gc.collect()

    # (最後のデータは何もないので捨てる)
    note_time = int(round(120 * (len_data_high/new_fs_high)))
    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        track.append(Message('note_off', note=0, channel=ch, time=note_time))

    print("\nMIDI保存中...")

    mid.save(out_midi_name)
