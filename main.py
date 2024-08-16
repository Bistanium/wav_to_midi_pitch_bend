import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from math import log10, modf, sqrt
import wave
import numpy as np
from numpy.fft import fft
from scipy import interpolate
from scipy.signal import firwin, lfilter

# midi化関数
def data2midi(F, fs, N, bol, minvol, overlap=2, m_q=1, z_p_m=1):
    half_n = N // 2
    sec = N / fs
    beforenote, maxvolume = 0, 0
    otolist = [[0 for _ in range(128)] for _ in range(10)]
    ntime = int(round(240 * sec / overlap / z_p_m, 0))
    volumes = (np.abs(F) / N * 2) ** 0.6 * m_q
    bendaves, bendave, beforeseisu, volumesum = [0 for _ in range(10)], 0, 0, 0

    for i in range(1, half_n):
        volume = volumes[i]
        if volume > minvol:
            i_sec = i / sec # 周波数
            if 32 < i_sec < 12873: # midiの範囲を指定
                # ノート番号計算
                raw_note = 69 + log10(i_sec / 440) / 0.025085832972
                midinote = round(raw_note, 1)

                # 可変ピッチベンド
                bendsyosu, bendseisu = modf(raw_note)
                bendseisu = int(bendseisu)
                bendsyosu = round(bendsyosu, 8)
                # 39.99→40.01などに切り替わるときの対応
                if beforeseisu == bendseisu - 1:
                    bendsyosu += 1
                else:
                    beforeseisu = bendseisu
                # 可変はこれまでの平均にする
                if bendave == 0:
                    bendave = bendsyosu * volume
                else:
                    bendave = bendave + bendsyosu * volume
                volumesum += volume

                if beforenote != midinote: # 音が変わったら前の音階をmidiに打ち込む
                    # 次の音の値まで入っているから除く
                    if volumesum == volume:
                        bendave /= volume
                    else:
                        bendave = (bendave - bendsyosu * volume) / (volumesum - volume)

                    syosu, seisu = modf(round(beforenote, 1)) # 整数部分と小数部分の分離
                    syosu = round(syosu, 1)
                    maxvolume *= 0.5 * (1 / m_q) # 音量調整 2のとき0.25, 1.2のとき0.4167
                    maxvolume = maxvolume if maxvolume <= 127 else 127
                    rounded_note, rounded_volume = int(round(beforenote, 0)), int(round(maxvolume, 0))
                    if syosu == 0.5:
                        if beforenote - rounded_note > 0: # 38.5→38になるとき
                            rounded_note = rounded_note + 1 # 39にして正しい四捨五入にする
                            if otolist[5][rounded_note] != 0:
                                continue

                    tracknum = int(syosu * 10) # トラックと対応させる
                    otolist[tracknum][rounded_note] = rounded_volume

                    # 0つ目のトラックのピッチベンド幅の変換など
                    if tracknum == 0 and bendave > 0.5:
                        bendave -= 1
                    if bendaves[tracknum] == 0:
                        bendaves[tracknum] = bendave
                    bendaves[tracknum] = (bendaves[tracknum] + bendave) / 2
                    if bendsyosu >= 1:
                        bendsyosu -= 1

                    beforeseisu, bendave, volumesum = bendseisu, bendsyosu * volume, volume # ピッチベンド用
                    beforenote, maxvolume = midinote, volume
                else:
                    maxvolume = sqrt(maxvolume ** 2 + volume ** 2)

    sim = 2 # 2~4が良い, -1でノートを繋げる機能無効化
    max_bend = 4096 # 5つ目以上のトラックのピッチベンド幅が0のときに-4096になるのを防ぐ
    bend_values = np.array([409.6 * (bend * 10) if i < 5 else -409.6 * ((1 - bend) * 10) for i, bend in enumerate(bendaves)])
    bend_values = np.clip(bend_values, -max_bend, max_bend)
    bend_values[np.abs(bend_values) == max_bend] = 0  # 範囲外の値を0に設定
    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        track.append(Message('pitchwheel', channel=ch, pitch=int(round(bend_values[i], 0))))
        before_vol_list = bol[i]
        vol_list = otolist[i]
        for j in range(24, 128):
            beforevol = before_vol_list[j]
            nowvol = vol_list[j]
            if beforevol != 0:
                if nowvol < beforevol - sim or beforevol + sim < nowvol or nowvol < minvol or nowvol == 0: # 音量変化が指定値より大きいor閾値以下のとき
                    track.append(Message('note_off', note=j, channel=ch, time=0))
                    if nowvol > minvol:
                        track.append(Message('note_on', note=j, velocity=nowvol, channel=ch, time=0))
                else: # nowvol >= beforevol - sim and beforevol + sim >= nowvol
                    otolist[i][j] = beforevol
            else: # 前の音がなかったとき
                if nowvol > minvol:
                    track.append(Message('note_on', note=j, velocity=nowvol, channel=ch, time=0))
        track.append(Message('note_off', note=0, channel=ch, time=ntime))

    return otolist


# Wave読み込み
def read_wav(file_path):
    wf = wave.open(file_path, "rb")
    buf = wf.readframes(-1) # 全部読み込む

    # 16bitごとに10進数化
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype=np.int16)
    else:
        data = np.zeros(len(buf), dtype=np.complex128)

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
def audio_split(data, win_size, overlap=2, zero_padding_multiplier=1):
    splited_data = []
    len_data = len(data)
    win = np.hanning(win_size)

    # ゼロ埋めに使用するサイズを計算
    padded_size = win_size * zero_padding_multiplier

    for i in range(0, len_data, win_size // overlap):
        endi = i + win_size
        if endi < len_data:
            segment = data[i:endi] * win
        else:
            win = np.hanning(len(data[i:-1]))
            segment = data[i:-1] * win

        if zero_padding_multiplier != 1:
            # ゼロパディングを適用
            padded_segment = np.pad(segment, (0, padded_size - len(segment)), 'constant')

            # 振幅補正
            if sum(np.abs(padded_segment)) != 0:
                acf = (sum(np.abs(segment)) / len(segment)) / (sum(np.abs(padded_segment)) / len(padded_segment)) / sqrt(zero_padding_multiplier)
            else:
                acf = 0
        else:
            padded_segment = segment
            acf = 1

        splited_data.append(padded_segment * acf)

    return splited_data


def upsampling(up_fs, data, fs):
    # FIRフィルタ
    nyqF = up_fs / 2          # 変換前のナイキスト周波数
    cF = (fs / 2 - 500) / nyqF  # カットオフ周波数
    taps = 511                # フィルタ係数
    b = firwin(taps, cF)      # LPF

    # 補間処理
    x = np.arange(0, len(data))
    try:
        interpolated = interpolate.interp1d(x, data, kind="cubic")
    except ValueError:
        interpolated = interpolate.interp1d(x, data, kind="linear")
    upsampled_length = int(round(len(data) * (up_fs / fs), 0))
    uplate = np.linspace(0, len(data) - 1, upsampled_length)
    uped_data = interpolated(uplate)

    # フィルタリング
    uped_data = lfilter(b, 1, uped_data)
    return uped_data


def downsampling(down_fs, data, fs):
    nyqF = down_fs / 2
    cF = (nyqF - 500) / (fs / 2)
    taps = 511
    b = firwin(taps, cF)
    # フィルタリング
    data = lfilter(b, 1, data)

    # 間引き処理, ここで音質低下
    len_data = len(data)
    downsampled_length = int(round(len_data * (down_fs / fs), 0))
    downlate = np.linspace(0, len_data - 1, downsampled_length)
    rounded_indices = np.round(downlate).astype(int)
    downed_data = data[rounded_indices]

    return downed_data


if __name__ == '__main__':
    # Settings_Area
    wav_name = "test.wav" # 入力されるWavファイル名
    out_midi_name = "test_wav_pitch_bend.mid" # 出力されるMIDIファイル名    
    midi_quality = 1 # 0.1~16まで。値が大きいと処理時間も増える

    # 変えない方が良い
    min_volume = 4 # MIDIの音量の最小値
    window_size = 1024 * 16 # ウィンドウサイズ
    overlap = 2 # 2:50%, 4:75%
    zero_padding_multiplier = 2 #ゼロ埋め倍数

    # midi定義
    mid = MidiFile()
    tracks = [MidiTrack() for _ in range(10)] #10個のトラックを作成
    mid.tracks.extend(tracks)

    tracks[0].append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(480)))

    # Wav読み込み
    data = read_wav(wav_name)

    # Wavの情報取得
    wi = info_wav(wav_name)

    # 再サンプリング
    new_fs = 40960
    if wi["fs"] > new_fs:
        print("start")
        samped_data = downsampling(new_fs, data, wi["fs"]) # ※間引き時に音質が劣化する
        print("end")
    elif wi["fs"] < new_fs:
        samped_data = upsampling(new_fs, data, wi["fs"])
    else:
        samped_data = data
    del data

    # 一応範囲チェック&変数名を短くする
    z_p_m = int(zero_padding_multiplier) if 1 <= zero_padding_multiplier <= 4 else 1
    m_q = midi_quality if 0.1 <= midi_quality <= 16 else 1

    # データ分割
    splited_data = audio_split(samped_data, window_size, overlap, z_p_m)
    del samped_data

    bol = [[0 for _ in range(128)] for _ in range(10)] # before_oto_list
    # FFT&midi化
    len_data = len(splited_data)
    for i in range(0, len_data):
        ffted_data = fft(splited_data[i])
        bol = data2midi(ffted_data, new_fs, len(ffted_data.imag), bol, min_volume, overlap, m_q, z_p_m)

    # 最後のデータ分の処理
    time = int(round(60 * (len_data/new_fs) * overlap, 0))
    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        for j in range(24, 128):
            if bol[i][j] > min_volume:
                track.append(Message('note_on', note=j, velocity=bol[i][j], channel=ch, time=0))
        # ノートオフ処理
        for j in range(24, 128):
            if j == 0:
                track.append(Message('note_off', note=0, channel=ch, time=time))
            else:
                track.append(Message('note_off', note=j, channel=ch, time=0))

    mid.save(out_midi_name)
