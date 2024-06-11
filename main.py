import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from math import log10, modf
import wave
import numpy as np
from numpy.fft import fft
from scipy import interpolate
from scipy.signal import firwin, lfilter

# midi化関数
def data2midi(F, fs, N):
    half_n = N // 2
    sec = N / fs
    beforenote, maxvolume = 0, 0
    otolist = [[0] for _ in range(10)]
    time = int(round(120 * sec, 0))
    volumes = (abs(F) / N * 2) ** 0.6 * 0.8
    # pitchbend
    pitches = [0, 410, 819, 1229, 1638, -2048, -1638, -1229, -819, -410]
    for i in range(1, 10):
        ch = i if i != 9 else 10
        tracks[i].append(Message('pitchwheel', channel=ch, pitch=pitches[i]))

    for i in range(1, half_n):
        volume = volumes[i]
        if volume > 4: # 音量の閾値 負荷によって調整
            i_sec = i / sec # 周波数
            if 64 < i_sec < 11175: # midiの範囲を指定
                # ノート番号計算
                midinote = round(69 + log10(i_sec / 440) / 0.025085832972, 1)

                if beforenote != midinote: # 音が変わったら前の音階をmidiに打ち込む
                    syosu, seisu = modf(beforenote) # 整数部分と小数部分の分離
                    syosu = round(syosu, 1)
                    if maxvolume > 127:
                        maxvolume = 127
                    rounded_note, rounded_volume = int(round(beforenote, 0)), int(round(maxvolume, 0))
                    if syosu == 0.5:
                        if beforenote - rounded_note > 0: # 四捨五入(38.5→38)になるとき
                            rounded_note = rounded_note + 1 # 39にして正しい四捨五入にする
                        if rounded_note in otolist[5]: # 重複チェック
                            continue
                    tracknum = int(syosu*10) # トラックと対応させる
                    ch = tracknum if tracknum != 9 else 10
                    otolist[tracknum].append(rounded_note)
                    tracks[tracknum].append(Message('note_on', note=rounded_note, velocity=rounded_volume, channel=ch, time=00))
                    beforenote, maxvolume = midinote, volume
                elif volume > maxvolume: # 同じ音階なら条件付きで音量を更新
                    maxvolume = volume

    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        for j, note in enumerate(otolist[i]):
            if j == 0:
                track.append(Message('note_off', note=note, channel=ch, time=time))
            else:
                track.append(Message('note_off', note=note, channel=ch, time=0))


# Wave読み込み
def read_wav(file_path):
    wf = wave.open(file_path, "rb")
    buf = wf.readframes(-1) # 全部読み込む

    # 16bitごとに10進数化
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype='int16')
    else:
        data = np.zeros(len(buf), dtype=np.complex128)

    # ステレオの場合左音声のみ
    if wf.getnchannels() == 2:
        data_l = data[::2]
    else:
        data_l = data
    wf.close()
    return data_l


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
def audio_split(data, win_size):
    splited_data = []
    len_data = len(data)
    win = np.hanning(win_size)
    for i in range(0, len_data, win_size // 2):
        endi = i + win_size
        if endi < len_data:
            splited_data.append(data[i:endi] * win)
        else:
            win = np.hanning(len(data[i:-1]))
            splited_data.append(data[i:-1] * win)
    return splited_data


def upsampling(up_fs, data, fs):
    # FIRフィルタ
    nyqF = fs / 2             # 変換前のナイキスト周波数
    cF = (nyqF - 500) / nyqF  # カットオフ周波数
    taps = 511                # フィルタ係数
    b = firwin(taps, cF)      # LPF

    # 補間処理
    x = np.arange(0, len(data))
    try:
        interpolated = interpolate.interp1d(x, data, kind="cubic")
    except:
        interpolated = interpolate.interp1d(x, data, kind="linear")
    upsampled_length = int(round(len(data) * up_fs / fs, 0))
    uplate = np.linspace(0, len(data) - 1, upsampled_length)
    uped_data = interpolated(uplate)

    # フィルタリング
    uped_data = lfilter(b, 1, uped_data)
    return uped_data


def downsampling(down_fs, data, fs):
    nyqF = fs / 2
    cF = (down_fs / 2 - 500) / nyqF
    taps = 511
    b = firwin(taps, cF)
    # フィルタリング
    data = lfilter(b, 1, data)

    # 間引き処理
    downsampled_length = int(round(len(data) * down_fs / fs, 0))
    downlate = np.linspace(0, len(data) - 1, downsampled_length)
    rounded_indices = np.round(downlate).astype(int)
    downed_data = data[rounded_indices]
    return downed_data


if __name__ == '__main__':

    # midi定義
    mid = MidiFile()
    tracks = [MidiTrack() for _ in range(10)] #10個のトラックを作成
    mid.tracks.extend(tracks)

    tracks[0].append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(480)))

    # Wav読み込み
    data = read_wav("test.wav")

    # Wavの情報取得
    wi = info_wav("test.wav")

    # ダウンサンプリング
    new_fs = 40960
    if wi["fs"] > new_fs:
        samped_data = downsampling(new_fs, data, wi["fs"])
    elif wi["fs"] < new_fs:
        samped_data = upsampling(new_fs, data, wi["fs"])
    else:
        new_fs = wi["fs"]
        samped_data = data
    del data

    # ウィンドウサイズ
    win_size = 1024 * 16

    # データ分割
    splited_data = audio_split(samped_data, win_size)
    del samped_data

    # FFT&midi化
    len_splited_data = len(splited_data)
    for i in range(0, len_splited_data):
        ffted_data = fft(splited_data[i])
        data2midi(ffted_data, new_fs, len(ffted_data.imag))

    out_file = "test_wav_pitch_bend.mid"
    mid.save(out_file)
