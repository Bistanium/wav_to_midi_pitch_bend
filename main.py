import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from math import log10, modf
import wave
import numpy as np
from numpy.fft import fft
from scipy.signal import firwin, lfilter

# midi化関数
def data2midi(F, fs, N):
    half_n = N // 2
    sec = N / fs
    soundtime = int(round(120*sec, 0))
    beforenote, maxvolume = 0, 0
    otolist = [[35] for _ in range(10)]
    # pitchbend
    pitches = [0, 410, 819, 1229, 1638, -2048, -1638, -1229, -819, -410]
    for i in range(1, 10):
        channel = i if i != 9 else 10
        tracks[i].append(Message('pitchwheel', channel=channel, pitch=pitches[i]))

    for i in range(1, half_n):
        if abs(F.imag[i]/N) > 4: # 音量の閾値 無いと負荷でマズイ
            i_sec = i/sec # 周波数
            if 64 < i_sec < 11175: # midiの範囲に収める
                # ノート番号計算
                midinote = 69 + log10(i_sec/440)/0.025085832972
                # 音量計算
                volume = (abs(F.imag[i]/N*2) ** (1.8/3))
                if volume > 127: volume = 127

                incomp_rounded_midinote = round(midinote, 1)
                if beforenote != incomp_rounded_midinote: # 音が変わったら前の音階をmidiに打ち込む
                    syosu, seisu = modf(beforenote) # 整数部分と小数部分の分離
                    syosu = round(syosu, 1)
                    rounded_midinote = int(round(beforenote, 0))
                    rounded_volume = int(round(maxvolume, 0))
                    if syosu == 0.5:
                        if beforenote - rounded_midinote > 0: # 四捨五入(38.5→38)になるとき
                            rounded_midinote = rounded_midinote + 1 # 39にして正しい四捨五入にする
                        if rounded_midinote in otolist[5]: #重複チェック
                            continue
                    otonum = int(syosu*10) #トラックと対応させる
                    ch = otonum if otonum != 9 else 10
                    otolist[otonum].append(rounded_midinote)
                    tracks[otonum].append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=ch, time=00))
                    beforenote, maxvolume = incomp_rounded_midinote, volume
                elif volume > maxvolume: # 同じ音階なら音量を今までの最大値にする
                    maxvolume = volume

    for i, track in enumerate(tracks):
        ch = i if i != 9 else 10
        for j, midinote in enumerate(otolist[i]):
            if j == 0:
                track.append(Message('note_off', note=midinote, channel=ch, time=soundtime))
            else:
                track.append(Message('note_off', note=midinote, channel=ch, time=0))


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
    for i in range(0, len_data, win_size//2):
        endi = i + win_size
        if endi < len_data:
            splited_data.append(data[i:endi] * win)
        else:
            win = np.hanning(len(data[i:-1]))
            splited_data.append(data[i:-1] * win)
    return splited_data


def downsampling(conversion_rate, data, fs):
    # FIRフィルタ
    nyqF = fs/2                       # 変換前のナイキスト周波数
    cF = (conversion_rate/2-500)/nyqF # カットオフ周波数
    taps = 511                        # フィルタ係数
    b = firwin(taps, cF)   # LPF
    # フィルタリング
    data = lfilter(b, 1, data)

    # 間引き処理
    downlate = np.arange(0, len(data)-1, fs/conversion_rate)
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
    data_l = read_wav("test.wav")

    # Wavの情報取得
    wi = info_wav("test.wav")

    # ダウンサンプリング
    if wi["fs"] > 40960:
        new_fs = 40960
        downed_data = downsampling(new_fs, data_l, wi["fs"])
    else: #アップサンプリング予定
        new_fs = wi["fs"]
        downed_data = data_l
    del data_l

    # ウィンドウサイズ
    win_size = 1024 * 16

    # データ分割
    splited_data = audio_split(downed_data, win_size)
    del downed_data

    # FFT&midi化
    len_splited_data = len(splited_data)
    for i in range(0, len_splited_data):
        ffted_data = fft(splited_data[i])
        data2midi(ffted_data, new_fs, len(ffted_data.imag))

    out_file = "test_wav_pitch_bend.mid"
    mid.save(out_file)
