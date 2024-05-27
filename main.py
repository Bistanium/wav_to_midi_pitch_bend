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
    count0 = count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count10 = 0
    otolist0, otolist1, otolist2, otolist3, otolist4, otolist5, otolist6, otolist7, otolist8, otolist10 = [35], [35], [35], [35], [35], [35], [35], [35], [35], [35]
    track1.append(Message('pitchwheel', channel=1, pitch=410))
    track2.append(Message('pitchwheel', channel=2, pitch=819))
    track3.append(Message('pitchwheel', channel=3, pitch=1229))
    track4.append(Message('pitchwheel', channel=4, pitch=1638))
    track5.append(Message('pitchwheel', channel=5, pitch=-2048))
    track6.append(Message('pitchwheel', channel=6, pitch=-1638))
    track7.append(Message('pitchwheel', channel=7, pitch=-1229))
    track8.append(Message('pitchwheel', channel=8, pitch=-819))
    track10.append(Message('pitchwheel', channel=10, pitch=-410))
    for i in range(1, half_n):
        if abs(F.imag[i]/N) > 4: # 音量の閾値 無いと負荷でマズイ
            i_sec = i/sec # i/secが周波数
            if 64 < i_sec < 11175: # midiの範囲に収める
                # ノート番号計算
                midinote = 69 + log10(i_sec/440)/0.025085832972

                # 音量計算
                volume = (abs(F.imag[i]/N*2) ** (1.8/3))
                if volume > 127: volume = 127

                incomp_rounded_midinote = round(midinote, 1)
                if not beforenote == incomp_rounded_midinote: # 音が変わったら前の音階をmidiに打ち込む
                    syosu, seisu = modf(beforenote) # 整数部分と小数部分の分離
                    syosu = round(syosu, 1)
                    rounded_midinote = int(round(beforenote, 0))
                    rounded_volume = int(round(maxvolume, 0))
                    if not rounded_volume == 0:
                        if syosu == 0.0:
                            otolist0.append(rounded_midinote)
                            track0.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=0, time=00))
                        elif syosu == 0.1:
                            otolist1.append(rounded_midinote)
                            track1.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=1, time=00))
                        elif syosu == 0.2:
                            otolist2.append(rounded_midinote)
                            track2.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=2, time=00))
                        elif syosu == 0.3:
                            otolist3.append(rounded_midinote)
                            track3.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=3, time=00))
                        elif syosu == 0.4:
                            otolist4.append(rounded_midinote)
                            track4.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=4, time=00))
                        elif syosu == 0.5:
                            if beforenote - rounded_midinote > 0: # 四捨五入(38.5→38)になるとき
                                rounded_midinote = rounded_midinote + 1
                            if not rounded_midinote in otolist5: # 重複チェック
                                otolist5.append(rounded_midinote)
                                track5.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=5, time=00))
                        elif syosu == 0.6:
                            otolist6.append(rounded_midinote)
                            track6.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=6, time=00))
                        elif syosu == 0.7:
                            otolist7.append(rounded_midinote)
                            track7.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=7, time=00))
                        elif syosu == 0.8:
                            otolist8.append(rounded_midinote)
                            track8.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=8, time=00))
                        elif syosu == 0.9:
                            otolist10.append(rounded_midinote)
                            track10.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=10, time=00))
                    beforenote, maxvolume = incomp_rounded_midinote, volume
                elif volume > maxvolume: # 同じ音階なら音量を今までの最大値にする
                    maxvolume = volume

    for j in otolist0:
        count0 += 1
        if count0 == 1:
            track0.append(Message('note_off', note=j, channel=0, time=soundtime))
        else:
            track0.append(Message('note_off', note=j, channel=0, time=0))
    for j in otolist1:
        count1 += 1
        if count1 == 1:
            track1.append(Message('note_off', note=j, channel=1, time=soundtime))
        else:
            track1.append(Message('note_off', note=j, channel=1, time=0))
    for j in otolist2:
        count2 += 1
        if count2 == 1:
            track2.append(Message('note_off', note=j, channel=2, time=soundtime))
        else:
            track2.append(Message('note_off', note=j, channel=2, time=0))
    for j in otolist3:
        count3 += 1
        if count3 == 1:
            track3.append(Message('note_off', note=j, channel=3, time=soundtime))
        else:
            track3.append(Message('note_off', note=j, channel=3, time=0))
    for j in otolist4:
        count4 += 1
        if count4 == 1:
            track4.append(Message('note_off', note=j, channel=4, time=soundtime))
        else:
            track4.append(Message('note_off', note=j, channel=4, time=0))
    for j in otolist5:
        count5 += 1
        if count5 == 1:
            track5.append(Message('note_off', note=j, channel=5, time=soundtime))
        else:
            track5.append(Message('note_off', note=j, channel=5, time=0))
    for j in otolist6:
        count6 += 1
        if count6 == 1:
            track6.append(Message('note_off', note=j, channel=6, time=soundtime))
        else:
            track6.append(Message('note_off', note=j, channel=6, time=0))
    for j in otolist7:
        count7 += 1
        if count7 == 1:
            track7.append(Message('note_off', note=j, channel=7, time=soundtime))
        else:
            track7.append(Message('note_off', note=j, channel=7, time=0))
    for j in otolist8:
        count8 += 1
        if count8 == 1:
            track8.append(Message('note_off', note=j, channel=8, time=soundtime))
        else:
            track8.append(Message('note_off', note=j, channel=8, time=0))
    for j in otolist10:
        count10 += 1
        if count10 == 1:
            track10.append(Message('note_off', note=j, channel=10, time=soundtime))
        else:
            track10.append(Message('note_off', note=j, channel=10, time=0))


# Wave読み込み
def read_wav(file_path):
    wf = wave.open(file_path, "rb")
    buf = wf.readframes(-1) # 全部読み込む

    # 16bitごとに10進数化
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype='int16')
    else:
        data = np.zeros(len(buf), dtype=np.complex128)

    # ステレオの場合，チャンネルを分離
    if wf.getnchannels() == 2:
        data_l = data[::2]
        data_r = data[1::2]
    else:
        data_l = data
        data_r = data
    wf.close()
    return data_l,data_r


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
    track0, track1, track2, track3, track4, track5, track6, track7, track8, track10 = MidiTrack(), MidiTrack(), MidiTrack(), MidiTrack(), MidiTrack(), MidiTrack(), MidiTrack(), MidiTrack(), MidiTrack(), MidiTrack()
    mid.tracks.extend([track0, track1, track2, track3, track4, track5, track6, track7, track8, track10])

    # テンポ
    miditempo = MetaMessage('set_tempo', tempo=mido.bpm2tempo(480))
    track0.append(miditempo)
    track1.append(miditempo)
    track2.append(miditempo)
    track3.append(miditempo)
    track4.append(miditempo)
    track5.append(miditempo)
    track6.append(miditempo)
    track7.append(miditempo)
    track8.append(miditempo)
    track10.append(miditempo)

    # Wav読み込み
    data_l,data_r = read_wav("test.wav")
    del data_r

    # Wavの情報取得
    wi = info_wav("test.wav")

    # ダウンサンプリング
    if wi["fs"] > 40960:
        new_fs = 40960
        downed_data = downsampling(new_fs, data_l, wi["fs"])
    else: #アップサンプリングは実装予定
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
