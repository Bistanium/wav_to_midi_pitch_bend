import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from math import log10, modf
import wave
import numpy as np
from numpy.fft import fft
from scipy.signal import firwin, lfilter

#midi化関数
def data2midi(F, fs, N):
    half_n = N // 2
    sec = N / fs
    beforenote, maxvolume = 0, 0
    count0, count1, count2, count3, count4, count5, count6, count7, count8, count10, count11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    otolist0, otolist1, otolist2, otolist3, otolist4, otolist5, otolist6, otolist7, otolist8, otolist10, otolist11 = [35], [35], [35], [35], [35], [35], [35], [35], [35], [35], [35]
    track0.append(Message('pitchwheel', channel=0, pitch=-1638))
    track1.append(Message('pitchwheel', channel=1, pitch=-1229))
    track2.append(Message('pitchwheel', channel=2, pitch=-819))
    track3.append(Message('pitchwheel', channel=3, pitch=-410))
    track5.append(Message('pitchwheel', channel=5, pitch=410))
    track6.append(Message('pitchwheel', channel=6, pitch=819))
    track7.append(Message('pitchwheel', channel=7, pitch=1229))
    track8.append(Message('pitchwheel', channel=8, pitch=1638))
    track10.append(Message('pitchwheel', channel=10, pitch=-2048))
    track11.append(Message('pitchwheel', channel=11, pitch=2048))
    for i in range(1, half_n):
        if abs(F.imag[i]/N) > 4: #音量の閾値 無いと負荷でマズイ
            i_sec = i/sec # i/secが周波数
            if 64 < i_sec < 11175: # midiの範囲に収める
                #ノート番号計算
                midinote = 69 + log10(i_sec/440)/0.025085832972

                #音量計算
                volume = (abs(F.imag[i]/N*2) ** (1.8/3))
                # 音量調整
                if midinote < 56:
                    volume = volume * 0.8
                    if midinote < 48:
                        volume = volume * 0.7
                        if midinote < 41:
                            volume = volume * 0.6
                if midinote > 107:
                    volume = volume * 0.8
                    if midinote > 112:
                        volume = volume * 0.7
                        if midinote > 118:
                            volume = volume * 0.6
                if volume > 127: volume = 127

                incomp_rounded_midinote = round(midinote, 1)
                if not beforenote == incomp_rounded_midinote: # 音が変わったら前の音階をmidiに打ち込む
                    syosu, seisu = modf(beforenote) #整数部分と小数部分の分離
                    syosu = round(syosu, 1)
                    rounded_midinote = int(round(beforenote, 0))
                    rounded_volume = int(round(maxvolume, 0))
                    if not rounded_volume == 0:
                        if syosu == 0.6:
                            otolist0.append(rounded_midinote)
                            track0.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=0, time=00))
                        elif syosu == 0.7:
                            otolist1.append(rounded_midinote)
                            track1.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=1, time=00))
                        elif syosu == 0.8:
                            otolist2.append(rounded_midinote)
                            track2.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=2, time=00))
                        elif syosu == 0.9:
                            otolist3.append(rounded_midinote)
                            track3.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=3, time=00))
                        elif syosu == 0.0:
                            otolist4.append(rounded_midinote)
                            track4.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=4, time=00))
                        elif syosu == 0.1:
                            otolist5.append(rounded_midinote)
                            track5.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=5, time=00))
                        elif syosu == 0.2:
                            otolist6.append(rounded_midinote)
                            track6.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=6, time=00))
                        elif syosu == 0.3:
                            otolist7.append(rounded_midinote)
                            track7.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=7, time=00))
                        elif syosu == 0.4:
                            otolist8.append(rounded_midinote)
                            track8.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=8, time=00))
                        elif syosu == 0.5:
                            note_note = midinote - rounded_midinote
                            if note_note < 0:
                                otolist10.append(rounded_midinote)
                                track10.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=10, time=00))
                            else:
                                otolist11.append(rounded_midinote)
                                track11.append(Message('note_on', note=rounded_midinote, velocity=rounded_volume, channel=11, time=00))
                    beforenote, maxvolume = incomp_rounded_midinote, volume
                elif volume > maxvolume: #同じ音階なら音量を今までの最大値にする
                    maxvolume = volume

    soundtime = int(round(120*sec, 0))
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
    for j in otolist11:
        count11 += 1
        if count11 == 1:
            track11.append(Message('note_off', note=j, channel=11, time=soundtime))
        else:
            track11.append(Message('note_off', note=j, channel=11, time=0))


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


def downsampling(conversion_rate, data, fs): #args(downsamp,data,nowsamp) return(data,downsamp)
    # FIRフィルタ
    nyqF = fs/2                       # 変換前のナイキスト周波数
    cF = (conversion_rate/2-500)/nyqF # カットオフ周波数を設定（変換後のナイキスト周波数より少し下を設定）
    taps = 511                        # フィルタ係数（奇数じゃないとだめ）
    b = firwin(taps, cF)   # LPFを用意
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
    track0 = MidiTrack()
    track1 = MidiTrack()
    track2 = MidiTrack()
    track3 = MidiTrack()
    track4 = MidiTrack()
    track5 = MidiTrack()
    track6 = MidiTrack()
    track7 = MidiTrack()
    track8 = MidiTrack()
    track10 = MidiTrack()
    track11 = MidiTrack()
    mid.tracks.append(track0)
    mid.tracks.append(track1)
    mid.tracks.append(track2)
    mid.tracks.append(track3)
    mid.tracks.append(track4)
    mid.tracks.append(track5)
    mid.tracks.append(track6)
    mid.tracks.append(track7)
    mid.tracks.append(track8)
    mid.tracks.append(track10)
    mid.tracks.append(track11)
 
    # Wav読み込み
    data_l,data_r = read_wav("test.wav")
    del data_r

    # Wavの情報取得
    wi = info_wav("test.wav")

    # ダウンサンプリング
    new_fs = 40960
    downed_data = downsampling(new_fs, data_l, wi["fs"])
    del data_l

    # ウィンドウサイズ
    win_size = 1024 * 16

    #テンポ(実データ速度に近似)
    miditempo = 480
    track0.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track1.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track2.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track3.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track4.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track5.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track6.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track7.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track8.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track10.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    track11.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))

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