# WAV to MIDI Converter in Python

[https://github.com/Bistanium/wav_to_midi_pitch_bend](https://github.com/Bistanium/wav_to_midi_pitch_bend)

<table>
	<thead>
    	<tr>
      		<th style="text-align:center"><a href="README.md">English</a></th>
			    <th style="text-align:center">日本語</th>
    	</tr>
  	</thead>
</table>

フーリエ変換を使ってwavファイルをmidiファイルに変換します。

## 必要なライブラリ
| Library   | Version |
|-----------|---------|
| mido      | 1.3.3   |
| numpy     | 2.2.6   |
| resampy   | 0.4.3   |
| scipy     | 1.16.1  |
| soundfile | 0.13.1  |
| tqdm      | 4.67.1  |

## 使い方
1. `start.bat` を実行します。
2. 処理したいwavファイルを選択します。
3. しばらく待ちます。
4. できたmidiファイルを軽量なmidiプレイヤーで再生します。

## 注意
- midiを再生するときは付属の正弦波サウンドフォントを使ってください。

## オプション
このプログラムにはユーザの目的に応じてコードを変更できる場所があります。

### ピッチベンドを無効にする方法
- 19行目の `disable_pitch_bend = False` を `disable_pitch_bend = True` に変更します。
> インターネット上にはピッチベンドを使わないで演奏されているものがあります。それらと同じようにしたい場合はこのように設定を変更してください。

### 軽量化したい場合
再生時の負荷を軽減したい場合は次のように変更してください。
- 23行目の `minimum_velocity = 4` を `minimum_velocity = 8` に変更します。
- 21行目の `similar_velocity_threshold = 4` を `similar_velocity_threshold = 8` に変更します。
> これらの値は好みに合わせて調整してください。ただし、値を大きくすると音質の低下を感じるかもしれません。
