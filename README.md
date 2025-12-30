# WAV to MIDI Converter in Python

[https://github.com/Bistanium/wav_to_midi_pitch_bend](https://github.com/Bistanium/wav_to_midi_pitch_bend)

<table>
	<thead>
    	<tr>
      		<th style="text-align:center">English</th>
      		<th style="text-align:center"><a href="README_ja.md">日本語</a></th>
    	</tr>
  	</thead>
</table>

Convert WAV to MIDI using Fourier transform

## Required Libraries
| Library   | Version |
|-----------|---------|
| mido      | 1.3.3   |
| numpy     | 2.2.6   |
| resampy   | 0.4.3   |
| scipy     | 1.16.1  |
| soundfile | 0.13.1  |
| tqdm      | 4.67.1  |

## How to Use
1. Run the `start.bat` file.
2. Select a wav file (any file you want to process).
3. Wait for the process to finish.
4. Play the generated MIDI file using a lightweight MIDI player.

## Note
- Please use the included sine wave soundfont when playing the MIDI file.

## Optional Settings
At the top of the code, there are several configuration options that users may freely adjust depending on their purpose.

### How to Disable Pitch Bend
- On line 19 of the program, change `disable_pitch_bend = False` to `disable_pitch_bend = True`.
> Some performances available on the internet are played without using pitch bend. If you want the output to behave similarly, please enable this setting.

### How to Reduce Processing Load (Lightweight Mode)
If you want to reduce processing load, you can adjust the following parameters:
- On line 23, change `minimum_velocity = 4` to `minimum_velocity = 8`.
- On line 21, change `similar_velocity_threshold = 4` to `similar_velocity_threshold = 8`.
> You may adjust these values to your preference. Please note that increasing these values may reduce sound quality.
