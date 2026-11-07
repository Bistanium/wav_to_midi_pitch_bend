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

## How to Use
1. Run the `wav2midi.exe` file.
2. Select a wav file (any file you want to process).
3. Wait for the process to finish.
4. Play the generated MIDI file using a lightweight MIDI player.

## Note
- Please use the included sine wave soundfont when playing the MIDI file.

## Optional Settings
You can adjust the settings in this program to suit your needs.

Settings can be modified via the `settings.txt` file, which is located in the same directory as `wav2midi.exe`.

If `settings.txt` does not exist, it will be generated when you run `wav2midi.exe`.

### How to Disable Pitch Bend
- Change `disable_pitch_bend = false` to `disable_pitch_bend = true`.
> Some performances available on the internet are played without using pitch bend. If you want the output to behave similarly, please enable this setting.

### How to Reduce Processing Load (Lightweight Mode)
If you want to reduce processing load, you can adjust the following parameters:
- Change `minimum_velocity = 2` to `minimum_velocity = 8`.
- Change `similar_velocity_threshold = 4` to `similar_velocity_threshold = 8`.
> You may adjust these values to your preference. Please note that increasing these values may reduce sound quality.
