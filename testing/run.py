import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
import subprocess


def process_audio(audio_path, output_dir="segments", scp_file="wav.scp"):

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(audio_path)

    
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)

    scp_lines = []
    segment_count = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        duration = end_ms - start_ms

        if duration < 300: 
            continue

        segment = audio[start_ms:end_ms]
        segment_name = f"seg_{segment_count}"
        segment_path = os.path.join(output_dir, f"{segment_name}.wav")
        segment.export(segment_path, format="wav")
        scp_lines.append(f"{segment_name} {os.path.abspath(segment_path)}")
        segment_count += 1


    with open(scp_file, "w") as f:
        f.write("\n".join(scp_lines))
    
    subprocess.run([
        "wespeaker",
        "--task", "embedding_kaldi",
        "--wav_scp", scp_file,
        "--output_file", "embeddings/segments"
    ], check=True)



process_audio("testing/audio/sample.wav")
