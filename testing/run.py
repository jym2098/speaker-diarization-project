import os
import subprocess
import json
from pydub import AudioSegment
from pyannote.core import Segment
from pyannote.audio import Pipeline

def process_audio(audio_path, output_dir="segments", scp_file="wav.scp", mapping_file="segment_mapping.json",
                  window_size=1.5, step_size=0.75):

    # Step 1: Use PyAnnote VAD pipeline
    vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=True)
    vad_output = vad_pipeline(audio_path)  # returns Annotation object
    speech_regions = vad_output.get_timeline()  # returns Timeline

    # Prepare directories and audio
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)

    scp_lines = []
    segment_mapping = []
    segment_count = 0

    # Step 2: Sliding window over VAD speech regions
    for region in speech_regions:
        t = region.start
        while t + window_size <= region.end:
            seg = Segment(t, t + window_size)
            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)

            segment = audio[start_ms:end_ms]
            segment_name = f"seg_{segment_count}"
            segment_path = os.path.join(output_dir, f"{segment_name}.wav")
            segment.export(segment_path, format="wav")

            scp_lines.append(f"{segment_name} {os.path.abspath(segment_path)}")
            segment_mapping.append((seg.start, seg.end, segment_name))
            segment_count += 1

            t += step_size

    # Step 3: Write wav.scp for WeSpeaker
    with open(scp_file, "w") as f:
        f.write("\n".join(scp_lines))

    # Step 4: Save segment mapping to JSON for clustering later
    with open(mapping_file, "w") as f:
        json.dump(segment_mapping, f)

    # Step 5: Run WeSpeaker to extract embeddings
    subprocess.run([
        "wespeaker",
        "--task", "embedding_kaldi",
        "--wav_scp", scp_file,
        "--output_file", "embeddings/segments"
    ], check=True)


# Run the function
process_audio("testing/audio/sample.wav")
