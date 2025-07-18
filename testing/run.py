import os
import subprocess
import json
import kaldiio
import numpy as np
from pydub import AudioSegment
from pyannote.core import Segment
from pyannote.audio import Pipeline
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import gradio as gr
import tempfile


def process_audio(audio_path, output_dir="segments", embeddings_dir="embeddings", scp_file="wav.scp", mapping_file="segment_mapping.json", window_size=1, step_size=0.5):
    # Run VAD to detect speech regions
    vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=True)
    vad_output = vad_pipeline(audio_path)
    speech_regions = vad_output.get_timeline()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Load and resample audio
    audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)

    scp_lines = []
    segment_mapping = []
    segment_count = 0

    # Slice audio into overlapping segments
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

    # Save wav.scp and segment mapping
    with open(scp_file, "w") as f:
        f.write("\n".join(scp_lines))

    with open(mapping_file, "w") as f:
        json.dump(segment_mapping, f)

    # Run WeSpeaker to generate embeddings
    subprocess.run([
        "wespeaker",
        "--task", "embedding_kaldi",
        "--wav_scp", scp_file,
        "--output_file", os.path.join(embeddings_dir, "segments")
    ], check=True)


def load_segment_mapping(mapping_path):
    # Load JSON mapping of segment times
    with open(mapping_path, "r") as f:
        segment_mapping = json.load(f)
    return [(float(s), float(e), name) for s, e, name in segment_mapping]


def load_embeddings(ark_path):
    # Load embeddings from ark file
    embeddings = []
    segment_names = []
    for key, vec in kaldiio.load_ark(ark_path):
        vec = np.array(vec)
        if np.isnan(vec).any():
            print(f"Skipping {key}: contains NaN")
            continue
        embeddings.append(vec)
        segment_names.append(key)

    if not embeddings:
        raise ValueError("No valid embeddings found.")

    embeddings = normalize(np.stack(embeddings), norm='l2')
    return embeddings, segment_names


def spectral_cluster(embeddings, n_speakers):
    # Cluster embeddings based on cosine similarity
    similarity = cosine_similarity(embeddings)
    clusterer = SpectralClustering(
        n_clusters=n_speakers,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = clusterer.fit_predict(similarity)
    return labels


def merge_segments(segments, labels):
    # Merge  segments with the same speaker label
    if not segments:
        return []

    merged = []
    prev_start, prev_end = segments[0]
    prev_label = labels[0]

    for i in range(1, len(segments)):
        start, end = segments[i]
        label = labels[i]

        if label == prev_label and start <= prev_end:
            prev_end = max(prev_end, end)
        else:
            merged.append((prev_start, prev_end, prev_label))
            prev_start, prev_end, prev_label = start, end, label

    merged.append((prev_start, prev_end, prev_label))
    return merged


def run_clustering(ark_path, mapping_path, n_speakers):
    # Run full clustering pipeline
    segment_mapping = load_segment_mapping(mapping_path)
    embeddings, segment_names = load_embeddings(ark_path)
    name_to_segment = {name: (start, end) for start, end, name in segment_mapping}
    segments_ordered = [name_to_segment[name] for name in segment_names]

    labels = spectral_cluster(embeddings, n_speakers)
    merged_segments = merge_segments(segments_ordered, labels)
    return merged_segments, labels


def gradio_interface(audio_file, n_speakers):
    # Gradio wrapper with temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        segments_dir = os.path.join(temp_dir, "segments")
        embeddings_dir = os.path.join(temp_dir, "embeddings")
        os.makedirs(segments_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

        scp_file = os.path.join(temp_dir, "wav.scp")
        mapping_file = os.path.join(temp_dir, "segment_mapping.json")
        embeddings_ark = os.path.join(embeddings_dir, "segments.ark")

        audio_path = audio_file

        process_audio(audio_path, output_dir=segments_dir, embeddings_dir=embeddings_dir, scp_file=scp_file, mapping_file=mapping_file)
        merged_segments, labels = run_clustering(embeddings_ark, mapping_file, n_speakers=int(n_speakers))

        # Format output for UI
        timeline_text = "\n".join([f"{start:.2f}s - {end:.2f}s → Speaker {label}" for start, end, label in merged_segments])
        return timeline_text

    finally:
        pass


# Gradio UI setup
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Audio(label="Upload Audio", type="filepath"),
        gr.Number(value=2, label="Number of Speakers")
    ],
    outputs=[
        gr.Textbox(label="Speaker Timeline", lines=20)
    ],
    title="Speaker Diarization Live Demo",
    description="Upload an audio file, specify the number of speakers, and get diarized speaker timestamps.",
    allow_flagging="never" 
)

if __name__ == "__main__":
    iface.launch()
