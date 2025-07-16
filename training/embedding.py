import os

folder = "speakers/speaker10"
scp_file = "wav.scp"

with open(scp_file, "w") as f:
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            clip_id = os.path.splitext(file)[0].replace(" ", "_")
            full_path = os.path.abspath(os.path.join(folder, file))
            f.write(f"{clip_id} {full_path}\n")
