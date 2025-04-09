import imageio
import os

with imageio.get_writer("pso_no_pbest_change.gif", mode='I', duration=0.02) as writer:
    for filename in sorted(os.listdir("./img")):
        if filename.endswith(".png") and filename.startswith("frame_"):
            image = imageio.imread(f"./img/{filename}")
            writer.append_data(image)
