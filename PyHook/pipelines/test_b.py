import numpy as np

name = "Test-Blue"
version = "0.0.1"
desc = "Dummy pipeline for testing. Maximizes blue channel."

def on_load() -> None:
    print(f'Pipeline="{name}" was loaded.')

def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    frame[:, :, 2] = 255
    return frame

def on_unload() -> None:
    print(f'Pipeline="{name}" was unloaded.')