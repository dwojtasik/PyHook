import numpy as np

name = "Test-Darken"
version = "0.0.1"
desc = "Dummy pipeline for testing. Darkens all channels."

def on_load() -> None:
    print(f'Pipeline="{name}" was loaded.')

def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    return frame // 3

def on_unload() -> None:
    print(f'Pipeline="{name}" was unloaded.')