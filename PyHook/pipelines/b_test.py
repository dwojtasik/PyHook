import numpy as np

name = "b_test"

def on_load() -> None:
    print(f'Pipeline="{name}" was loaded.')

def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    frame[:, :, 2] = 255
    return frame

def on_unload() -> None:
    print(f'Pipeline="{name}" was unloaded.')