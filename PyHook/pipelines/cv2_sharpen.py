from pipeline_utils import *
import numpy as np

with use_local_python():
    import cv2

name = "[CV2] Sharpen"
version = "1.2.9"
desc = "OpenCV sharpen filter"
supports = [64]

settings = {"Amount": build_variable(1.0, 0.0, 5.0, 0.1, "Amount of sharpening to apply.")}


def on_load() -> None:
    print(f'Pipeline="{name}" was loaded with CPU support.')


def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    amount = read_value(settings, "Amount")
    blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
    return cv2.addWeighted(frame, amount + 1, blurred, -amount, 0)


def on_unload() -> None:
    print(f'Pipeline="{name}" was unloaded.')
