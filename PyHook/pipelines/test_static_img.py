from utils import *

import numpy as np

with use_local_python():
    import cv2

name = "[Test] Static Image"
version = "0.0.1"
desc = "Dummy pipeline for testing. Displays static image."

img = None

def on_load() -> None:
    global img
    image_path = resolve_path('test_static_img/test.jpg')
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f'Pipeline="{name}" was loaded.')

def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    global img
    if width != img.shape[1] or height != img.shape[0]:
        img = cv2.resize(img, (width, height))
    return img.copy()

def on_unload() -> None:
    global img
    del img
    img = None
    print(f'Pipeline="{name}" was unloaded.')