import numpy as np

from utils import *

with use_local_python():
    import cv2

name = "[AI] DNN Super Resolution"
version = "0.0.1"
desc = """OpenCV pipeline for DNN super resolution."""
multistage = 2

settings = {
    "Model": build_variable(1, 0, 2, 1, "%COMBO[ESPCN,FSRCNN,FSRCNN-small]Super-Resolution DNN model.\
ESPCN - Efficient Sub-pixel Convolutional Neural Network\
FSRCNN - Fast Super-Resolution Convolutional Neural Network\
FSRCNN-small - FSRCNN with fewer parameters"),
    "Scale": build_variable(4, 2, 4, 1, "Scale multiplier.")
}

models = [
    (resolve_path("ai_dnn_super_resolution\\ESPCN_x"), "espcn"),
    (resolve_path("ai_dnn_super_resolution\\FSRCNN_x"), "fsrcnn"),
    (resolve_path("ai_dnn_super_resolution\\FSRCNN-small_x"), "fsrcnn")
]

sr = None

def setup_model() -> None:
    global sr
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model = read_value(settings, "Model")
    scale = read_value(settings, "Scale")
    sr.readModel(f'{models[model][0]}{scale}.pb')
    sr.setModel(models[model][1], scale)

def after_change_settings(key: str, value: float) -> None:
    if key == "Scale" or key == "Model":
        setup_model()

def on_load() -> None:
    setup_model()
    print(f'Pipeline="{name}" was loaded.')

def on_frame_process_stage(frame: np.array, width: int, height: int, frame_num: int, stage: int) -> np.array:
    scale = read_value(settings, "Scale")
    if stage == 1:
        if scale == 3:
            return cv2.resize(frame, (width // scale + 1, height // scale + 1))
        return cv2.resize(frame, (width // scale, height // scale))
    if scale == 3:
        out_w = (width * 3) // 10 * 10
        out_h = (height * 3) // 10 * 10
        return sr.upsample(frame)[:out_h,:out_w,:]
    return sr.upsample(frame)

def on_unload() -> None:
    global sr
    sr = None
    print(f'Pipeline="{name}" was unloaded.')
