from utils import *
import numpy as np

with use_local_python():
    import cv2

name = "[CV2] Sharpen"
version = "0.8.1"
desc = "OpenCV sharpen filter"

settings = {"Amount": build_variable(1.0, 0.0, 5.0, 0.1, "Amount of sharpening to apply.")}

has_cuda = False
blur_mat = None
mult1_mat = None
mult2_mat = None
last_data = None


def on_load() -> None:
    global has_cuda, blur_mat, mult1_mat, mult2_mat, cuda_blur
    has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if has_cuda:
        blur_mat = cv2.cuda_GpuMat()
        mult1_mat = cv2.cuda_GpuMat()
        mult2_mat = cv2.cuda_GpuMat()
        cuda_blur = cv2.cuda.createGaussianFilter(
            srcType=cv2.CV_32FC3, dstType=cv2.CV_32FC3, ksize=(5, 5), sigma1=1, sigma2=0
        )
    print(f'Pipeline="{name}" was loaded with {"GPU" if has_cuda else "CPU"} support.')


def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    global has_cuda, blur_mat, mult1_mat, mult2_mat, cuda_blur, last_data
    amount = read_value(settings, "Amount")

    if has_cuda:
        blur_mat.upload(frame.astype(np.float32))
        frame_mat = blur_mat.clone()
        cv2.cuda_Filter.apply(cuda_blur, blur_mat, blur_mat)

        new_data = (frame.shape, amount)
        if new_data != last_data:
            last_data = (frame.shape, amount)
            mult1_mat.upload(np.full(frame.shape, amount + 1, dtype=np.float32))
            mult2_mat.upload(np.full(frame.shape, amount, dtype=np.float32))

        cv2.cuda.multiply(frame_mat, mult1_mat, frame_mat)
        cv2.cuda.multiply(blur_mat, mult2_mat, blur_mat)
        cv2.cuda.subtract(frame_mat, blur_mat, frame_mat)
        sharpened = frame_mat.download()
    else:
        blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
        sharpened = float(amount + 1) * frame - float(amount) * blurred

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def on_unload() -> None:
    global has_cuda, blur_mat, mult1_mat, mult2_mat, last_data
    if has_cuda:
        if blur_mat is not None:
            blur_mat.release()
        if mult1_mat is not None:
            mult1_mat.release()
        if mult2_mat is not None:
            mult2_mat.release()
        if cuda_blur is not None:
            cuda_blur.clear()
    has_cuda = False
    blur_mat = None
    mult1_mat = None
    mult2_mat = None
    last_data = None
    print(f'Pipeline="{name}" was unloaded.')
