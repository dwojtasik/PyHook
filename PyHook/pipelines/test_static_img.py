from utils import *

import numpy as np

with use_local_python():
    import cv2

name = "[Test] Static Image"
version = "1.0.0"
desc = "Dummy pipeline for testing. Displays static image."

settings = {
    "Image": build_variable(0, 0, 14, 1, "%COMBO[Crysis 3,CS:GO,Don't Starve,Far Cry 5,God of War,GTA V,Hollow Knight,Limbo,Minecraft,Return of the Obra Dinn,Skyrim,Tomb Raider,Trek to Yomi,Witcher 3,Zelda BoTW]Static image to be displayed.")
}

paths = [
    resolve_path('test_static_img\\crysis3.jpg'),
    resolve_path('test_static_img\\csgo.jpg'),
    resolve_path('test_static_img\\dont_starve.jpg'),
    resolve_path('test_static_img\\far_cry_5.jpg'),
    resolve_path('test_static_img\\god_of_war.jpg'),
    resolve_path('test_static_img\\gta5.jpg'),
    resolve_path('test_static_img\\hollow_knight.jpg'),
    resolve_path('test_static_img\\limbo.jpg'),
    resolve_path('test_static_img\\minecraft.jpg'),
    resolve_path('test_static_img\\return_of_the_obra_dinn.jpg'),
    resolve_path('test_static_img\\skyrim.jpg'),
    resolve_path('test_static_img\\tomb_raider.jpg'),
    resolve_path('test_static_img\\trek_to_yomi.jpg'),
    resolve_path('test_static_img\\witcher3.jpg'),
    resolve_path('test_static_img\\zelda_botw.jpg')
]

img = None

def after_change_settings(key: str, value: float) -> None:
    if key == "Image":
        global img
        img = cv2.imread(paths[int(value)])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def on_load() -> None:
    global img
    img = cv2.imread(paths[read_value(settings, "Image")])
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