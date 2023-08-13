
import requests
from PIL import Image
import gradio as gr

from modules.room_layout_transfomer import Pano2FloorPlan

room = Pano2FloorPlan()

output_dir= 'storage/results'

def predict(image_pil):
    image_path = 'image'
    img_floor_plan= room.process(img_pil=image_pil, image_path=image_path, save_dir=output_dir)
    return img_floor_plan


gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=['image'],
             examples=["storage/asserts/floor_01_partial_room_01_pano_15.jpg", "storage/asserts/floor_01_partial_room_07_pano_19.jpg"]).launch()