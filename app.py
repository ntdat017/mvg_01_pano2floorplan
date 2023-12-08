
import requests
from PIL import Image
import os
import gradio as gr

from modules.pano2floorplan import Pano2FloorPlan
from modules.bird_view import export_bird_view

room = Pano2FloorPlan()

output_dir= 'storage/results'

def predict(image_pil):
    image_path = 'image'
    img_floor_plan, img_bird_view = room.process(img_pil=image_pil, image_path=image_path, save_dir=output_dir)
    return img_floor_plan, img_bird_view


# gr.Interface(fn=predict,
#              inputs=gr.Image(type="pil"),
#              outputs=['image', 'image'],
#              examples=["storage/asserts/floor_01_partial_room_01_pano_15.jpg", "storage/asserts/floor_01_partial_room_07_pano_19.jpg"]).launch(server_name='0.0.0.0', server_port=6870, show_api=True)


with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column(scale=1):


            number_1 = gr.Number(label="FOV degree", value=115)
            number_2 = gr.Number(label="U degree", value=-90)
            number_3 = gr.Number(label="V degree", value=0)

            btn_1 = gr.Button(value="Submit")
            btn_2 = gr.Button(value="Bird view Image")
 
        with gr.Column(scale=2):
            img_input = gr.Image(type="pil", lines=2)
    with gr.Row():
        with gr.Column(scale=2):
            im = gr.Image(label="Output Floor Plan")
        with gr.Column(scale=2):
            im_2 = gr.Image(label="Output Bird View")

    btn_1.click(predict, inputs=[img_input], outputs=[im, im_2])
    btn_2.click(export_bird_view, inputs=[img_input, number_1, number_2, number_3], outputs=[im_2])

    gr.Examples(
        examples=[["storage/asserts/floor_01_partial_room_01_pano_15.jpg", 115, -90, 0], ["storage/asserts/floor_01_partial_room_07_pano_19.jpg", 115, -90, 0]],
        inputs=[img_input, number_1, number_2, number_3],
        outputs=[im_2],
        fn=export_bird_view,
        cache_examples=True,
    )


demo.launch(server_name='0.0.0.0', server_port=6870)
