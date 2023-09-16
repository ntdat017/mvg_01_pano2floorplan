import os
from PIL import Image
import time

from modules.pano2floorplan import Pano2FloorPlan


if __name__ == '__main__':

    save_dir = 'storage/results'
    image_path = "storage/asserts/floor_01_partial_room_07_pano_19.jpg"
    image_name = os.path.basename(image_path)
    image_pil = Image.open(image_path)

    room = Pano2FloorPlan()

    for i in range(4):
        tt = time.time()
        room.process(image_path, img_pil=image_pil, save_dir=save_dir)
        print('time', time.time() - tt)