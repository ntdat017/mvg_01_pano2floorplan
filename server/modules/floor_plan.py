import os
import numpy as np
from PIL import Image

from .room_layout_detector import RoomLayoutDetector
from .door_detector import DoorDetector

from .utils import np_coorx2u, np_coor2xy, project_point_on_line, PI
from .plot import plot_floor_plan

door_config = {
    'model_path': ''
}
room_layout_config = {
    'args': {
        'pth': '',
        'cpu': True,
        'device': 'cuda',
        # ''
    }
}

class Pano2FloorPlan():
    def __init__(self):
        self.door_detector = DoorDetector()
        self.room_layout_detector = RoomLayoutDetector()

    def process(self, image_path, save_file=''):
        image_name = os.path.basename(image_path)
        img_pil = Image.open(image_path)
        W, H = img_pil.size

        # door_coords: (top, left, bottom, right)
        door_coords, door_classes = self.door_detector.detect(img_pil)

        # corner_coords: (x, y)
        corner_coords, _, _ = self.room_layout_detector.inference(img_pil, image_name, is_vis=True)

        corner_xy, door_xy = self.coord2xy(corner_coords, door_coords)

        ax = plot_floor_plan(corner_xy, door_xy, door_classes)
        ax.savefig('temp.png')


    def coord2xy(self, corner_coords, door_coords, W, H):
        floor_door_z = -1.6

        corner_xy = np_coor2xy(corner_coords, floor_door_z, W, H, floorW=1, floorH=1)

        # reshape door coord (x, y)
        rs_door_coords = np.reshape(door_coords, (-1, 2))
        door_xy = np_coor2xy(rs_door_coords, floor_door_z, W, H, floorW=1, floorH=1)

        door_wall_mapping = self.mapping_door_in_wall(door_coords, corner_coords, H, W)

        for door_i, wall_i in enumerate(door_wall_mapping):
            corner_l_xy = corner_xy[wall_i]
            corner_r_xy = corner_xy[wall_i + 1 % len(corner_xy)] # roll back to index 0 if wall_i equal lenght of corner_xy
            
            door_l_xy = door_xy[2 * door_i]
            door_r_xy = door_xy[2 * door_i + 1]
            door_xy[2 * door_i] = project_point_on_line(corner_l_xy, corner_r_xy, door_l_xy)
            door_xy[2 * door_i + 1] = project_point_on_line(door_r_xy, corner_r_xy, door_l_xy)

        return corner_xy, door_xy


    def mapping_door_in_wall(self, door_coords, corner_coords, H, W):
        " corner_x meaning x coordinate of corner "

        door_center_x = (door_coords[:, 0] + door_coords[:, 2])/2
        is_z_view_opendoor = (door_coords[:, 2] - door_coords[:, 0]) > W/2

        door_center_u = np_coorx2u(door_center_x, coorW=W)
        door_center_u[is_z_view_opendoor] += PI

        corner_x = corner_coords[:, 0]
        corner_u = np_coorx2u(corner_x, coorW=W)
        corner_u_inc_1 = np.append(corner_u[1:], corner_u[0] + 2*PI)

        door_wall_mapping = np.array([-1] * len(door_center_u))

        for wall_i, (cor_u0, cor_u1) in enumerate(zip(corner_u, corner_u_inc_1)):
            # print(i, cor_u0, cor_u1)
            is_in_wall_i = (door_center_u < cor_u1) * (door_center_u > cor_u0)
            door_wall_mapping[is_in_wall_i] = wall_i
            # print(idx)
        assert np.any(door_wall_mapping == -1), "Mapping door in wall False"
        return door_wall_mapping


