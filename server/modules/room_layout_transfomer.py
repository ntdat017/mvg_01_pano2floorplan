import numpy as np
from PIL import Image

from .horizonnet_wrapper import HorizonNetWrapper
from .door_detector import DoorDetector

from .utils import np_coorx2u, np_coor2xy, project_point_on_line, intersection_point_on_line, get_session_name, PI
from .plot import plot_floor_plan



class Pano2FloorPlan():
    def __init__(self):
        self.door_detector = DoorDetector()
        self.horizonnet_wrapper = HorizonNetWrapper()

    def process(self, image_path, save_dir='', img_pil=None):
        image_name, vis_name = get_session_name(image_path)
        if not img_pil:
            img_pil = Image.open(image_path)

        # preprocessed
        img_preprocessed_pil = self.horizonnet_wrapper.preprocess(img_pil, vis_name, is_vis=True)
        
        # corner_coords: (x, y)
        corner_coords, _, _ = self.horizonnet_wrapper.inference(img_preprocessed_pil, vis_name, is_vis=True)

        # door_coords: (top, left, bottom, right)
        door_coords, door_classes = self.door_detector.detect(img_preprocessed_pil, vis_name, is_vis=True)

        W, H = img_preprocessed_pil.size

        corner_xy, door_xy = self.transform_coord2xy(corner_coords, door_coords, W, H)

        ax = plot_floor_plan(corner_xy, door_xy, door_classes)
        ax.figure.savefig(f'{save_dir}/{vis_name}.floorplan.png')

        img_floor_plan = Image.frombytes('RGB', ax.figure.canvas.get_width_height(), ax.figure.canvas.tostring_rgb())
        return img_floor_plan


    def transform_coord2xy(self, corner_coords, door_coords, W, H):
        floor_door_z = -1.6

        corner_xy = np_coor2xy(corner_coords, floor_door_z, W, H, floorW=1, floorH=1)

        # reshape door coord (x, y)
        door_coords[:, 1] = door_coords[:, 3]
        rs_door_coords = np.reshape(door_coords, (-1, 2))
        door_xy = np_coor2xy(rs_door_coords, floor_door_z, W, H, floorW=1, floorH=1)

        # print('corner_coords', corner_coords)
        # print('door_coords', door_coords)

        # print('corner_xy', corner_xy)
        # print('door_xy', door_xy)

        door_wall_mapping = self.mapping_door_in_wall(door_coords, corner_coords, H, W)

        wall_xy = np.vstack((corner_xy[-1], corner_xy))

        for door_i, wall_i in enumerate(door_wall_mapping):
            corner_l_xy = wall_xy[wall_i]
            corner_r_xy = wall_xy[wall_i + 1] # roll back to index 0 if wall_i equal lenght of corner_xy
            
            door_l_xy = door_xy[2 * door_i]
            door_r_xy = door_xy[2 * door_i + 1]
            door_xy[2 * door_i] = intersection_point_on_line(corner_l_xy, corner_r_xy, door_l_xy)
            # print('count')
            # print(corner_l_xy, corner_r_xy, door_l_xy)
            # print(project_point_on_line(corner_l_xy, corner_r_xy, door_l_xy))

            door_xy[2 * door_i + 1] = intersection_point_on_line(door_l_xy, corner_r_xy, door_r_xy)
            # print(door_r_xy, corner_r_xy, door_l_xy)
            # print(project_point_on_line(door_l_xy, corner_r_xy, door_r_xy))
            # break

        return corner_xy, door_xy


    def mapping_door_in_wall(self, door_coords, corner_coords, H, W):
        " corner_x meaning x coordinate of corner "

        corner_x = corner_coords[:, 0]
        corner_u = np_coorx2u(corner_x, coorW=W)
        corner_u_dsc_1 = np.insert(corner_u[:-1], 0, corner_u[-1] - 2*PI)

        door_center_x = (door_coords[:, 0] + door_coords[:, 2])/2
        is_z_view_opendoor = (door_coords[:, 2] - door_coords[:, 0]) > W/2

        door_center_u = np_coorx2u(door_center_x, coorW=W)
        door_center_u[is_z_view_opendoor] -= PI
        door_center_u[door_center_u > corner_u[-1]] -= 2 * PI


        door_wall_mapping = np.array([-1] * len(door_center_u))

        # print(door_center_u)
        for wall_i, (cor_u0, cor_u1) in enumerate(zip(corner_u_dsc_1, corner_u)):
            # print(wall_i, cor_u0, cor_u1)
            is_in_wall_i = (door_center_u < cor_u1) * (door_center_u > cor_u0)
            door_wall_mapping[is_in_wall_i] = wall_i
        #     print(is_in_wall_i)
        # print(door_wall_mapping)
        assert np.all(door_wall_mapping != -1), "Mapping door in wall False"
        return door_wall_mapping
