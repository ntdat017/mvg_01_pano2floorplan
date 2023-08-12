import numpy as np
import matplotlib.pyplot as plt

from .utils import get_perpendicular_point

# def plot_quater_circle(ax, a, b):
#     k = np.linspace(0, np.pi/2, 100)
#     x = np.sin(k)
#     y = np.cos(k)
#     ax.plot(x, y)


def plot_door(ax, door_l_xy, door_r_xy):
    top_point = get_perpendicular_point(door_l_xy, door_r_xy) 
    ax.plot([top_point[0], door_l_xy[0]], [top_point[1], door_l_xy[1]], linewidth=3, color='green')
    ax.plot([top_point[0], door_r_xy[0]], [top_point[1], door_r_xy[1]], linewidth=3, color='green')

    return ax

def plot_window(ax, door_l_xy, door_r_xy):
    ax.plot([door_l_xy[0], door_l_xy[0]], [door_l_xy[1], door_l_xy[1]], linewidth=5, color='red')
    return ax

def plot_opendoor(ax, door_l_xy, door_r_xy):
    pass

def plot_floor_plan(corner_xy, door_xy, door_classes):
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    corner_x, corner_y = corner_xy[:, 0], corner_xy[:, 1]
    door_x, door_y = door_xy[:, 0], door_xy[:, 1]
    ax.plot(corner_x, door_y)
    
    for i, (door_l_xy, door_class) in enumerate(zip(door_xy, door_classes)):
        door_r_xy = door_xy[(i + 1) % len(door_xy)]
        if door_class == 0:
            ax = plot_door(ax, door_l_xy, door_r_xy)

        if door_class == 1:
            ax = plot_window(ax, door_l_xy, door_r_xy)

        # if door_class == 2:
        #     plot_opendoor(ax, door_l_xy, door_r_xy)
    return ax