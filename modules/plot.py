import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


from modules.utils import get_perpendicular_point

# def plot_quater_circle(ax, a, b):
#     k = np.linspace(0, np.pi/2, 100)
#     x = np.sin(k)
#     y = np.cos(k)
#     ax.plot(x, y)


def plot_door(ax, door_l_xy, door_r_xy):
    top_point = get_perpendicular_point(door_l_xy, door_r_xy) 
    ax.plot([top_point[0], door_l_xy[0]], [top_point[1], door_l_xy[1]], linewidth=2, color='green')
    ax.plot([top_point[0], door_r_xy[0]], [top_point[1], door_r_xy[1]], linewidth=2, color='green')
    ax.plot([door_l_xy[0], door_r_xy[0]], [door_l_xy[1], door_r_xy[1]], linewidth=2, color='green')

    return ax

def plot_window(ax, door_l_xy, door_r_xy):
    ax.plot([door_l_xy[0], door_r_xy[0]], [door_l_xy[1], door_r_xy[1]], linewidth=4, color='red')
    return ax

def plot_opendoor(ax, door_l_xy, door_r_xy):
    pass

def plot_floor_plan(corner_xy, door_xy=None, door_classes=None):
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    corner_x, corner_y = corner_xy[:, 0], corner_xy[:, 1]
    # door_x, door_y = door_xy[:, 0], door_xy[:, 1]

    ax.plot(np.hstack((corner_x, corner_x[0])), np.hstack((corner_y, corner_y[0])), linewidth=2.0)

    drawObject = Circle((0, 0), radius=0.2, fill=True, color="black")
    ax.add_patch(drawObject)

    if door_xy:
        # (x, y) to (x, y, x, y)
        # print('door_xy', door_xy)
        rs_door_xy = door_xy.reshape((-1, 4))
        
        for i, (door_i_xy, door_class) in enumerate(zip(rs_door_xy, door_classes)):

            door_l_xy = door_i_xy[:2]
            door_r_xy = door_i_xy[2:]
            # print((door_l_xy, door_r_xy, door_class))
            if door_class == 1:
                ax = plot_door(ax, door_l_xy, door_r_xy)

            if door_class == 0:
                ax = plot_window(ax, door_l_xy, door_r_xy)

            # if door_class == 2:
            #     plot_opendoor(ax, door_l_xy, door_r_xy)

    max_lim = np.max(np.abs(corner_xy)) * 1.1
    int_max_lim = int(max_lim)

    ax.set_aspect(1)
    ax.set(xlim=(-max_lim, max_lim), 
           ylim=(max_lim, -max_lim))
    ax.minorticks_on()

    ax.set_xticks(np.arange(-int_max_lim, int_max_lim + 1))
    ax.set_yticks(np.arange(-int_max_lim, int_max_lim + 1))

    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    return ax