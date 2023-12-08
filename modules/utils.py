import os
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import datetime

PI = float(np.pi)


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])


def project_point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result

def intersection_point_on_line(p1, p2, p3, p4=(0, 0)):
    # Line 1 dy, dx and determinant
    a11 = (p1[1] - p2[1])
    a12 = (p2[0] - p1[0])
    b1 = (p1[0]*p2[1] - p2[0]*p1[1])

    # Line 2 dy, dx and determinant
    a21 = (p3[1] - p4[1])
    a22 = (p4[0] - p3[0])
    b2 = (p3[0]*p4[1] - p4[0]*p3[1])

    # Construction of the linear system
    # coefficient matrix
    A = np.array([[a11, a12],
                [a21, a22]])

    # right hand side vector
    b = -np.array([b1, b2])
    # solve
    try:
        intersection_point = np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        print('No single intersection point detected')
        return p3
    return intersection_point

def get_perpendicular_point(a, b):
    ab = b - a
    x, y = ab
    c = a + np.array([-y, x])
    return c

def get_session_name(image_path):
    now=datetime.datetime.now()
    session_dir = now.isoformat()
    image_name = os.path.basename(image_path)

    vis_name = os.path.join(session_dir, image_name)

    return image_name, vis_name
