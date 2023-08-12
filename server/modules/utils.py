import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

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


def get_perpendicular_point(a, b):
    ab = b - a
    x, y = ab
    c = a + np.array([-y, x])
    return c
