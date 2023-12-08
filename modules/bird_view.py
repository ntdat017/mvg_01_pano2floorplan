import numpy as np
from py360convert import e2p

import time

PI = float(np.pi)

FOV_DEG = 115 # degree
V_DEG = -90
U_DEG = 0
OUT_HW = (512, 512)

def export_bird_view(pano_image_pil, fov_deg=FOV_DEG, v_deg=V_DEG, u_deg=U_DEG):
    pano_image_np = np.array(pano_image_pil)
    tt = time.time()

    bird_view_image = e2p(pano_image_np, fov_deg=(fov_deg, fov_deg), u_deg=u_deg, v_deg=v_deg, out_hw=OUT_HW, in_rot_deg=0, mode='bilinear')
    print(time.time() - tt)

    return bird_view_image

