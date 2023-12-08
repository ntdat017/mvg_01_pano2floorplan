
from pathlib import Path

import os
import json
import torch

import numpy as np
from PIL import Image

from modules.HorizonNet.inference import inference, HorizonNet, utils
from modules.HorizonNet.misc.pano_lsd_align import panoEdgeDetection, rotatePanorama


CURRENT_DIR = Path(os.path.abspath(__file__)).parent.parent.parent.absolute()
print(CURRENT_DIR)

args = {
    'cpu': False,
    'model_path': f'{CURRENT_DIR}/storage/resnet50_rnn__zind.pth',
    'flip': False,
    'rotate': [],
    'visualize': True,
    'force_cuboid': False,
    'force_raw': False,
    'min_v': None,
    'r': 0.05,
    # preprocess params
    'q_error': 0.7,
    'refine_iter': 3,

    'output_dir': 'storage/results',
}
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

ARGS = Struct(**args)

class HorizonNetWrapper():
    def __init__(self, args=ARGS):
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.args = args

        # Loaded trained model
        net = utils.load_trained_model(HorizonNet, args.model_path).to(self.device)
        net.eval()

        self.net = net

    def inference(self, img_pil, image_name, is_vis=False):
        W, H = img_pil.size
        # Inferencing
        with torch.no_grad():
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # Inferenceing corners
            cor_id, z0, z1, vis_out = inference(net=self.net, x=x, device=self.device,
                                                flip=self.args.flip, rotate=self.args.rotate,
                                                visualize=self.args.visualize,
                                                force_cuboid=self.args.force_cuboid,
                                                force_raw=self.args.force_raw,
                                                min_v=self.args.min_v, r=self.args.r)


            if vis_out is not None and is_vis:
                # Output result
                with open(os.path.join(self.args.output_dir, image_name + '.corner.json'), 'w') as f:
                    json.dump({
                        'z0': float(z0),
                        'z1': float(z1),
                        'uv': [[float(u), float(v)] for u, v in cor_id],
                    }, f)

                vis_path = os.path.join(self.args.output_dir, image_name + '.corner.jpg')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                    .resize((vw//2, vh//2), Image.LANCZOS)\
                    .save(vis_path)
            
            cor_id = np.array(cor_id, np.float32)
            cor_id[:, 0] *= W
            cor_id[:, 1] *= H

        # Get only floor's corner
        cor_id = cor_id[1::2]

        return cor_id, z0, z1
    
    def detect(self, image):
        cor_id, z0, z1 = self.inference(image)

    def preprocess(self, img_pil, image_name, is_vis=False):          
        print('Processing', image_name, flush=True)

        # Load and cat input images
        img_ori = np.array(img_pil.resize((1024, 512), Image.BICUBIC))[..., :3]

        # VP detection and line segment extraction
        _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(img_ori,
                                                        qError=self.args.q_error,
                                                        refineIter=self.args.refine_iter)
        panoEdge = (panoEdge > 0)

        # Align images with VP
        i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])
        l_img = rotatePanorama(panoEdge.astype(np.float32), vp[2::-1])

        i_img_pil = Image.fromarray((i_img * 255).astype(np.uint8))
        l_img_pil = Image.fromarray((l_img * 255).astype(np.uint8))

        # Dump results
        if is_vis:
            path_VP = os.path.join(self.args.output_dir, '%s_VP.txt' % image_name)
            path_raw_img = os.path.join(self.args.output_dir, '%s.raw.jpg' % image_name)
            path_i_img = os.path.join(self.args.output_dir, '%s_preprocessed_rgb.jpg' % image_name)
            path_l_img = os.path.join(self.args.output_dir, '%s_preprocessed_line.jpg' % image_name)

            os.makedirs(os.path.dirname(path_VP), exist_ok=True)

            with open(path_VP, 'w') as f:
                for i in range(3):
                    f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
            img_pil.save(path_raw_img)
            i_img_pil.save(path_i_img)
            l_img_pil.save(path_l_img)

        return i_img_pil
