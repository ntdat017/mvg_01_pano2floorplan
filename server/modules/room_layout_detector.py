import sys
sys.path.append('../../modules/HorizonNet')

import os
import json
import torch

import numpy as np
from PIL import Image

from inference import inference, HorizonNet, utils


class RoomLayoutDetector():
    def __init__(self, args):
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.args = args

        # Loaded trained model
        net = utils.load_trained_model(HorizonNet, args.pth).to(self.device)
        net.eval()

        self.net = net

    def inference(self, img_pil, image_name, is_vis=False):
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
                with open(os.path.join(self.args.output_dir, image_name + '.json'), 'w') as f:
                    json.dump({
                        'z0': float(z0),
                        'z1': float(z1),
                        'uv': [[float(u), float(v)] for u, v in cor_id],
                    }, f)

                vis_path = os.path.join(self.args.output_dir, image_name + '.raw.png')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                    .resize((vw//2, vh//2), Image.LANCZOS)\
                    .save(vis_path)
        return cor_id, z0, z1
    
    def detect(self, image):
        cor_id, z0, z1 = self.inference(image)



