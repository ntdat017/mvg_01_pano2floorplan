import os
from ultralytics import YOLO
from PIL import Image

class DoorDetector():
    def __init__(self, model_path="storage/door_yolo_best8.pt"):
        # Load a COCO-pretrained YOLOv8n model
        model = YOLO(model_path)
        
        # Display model information (optional)
        model.info()

        self.output_dir = 'storage/results'

        self.model = model

    def detect(self, image, image_name, is_vis=False):
        results = self.model.predict(image, save=False, imgsz=640, conf=0.5)
        boxes = results[0].boxes
        pred_classes = boxes.cls.cpu().numpy()
        pred_coords = boxes.xyxy.cpu().numpy()

        if is_vis:
            vis_path = os.path.join(self.output_dir, image_name + '.yolo.jpg')

            im_array = results[0].plot()
            img_vis = Image.fromarray(im_array[..., ::-1])
            img_vis.save(vis_path)

        return pred_coords, pred_classes
