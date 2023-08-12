from ultralytics import YOLO
from PIL import Image

class DoorDetector():
    def __init__(self, model_path):
        # Load a COCO-pretrained YOLOv8n model
        model = YOLO(model_path)
        
        # Display model information (optional)
        model.info()

        self.model = model

    def detect(self, image):
        results = self.model.predict(image, save=True, imgsz=640, conf=0.5)
        pred_classes = results.cls.cpu().numpy()
        pred_coords = results.xyxy.cpu().numpy()

        return pred_coords, pred_classes
