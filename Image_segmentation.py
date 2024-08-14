import cv2
from ultralytics import YOLO
import numpy as np

class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
   
    def detect(self, img):
        height, width, channels = img.shape

        # Use the model to predict
        results = self.model.predict(source=img.copy(), save=True)
        result = results[0]
        
        # Prepare lists to hold detection results
        bboxes = []
        classes = []
        segmentation = []
        scores = []

        if hasattr(result, 'masks'):
            masks = result.masks.data
            print(masks)
            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy()
                segmentation.append(mask_np)
                mask_img = np.uint8(mask_np * 255)
                

                # Find bounding box from mask
                x, y, w, h = cv2.boundingRect(mask_img)
                bboxes.append([x, y, x + w, y + h])

                # Store class and score
                classes.append(result.names[i])
                scores.append(result.boxes.conf[i].item())

                # Optionally display the mask
                cv2.imshow(f"Mask {i}", mask_img)
                cv2.waitKey(0)
                
        return bboxes, classes, segmentation, scores

# Initialize YOLO segmentation detector
ys = YOLOSegmentation("yolov8m-seg.pt")

# Read and resize the image
img = cv2.imread("Cat.jpeg")
img = cv2.resize(img, (800, 600))
cv2.imshow("Image", img)

# Detect objects and segmentations
bboxes, classes, segmentation, scores = ys.detect(img)

print("Bounding Boxes:", bboxes)
print("Classes:", classes)
print("Segmentation:", segmentation)
print("Scores:", scores)

cv2.waitKey(0)
cv2.destroyAllWindows()
