import cv2
from ultralytics import YOLO
import numpy as np

class YOLOSegmentation:
    def __init__(self,model_path) :
        self.model=YOLO(model_path)
   
    def detect(self,img):
        height,width,channels=img.shape

        results=self.model.predict(source=img.copy(),save=True,save_txt=True)
        result=results[0]
        print(result)
        
        segmentation_contours_idx=[]
       # print(result)
        masks = result.masks
        # Iterate over detected objects
        for i, mask in enumerate(masks.data):
            # Convert mask to numpy array
            mask_np = mask.cpu().numpy()
    
        # Process the mask as needed
        # For example, to visualize the mask:
        mask_img = np.uint8(mask_np * 255)
        cv2.imshow(f"Mask {i}", mask_img)
        cv2.waitKey(0)    

        