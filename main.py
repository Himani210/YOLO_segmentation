import cv2
from yolo_segmentation import YOLOSegmentation

img=cv2.imread("Cat.jpeg")

img=cv2.resize(img,(800,600))
cv2.imshow("Image",img)

#Segmentation detector
ys=YOLOSegmentation("yolov8m-seg.pt")
bboxes,classes,segmentation,scores=ys.detect(img)

# for bbox, class_id,seg,score in zip(bboxes,classes,segmentation,scores):
#     print("bbox :",bbox,"class id :","class_id:",class_id,"seg :",seg,"score :",score)

#     (x,y,x2,y2)=bbox
#     cv2.rectangle(img,(x,y),(x2,y2),(0,0,255),2)

#     cv2.polylines(img,[seg],True,(255,0,0),2) #thickness








cv2.waitKey(0)