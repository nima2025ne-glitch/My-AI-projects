#https://github.com/ultralytics/ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
from random import randint

model = YOLO('yolov8n.pt')
# result = model('WIN_20260403_17_09_04_Pro.jpg')
result = model('sakht.jpg')
# result = model('bus.jpg')
print("\n")
print(result[0].boxes.cls[0])
cls_tensor = result[0].boxes.cls
print( cls_tensor)
cls_list = cls_tensor.tolist()
class_names = [result[0].names[int(c)] for c in cls_list]
print(class_names)
print(result[0].boxes.conf)
result[0].boxes.xywh
# img = cv2.imread("bus.jpg")
# cv2.imshow("pic", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows
# cv2.destroyAllWindows
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
from random import randint

model = YOLO('yolov8n.pt')

img = cv2.imread("sakht.jpg")

result = model('sakht.jpg')

box = Annotator(img)
 

for i in range(len(result[0].boxes)):
    r = randint(1,255)
    g = randint(1,255)
    b = randint(1,255)
    if (float(result[0].boxes[i].conf)>0.1):
        box.box_label(result[0].boxes[i].xyxy[0], f"{result[0].names[int(result[0].boxes[i].cls[0].item())]} {float(result[0].boxes[i].conf):.2}", (b,g,r))
    



cv2.imshow("ax",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
