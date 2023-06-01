import json
from pycocotools.coco import COCO
from pycocotools import mask
import cv2
with open("/home/qazw5741/segment-anything/HR-LSP_seg.json","r") as f:
    test1=json.load(f)

for i in test1.keys():
    test_mask=mask.decode(test1[i]['Seg'])
    cv2.imwrite("test_maks.jpg",test_mask*255)
    print(i)
    break