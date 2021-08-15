from pathlib import Path
from PIL import Image 
import os
import cv2

ROOT_DIR = r"dataset/renamed/Sus-scrofa/"

# cnt = 0
# for _ in os.listdir(ROOT_DIR):
#     image_file = Path(ROOT_DIR + str(cnt) + ".jpg")
#     with image_file.open('rb') as f:
#         im = Image.open(f, 'r')
#         im.verify()
#     cnt += 1

img = cv2.imread(ROOT_DIR+"126.jpg")
cv2.imshow("image",img)
cv2.waitKey(10000)