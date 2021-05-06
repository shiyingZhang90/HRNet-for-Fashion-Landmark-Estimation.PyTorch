
import os
import shutil
import cv2

root = './data/data/logo'

for file in os.listdir(root):
    path = os.path.join(root, file)
    img = cv2.imread(path)
    h, w, _ = img.shape
    img = cv2.resize(img, (w // 4, h // 4))
    cv2.imwrite(path, img)
