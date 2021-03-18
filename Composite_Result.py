import os
from os.path import join
import numpy as np
import cv2
from PIL import Image
import datetime
from random import randrange, randint


save_path = "./data/hh/"
img = cv2.imread("./data/hh/aa1.png")
mask = cv2.imread("./data/hh/aa.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
_, mask_inv = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)
result = cv2.bitwise_and(img, img, mask=mask_inv)


img = cv2.GaussianBlur(img, (5, 5), 0)
img_mask_inv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_mask_inv = cv2.threshold(img_mask_inv, 250, 255, cv2.THRESH_BINARY_INV)
result = cv2.bitwise_and(result, result, mask=img_mask_inv)

basename = "result"
suffix = datetime.datetime.utcnow().strftime("%y%m%d_%H%M%S_%f")
date_filename = "_".join([basename, suffix])

img_filename = date_filename + ".png"
img_filename = join(save_path, img_filename)
cv2.imwrite(img_filename, result)


cv2.imshow('img_mask_inv', img_mask_inv)
# cv2.imshow('img', mask_inv)
cv2.waitKey(0)