import os
from os.path import join
import numpy as np
import cv2
from PIL import Image
import datetime
from random import randrange, randint

class Composite():
    def __init__(self, AnomalyImageDir="", TrueImageDir="", img_save_path="", label_save_path=""):
        self.AnomalyImageDir = AnomalyImageDir
        self.TrueImageDir = TrueImageDir
        self.Img_save_path = img_save_path
        self.Img_label_path = label_save_path
        self.Anormalydata = []  # List of image files for reader
        self.Anormalydata += [each for each in os.listdir(self.AnomalyImageDir) if
                              each.endswith('.PNG') or each.endswith('.JPEG') or each.endswith('.TIF') or each.endswith(
                                  '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith(
                                  '.tif') or each.endswith('.gif')]  # Get list of training images
        self.Truedata = []  # List of image files for reader
        self.Truedata += [each for each in os.listdir(self.TrueImageDir) if
                          each.endswith('.PNG') or each.endswith('.JPEG') or each.endswith('.TIF') or each.endswith(
                              '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith(
                              '.tif') or each.endswith('.gif')]  # Get list of training images
        self.Truedata.sort()

    def getitem(self, idx, bg_true=False):
        # 정상제품이 아래(bg) 있으면 True, 위(img)에 있으면 False
        # TrueName = self.Truedata[idx]
        tmp = randrange(len(self.Truedata)) #이물 이미지가 절대적으로 많으므로
        TrueName = self.Truedata[tmp]
        true_name = join(self.TrueImageDir, TrueName)
        AnomalyName = self.Anormalydata[idx]
        anomaly_name = join(self.AnomalyImageDir, AnomalyName)

        lower_contour_color = 0
        upper_contour_color = 120

        if bg_true: #배경이 정상
            bg_img = cv2.imread(true_name)
            bg_img = cv2.flip(bg_img, randint(0,1)) # 랜덤 상하좌우 반전 => 이물 이미지가 절대적으로 많으므로
            bg_img = cv2.flip(bg_img, randint(0, 1)) # 랜덤 상하좌우 반전 => 이물 이미지가 절대적으로 많으므로
            fg_img = cv2.imread(anomaly_name)
            fg_img = cv2.resize(fg_img, dsize=(bg_img.shape[1], bg_img.shape[0]), interpolation=cv2.INTER_AREA) #정상제품에 맞게 사이즈 조정
        else: #배경이 이물
            fg_img = cv2.imread(true_name)
            fg_img = cv2.flip(fg_img, randint(0, 1)) # 랜덤 상하좌우 반전 => 이물 이미지가 절대적으로 많으므로
            fg_img = cv2.flip(fg_img, randint(0, 1)) # 랜덤 상하좌우 반전 => 이물 이미지가 절대적으로 많으므로
            bg_img = cv2.imread(anomaly_name)
            bg_img = cv2.resize(bg_img, dsize=(fg_img.shape[1], fg_img.shape[0]), interpolation=cv2.INTER_AREA) #정상제품에 맞게 사이즈 조정

        bg_mask = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)
        # bg_mask = cv2.inRange(bg_mask, lower_contour_color, upper_contour_color) #범위 내 픽셀들(background)은 흰색 나머지는 검정색


        cv2.imshow('win', bg_mask)
        cv2.waitKey(0)
        fg_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)

        _, fg_binary = cv2.threshold(fg_gray, lower_contour_color, upper_contour_color, cv2.THRESH_BINARY)  #foreground만 검정 이미지
        _, bg_binary = cv2.threshold(bg_gray, lower_contour_color, upper_contour_color, cv2.THRESH_BINARY)  #background만 검정 이미지

        _, inv_fg_mask = cv2.threshold(fg_gray, lower_contour_color, upper_contour_color, cv2.THRESH_BINARY_INV) #foreground만 흰색 이미지
        _, inv_bg_mask = cv2.threshold(bg_gray, lower_contour_color, upper_contour_color, cv2.THRESH_BINARY_INV) #background만 흰색 이미지


        fg_black = cv2.bitwise_and(fg_img, fg_img, mask=inv_fg_mask) #foreground 이미지 나머지 검정색
        bg_white = cv2.bitwise_and(bg_img, bg_img, mask=fg_binary)  #background 이미지 나머지 검정색

        composite_img = cv2.add(fg_black, bg_white) #합성 이미지 결과

        #라벨 이미지 저장
        if bg_true: #배경이 정상
            composite_label = cv2.bitwise_and(inv_bg_mask, inv_bg_mask, mask=fg_binary)
        else: #배경이 이물
            composite_label = inv_fg_mask

        # ret, bw = cv2.threshold(composite_label, 127, 255, cv2.THRESH_BINARY)
        composite_label_np = np.array(composite_label)
        composite_label_np = composite_label_np / 255 # 레이블링하게 255 -> 1로
        composite_label_np = composite_label_np.astype(np.uint8)
        composite_label_np = Image.fromarray(composite_label_np)

        basename = "image"
        # suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        suffix = datetime.datetime.utcnow().strftime("%y%m%d_%H%M%S_%f")
        date_filename = "_".join([basename, suffix])

        img_filename = date_filename + ".png"
        img_filename = join(self.Img_save_path, img_filename)
        # composite_img_np = Image.fromarray(composite_img)
        # composite_img_np.save(img_filename) #이미지가 이상하게 나옴
        cv2.imwrite(img_filename, composite_img)

        labe_filename = date_filename + "_mask.png"
        labe_filename = join(self.Img_label_path, labe_filename)
        composite_label_np.save(labe_filename)


def main():
    Anomaly_path = "./data/AnomalyImg2/"
    # Anomaly_path= "../Img/TestImg/"
    True_path = "./data/TrueImg2/"
    img_save_path = "./data/img_result2/"
    label_save_path = "./data/label_result2/"

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    data = Composite(Anomaly_path, True_path, img_save_path, label_save_path)

    for idx in range(len(data.Anormalydata)):
        print(idx)
        data.getitem(idx, False) # True: 정상제품이 아래 / False: 정상제품이 위

main()