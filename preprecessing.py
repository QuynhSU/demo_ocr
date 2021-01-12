import os
import cv2
import numpy as np
def pre_process(img):
    # print(img.shape)
    h, w, _ = img.shape
    ratio = 64.0/h
    new_w = int(w*ratio)
    
    if new_w < 256:
        img = cv2.resize(img, (new_w, 64), interpolation=cv2.INTER_CUBIC)
        pad_img = np.ones((64, 256-new_w, 3), dtype=np.uint8)*127
        img = np.concatenate((img, pad_img), axis=1)
    else:
        img = cv2.resize(img, (256, 64), interpolation=cv2.INTER_CUBIC)
    pad_img = np.ones((64, 256, 3), dtype=np.uint8)*127
    img = np.concatenate((img, pad_img), axis=0)
    # plt.imshow(new_im)
    # plt.show()
    # print(new_im.shape
    return img
