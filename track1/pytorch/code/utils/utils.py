# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:39:02 2022

@author: DELL
"""

import cv2
import numpy as np


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def randomHorizontalFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
        batch_hm = cv2.flip(batch_hm, 1)
        batch_wh = cv2.flip(batch_wh, 1)
        batch_reg = cv2.flip(batch_reg, 1)
        batch_reg_mask = cv2.flip(batch_reg_mask, 1)

    return image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask


def randomVerticleFlip(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
        batch_hm = cv2.flip(batch_hm, 0)
        batch_wh = cv2.flip(batch_wh, 0)
        batch_reg = cv2.flip(batch_reg, 0)
        batch_reg_mask = cv2.flip(batch_reg_mask, 0)

    return image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask


def randomRotate90(image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        label=np.rot90(label)
        batch_hm=np.rot90(batch_hm)
        batch_wh=np.rot90(batch_wh)
        batch_reg=np.rot90(batch_reg)
        batch_reg_mask=np.rot90(batch_reg_mask)

    return image, label, batch_hm, batch_wh, batch_reg, batch_reg_mask


def preprocess_input(image):
    mean    = [0.38983384, 0.41159928, 0.3616227]
    std     = [0.20442073, 0.17412226, 0.1663693]
    return (image / 255. - mean) / std

