import math
import random
import numpy as np
import cv2

def transform_preds(coords, center, scale, output_size):
    """transform prediction to new coords"""
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """get affine matrix"""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    """get new position after affine"""
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    """get the third point to calculate affine matrix"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """get new pos after rotate"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def post_process(dets, c, s, h, w, num_classes):
    """rescale detection to original scale"""
    # c, s, h, w = meta['c'], meta['s'], meta['out_height'], meta['out_width']
    # ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c, s, (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c, s, (w, h))
        # classes = dets[i, :, -1]
        # for j in range(num_classes):
        #     inds = (classes == j)
        #     top_preds[j + 1] = np.concatenate([
        #         dets[i, inds, :4].astype(np.float32),
        #         dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        # ret.append(top_preds)
    
    # print("++++++++++++")
    # print(dets)
    # print("++++++++++++")
    # for j in range(1, num_classes + 1):
    #     ret[0][j] = np.array(ret[0][j], dtype=np.float32).reshape(-1, 5)
    #     ret[0][j][:, :4] /= scale
    # return ret[0]
    return dets