import os
import cv2
import ast
import json
import tqdm
import time
import glob
import warnings
import argparse
import numpy as np
from skimage import io

import mindspore
from mindspore import nn
from mindspore import Model
from mindspore import Tensor
from mindspore import context
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from pre.gdal_imread import imread_gdal

from src.hrnet import get_seg_model
from src.decode import DetectionDecode
from src.post_process import post_process
from src.config import organize_configuration
# from src.config import config_hrnetv2_w48 as config
from src.config import config_hrnetv2_w18 as config
from src.utils_bbox_20221026 import decode_bbox, postprocess

# 忽略警告信息
warnings.filterwarnings('ignore')

def parse_args():
    """
    Get arguments from command-line.
    """
    parser = argparse.ArgumentParser(description='OCRNet Semantic Segmentation Inference.')
    parser.add_argument("--data_url", type=str, default=None,
                        help="Storage path of dataset.")
    parser.add_argument("--train_url", type=str, default=None,
                        help="Storage path of evaluation results in OBS. It's useless here.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Storage path of dataset in OBS.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Storage path of evaluation results on machine. It's useless here.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run online or offline.")
    parser.add_argument("--checkpoint_url", type=str,
                        help="Storage path of checkpoint file in OBS.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Storage path of checkpoint file on machine.")
    parser.add_argument("--checkpoint_path2", type=str, default=None,
                        help="Storage path of checkpoint file on machine.")
    return parser.parse_args()

class Hrnet_infer_noTTA(nn.Cell):
    """
    Encapsulation class of centernet testing.

    Args:
        net_config: The config info of CenterNet network.
        K(number): Max number of output objects. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: False.

    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, network, K=100, enable_nms_fp16=False):
        super(Hrnet_infer_noTTA, self).__init__()
        self.network = network
        self.act = ops.Sigmoid()
        self.decode = DetectionDecode(K, enable_nms_fp16)

    def construct(self, image):
        """Calculate prediction scores"""
        seg, hm, wh, offset = self.network(image)
        seg_outputs = self.act(seg)
        hm =self.act(hm)
        det_outputs = self.decode(hm, wh, offset)
        return seg_outputs, det_outputs
    
class Hrnet_infer_TTA_one_model(nn.Cell):
    """
    Encapsulation class of centernet testing.

    Args:
        net_config: The config info of CenterNet network.
        K(number): Max number of output objects. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: False.

    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, network, K=100, enable_nms_fp16=False):
        super(Hrnet_infer, self).__init__()
        self.network = network
        self.act = ops.Sigmoid()
        self.flip2 = ops.ReverseV2(axis=[2])
        self.flip3 = ops.ReverseV2(axis=[3])
        self.decode = DetectionDecode(K, enable_nms_fp16)

    def construct(self, image):
        """Calculate prediction scores"""
        
        image_flip2 = self.flip2(image)
        image_flip3 = self.flip3(image)
        
        seg1, hm1, wh1, offset1 = self.network(image)
        seg2, hm2, wh2, offset2 = self.network(image_flip2)
        seg3, hm3, wh3, offset3 = self.network(image_flip3)
        
        seg_outputs = (self.act(seg1) + 
                       self.flip2(self.act(seg2)) + 
                       self.flip3(self.act(seg3))) / 3
        
        hm = (self.act(hm1) + 
              self.flip2(self.act(hm2)) + 
              self.flip3(self.act(hm3))) / 3
        
        wh = (wh1 + 
              self.flip2(wh2) +
              self.flip3(wh3)) / 3
        
        offset = (offset1 + 
                  self.flip2(offset2) +
                  self.flip3(offset3)) / 3
        
        det_outputs = self.decode(hm, wh, offset)
        return seg_outputs, det_outputs

class Hrnet_infer(nn.Cell):
    """
    Encapsulation class of centernet testing.

    Args:
        net_config: The config info of CenterNet network.
        K(number): Max number of output objects. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: False.

    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, network, network2, K=100, enable_nms_fp16=False):
        super(Hrnet_infer, self).__init__()
        self.network = network
        self.network2 = network2
        self.act = ops.Sigmoid()
        self.flip2 = ops.ReverseV2(axis=[2])
        self.flip3 = ops.ReverseV2(axis=[3])
        self.decode = DetectionDecode(K, enable_nms_fp16)

    def construct(self, image):
        """Calculate prediction scores"""
        
        image_flip2 = self.flip2(image)
        image_flip3 = self.flip3(image)
        
        seg1, hm1, wh1, offset1 = self.network(image)
        seg2, hm2, wh2, offset2 = self.network(image_flip2)
        seg3, hm3, wh3, offset3 = self.network(image_flip3)
        seg4, hm4, wh4, offset4 = self.network2(image)
        seg5, hm5, wh5, offset5 = self.network2(image_flip2)
        seg6, hm6, wh6, offset6 = self.network2(image_flip3)
        
        seg_outputs = (self.act(seg1) + 
                       self.flip2(self.act(seg2)) + 
                       self.flip3(self.act(seg3)) +
                       self.act(seg4) + 
                       self.flip2(self.act(seg5)) + 
                       self.flip3(self.act(seg6))
                      ) / 6
        
        hm = (self.act(hm1) + 
              self.flip2(self.act(hm2)) + 
              self.flip3(self.act(hm3)) + 
              self.act(hm4) + 
              self.flip2(self.act(hm5)) + 
              self.flip3(self.act(hm6))
             ) / 6
        
        wh = (wh1 + 
              self.flip2(wh2) +
              self.flip3(wh3) +
              wh4 + 
              self.flip2(wh5) +
              self.flip3(wh6)
             ) / 6
        
        offset = (offset1 + 
                  self.flip2(offset2) +
                  self.flip3(offset3) +
                  offset4 + 
                  self.flip2(offset5) +
                  self.flip3(offset6)
                 ) / 6
        
        det_outputs = self.decode(hm, wh, offset)
        return seg_outputs, det_outputs
    
def preprocess_input(image):
    mean    = [0.38983384, 0.41159928, 0.3616227]
    std     = [0.20442073, 0.17412226, 0.1663693]
    return (image / 255. - mean) / std


def infer(model, image):
    
    image_shape = image.shape[0:2]
    height, width = image_shape
    # print(image.shape)
    image = preprocess_input(image)
    image = image.transpose((2, 0, 1))      # CHW
    image = np.expand_dims(image,axis=0)
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print("图像预处理完毕")
    # print(image.shape)
    
    seg_outputs, det_outputs = model.predict(Tensor(image, mstype.float32))
    # # add new
    # seg_outputs, det_outputs = model.predict(Tensor(image, mstype.float16))
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print("模型处理完毕")
    seg_outputs = seg_outputs.asnumpy().squeeze()
    
    seg_threshold = 0.30
    seg_outputs[seg_outputs>seg_threshold] = 1
    seg_outputs[seg_outputs<=seg_threshold] = 0
    seg_outputs = seg_outputs.astype(np.uint8)
    
    # print("model_outputs:", det_outputs.asnumpy())
    
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(width, height) * 1.0
    down_ratio = 4
    h = height // down_ratio
    w = width // down_ratio
    det_results = post_process(det_outputs.asnumpy(), c, s, h, w, 2)
    det_results = det_results[0]
    
    # print("det_results:")
    # print(np.array(det_results).shape)
    # start_nms_time = time.time()
    # NMS
    unique_labels = np.unique(det_results[:, -1])
    nms_detections = []
    for c in unique_labels:
        detections_class = det_results[det_results[:, -1] == c]
        # 按照存在物体的置信度排序
        conf_sort_index = np.argsort(detections_class[:, 4])[::-1]
        detections_class = detections_class[conf_sort_index]
        detections_class_inputs = Tensor(detections_class[:, :5], mindspore.float32)
        output_boxes, indices, mask = ops.NMSWithMask(0.3)(detections_class_inputs)
        indices_np = indices.asnumpy()
        bbox_index = indices_np[mask.asnumpy()]
        detections_class = detections_class[bbox_index]
        nms_detections.append(detections_class)
    nms_detections = np.concatenate(nms_detections)
    
    
    # print("nms_detections:")
    # print(np.array(nms_detections).shape)
    # print(time.time()-start_nms_time)
#     # 如果没有检测到物体，则返回原图
#     if det_results[0] is None:

#         return seg_outputs, []

#     top_label   = np.array(det_results[0][:, 5], dtype = 'int32')
#     top_conf    = det_results[0][:, 4]
#     top_boxes   = det_results[0][:, :4]
    

    top_label   = np.array(nms_detections[:, 5], dtype = 'int32')
    top_conf    = nms_detections[:, 4]
    top_boxes   = nms_detections[:, :4]
    
    
    det_outputs = []
    #---------------------------------------------------------#
    #   图像绘制
    #---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        label           = int(c)
        box             = top_boxes[i]
        score           = top_conf[i]

        top, left, bottom, right = box
        
        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image_shape[0], np.floor(bottom).astype('int32'))
        right   = min(image_shape[1], np.floor(right).astype('int32'))
        
        det_outputs.append([top,left,bottom-top,right-left,label+1, round(score,4)])
        # det_outputs.append([left,top,right-left,bottom-top,label+1, round(score,4)])
        # cv2.rectangle(image_copy, (top,left),(bottom,right),(255,255,255),3)
        # print(top, left, bottom, right, label, score)
        
    return seg_outputs,det_outputs

    
if __name__ == '__main__':
    args = parse_args()
    organize_configuration(cfg=config, args=args)

    context.set_context(mode=context.GRAPH_MODE, 
                        device_target='Ascend', 
                        save_graphs=False,
                        device_id=int(0))
    
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    
    net = get_seg_model(config)
    param_dict = load_checkpoint(ckpt_file_name=config.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    
    # net2 = get_seg_model(config)
    # param_dict2 = load_checkpoint(ckpt_file_name=config.checkpoint_path2)
    # load_param_into_net(net2, param_dict2)
    # net2.set_train(False)
    
    net_infer = Hrnet_infer_noTTA(net, K=1000)
    
    # # add new
    # net_infer = net_infer.to_float(mstype.float16)
    
    # net_infer = Hrnet_infer(net, net2, K=1000)
    net_infer.set_train(False)
    model = Model(net_infer)
    
    
    test_json_path = "data/instances_test.json"
    test_image_folder = "data/images"
    seg_save_folder = "results/masks"
    if not os.path.exists(seg_save_folder):
        os.makedirs(seg_save_folder)
    with open(test_json_path, encoding="utf-8") as f:
        annotations_test = json.load(f)
    images_info = annotations_test["images"]
    
    
    # juesai_results = []
    
    
    # # 测试用
    # images_info = images_info[64:66]
    # print(images_info)
    
    # image_paths = glob.glob(test_image_folder+"/*.tif")
    images_num = len(images_info)
    # images_num = len(image_paths)
    annotation_id = 0
    
    predict_size = 5920
    
    # resize_scale = 1.0
    for i in tqdm.tqdm(range(images_num)):
        # image_path = image_paths[i]
        # image_name = os.path.basename(image_path)
        # image_id = image_name.split(".")[0]
        
        image_name = images_info[i]["file_name"]
        image_id = images_info[i]["id"]
        image_path = os.path.join(test_image_folder,image_name)
        
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            image = io.imread(image_path)
            image = image[:,:,0:3]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 掉分
            image = np.uint8(image)
        except:
            image = imread_gdal(image_path)
            image = image[:3].transpose((1,2,0))
            image = np.uint8(image)
        
        image_shape = image.shape[0:2]
        
        
        # 从上往下全零值的行数
        zero_row_start = 0
        # 从下往上全零值的行数
        zero_row_end = 0
        # 从左往右全零值的列数
        zero_col_start = 0
        # 从右往左全零值的列数
        zero_col_end = 0
        # 删除不正常图像的那些nodata(0值)
        if (image_shape[0]>6400 or image_shape[1]>6400):
            for i in range(image.shape[0]):
                row = i
                if np.all(image[row] == 0):
                    zero_row_start = i+1
                else:
                    break
            for i in range(image.shape[0]):
                row = image.shape[0]-1-i
                if np.all(image[row] == 0):
                    zero_row_end = i+1
                else:
                    break
            for i in range(image.shape[1]):
                col = i
                if np.all(image[:,col] == 0):
                    zero_col_start = i+1
                else:
                    break
            for i in range(image.shape[1]):
                col = image.shape[1]-1-i
                if np.all(image[:,col] == 0):
                    zero_col_end = i+1
                else:
                    break
            image = image[zero_row_start:-zero_row_end if zero_row_end!=0 else None,
                          zero_col_start:-zero_col_end if zero_col_end!=0 else None,]
            # print(zero_row_start, zero_row_end, zero_col_start, zero_col_end, image_name)
        
        
        
        # 不够 predict_size*predict_size 的进行补零
        image_shape = image.shape[0:2]
        height, width = image_shape
        height_pad = max(0, predict_size - height)
        width_pad = max(0, predict_size - width)
        image = cv2.copyMakeBorder(image, 0, height_pad, 0, width_pad, 
                                   cv2.BORDER_CONSTANT, value = (0,0,0))
        image_pad_shape = image.shape[0:2]
        
        
        image = cv2.resize(image,(predict_size, predict_size))
        
        resize_scale_h = predict_size / image_pad_shape[0]
        resize_scale_w = predict_size / image_pad_shape[1]
        
        
        
        resize_scale_s = [resize_scale_w, resize_scale_h, resize_scale_w, resize_scale_h]
        
        seg_outputs, det_outputs = infer(model, image)
        
        seg_outputs = cv2.resize(seg_outputs, (image_pad_shape[1],image_pad_shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_outputs = seg_outputs[:seg_outputs.shape[0]-height_pad,
                                  :seg_outputs.shape[1]-width_pad]
        
        
        # 把不正常的nodata区域的结果用0值补回去
        seg_outputs = cv2.copyMakeBorder(seg_outputs, zero_row_start, zero_row_end, zero_col_start, zero_col_end, 
                                         cv2.BORDER_CONSTANT, value = (0))
        
        
        seg_save_path = os.path.join(seg_save_folder,image_name.replace("tif","png"))
        cv2.imwrite(seg_save_path,seg_outputs)
        # print(det_outputs)
        
        offsets = [zero_col_start, zero_row_start, 0, 0]
        
        for det_output in det_outputs:
            # if float(det_output[5])>0.01:
                # print(det_output)
            # annotations_test["annotations"].append({
            annotations_test["annotations"].append({
                "area": int(det_output[2]/resize_scale_w*det_output[3]/resize_scale_h),
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [int(_/resize_scale + offset) for _,resize_scale,offset in zip(det_output[:4],resize_scale_s, offsets)],
                "category_id": int(det_output[4]),
                "score": float(det_output[5]),
                "id": annotation_id,
                "ignore": 0,
                "segmentation": []
                })
            annotation_id += 1
        # break
    with open("results/test.bbox.json", "w") as f:
        json.dump(annotations_test["annotations"], f)
        
    os.system("zip -r results.zip results")