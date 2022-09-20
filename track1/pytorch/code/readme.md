# [线下决赛赛道1：道路提取和交叉口识别](http://rsipac.whu.edu.cn/subject_one)

## 数据预处理

### 数据可视化

pre/draw_box.py 用来可视化训练数据集

### json转txt

pre/json2txt.py 为每个image配置一个内容为[xmin,ymin,xmax,ymax,category_id]的txt。方便后面进行数据裁剪。

### 数据裁剪

pre/data_clip.py 图像尺寸太大，需要进行裁剪，使用滑动窗口裁剪为1024*1024大小。

### 生成训练txt

pre/gen_train_txt.py 生成训练txt,由[image_paths, label_paths, annotations]组成。

## 模型训练

train.py 使用hrnet作为backbone，fcn head作为道路分割头, centernet head作为交叉口检测头。


