import os
# import cv2
# import tqdm
import numpy as np
try:
    import gdal
except:
    from osgeo import gdal
    

def imread_gdal(image_path):
    dataset = gdal.Open(image_path)
    width = dataset.RasterXSize 
    height = dataset.RasterYSize 
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    print(nodata)
    image_h_s = []
    image_w_s = []
    step = 100
    # for height_i in tqdm.tqdm(range(0,height,step)):
    for height_i in range(0,height,step):
        try:
            image_i = dataset.ReadAsArray(0, height_i, width, min(step,height-height_i))
            image_i[image_i==nodata] = 0
            image_i = image_i.astype(np.uint8)
        except:
            image_i = np.zeros((3,step,width), np.uint8)
        image_h_s.append(image_i)
    image_h = np.concatenate(image_h_s,1)
    
    # for width_i in tqdm.tqdm(range(0,width,step)):
    for width_i in range(0,width,step):
        try:
            image_i = dataset.ReadAsArray(width_i, 0, min(step,width-width_i), height)
            image_i[image_i==nodata] = 0
            image_i = image_i.astype(np.uint8)
        except:
            image_i = np.zeros((3,height,step), np.uint8)
        image_w_s.append(image_i)
    image_w = np.concatenate(image_w_s,2)
    
    image = np.where(image_h>image_w, image_h, image_w)
    return image

def imwrite_gdal(path, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype,
                            options=["TILED=YES", "COMPRESS=LZW"])
    if(dataset!= None):
        dataset.SetGeoTransform((0,0,0,0,0,0)) #写入仿射变换参数
        dataset.SetProjection("") #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset
    
if __name__ == "__main__":
    
    error_tif_names = ["148.tif","156.tif","157.tif","235.tif","262.tif"]
    error_tif_dir = "././data/fusai_release/testA/images"
    correct_tif_dir = error_tif_dir.replace("images","images_correct")
    if not os.path.exists(correct_tif_dir): os.makedirs(correct_tif_dir)
    for error_tif_name in error_tif_names:
        image_path = error_tif_dir + "/" + error_tif_name
        img = imread_gdal(image_path)
        # img = np.transpose(img,(1,2,0))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        correct_tif_path = correct_tif_dir + "/" + error_tif_name
        imwrite_gdal(correct_tif_path, img)