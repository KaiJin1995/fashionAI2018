#import multiprocessing
from tqdm import tqdm
crop_size=393
def over_sample(img):  # 12 crops of image
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    sample_list = [img[:crop_size, :crop_size], img[-crop_size:, -crop_size:], img[:crop_size, -crop_size:],
                   img[-crop_size:, :crop_size], img[yy: yy + crop_size, xx: xx + crop_size],
                   cv2.resize(img, (crop_size, crop_size))]
    return sample_list


def mirror_crop(img):  # 12*len(size_list) crops
    crop_list = []
    #img_resize = cv2.resize(img, (base_size, base_size))
    mirror = img[:, ::-1]
    crop_list.extend(over_sample(img))
    crop_list.extend(over_sample(mirror))
    return crop_list

def image_preprocess(img):
    b, g, r = cv2.split(img)
    return cv2.merge([(b-mean_value[0])/std[0], (g-mean_value[1])/std[1], (r-mean_value[2])/std[2]])
import sys
caffe_root = '/home/freedom/caffe/python'
sys.path.insert(1, caffe_root)
import caffe
import cv2
import os
import numpy as np
import csv
model_def = '/home/freedom/caffe/tianchifusai/InceptionV3/20180526/deploy/InceptionV3_deploy_neckline.prototxt'
caffe.set_device(1)
caffe.set_mode_gpu()
net_weights = '/home/freedom/caffe/tianchifusai/InceptionV3/20180526/weights/InceptionV3_RMS_design_iter_90000.caffemodel'
net = caffe.Net(model_def, net_weights, caffe.TEST)

input_shape = net.blobs['data'].data.shape


transformer = caffe.io.Transformer({'data': input_shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
#transformer.set_row_scale('data', 0.00390625)

def convert_array(Arrays):
    Arrays_string = ';'.join((str(i)) for i in Arrays)
    return Arrays_string




testFile = '/home/freedom/Extraspace/week-rank/Tests/neckline_design_labels.txt'
csv_w = open('/home/freedom/caffe/tianchifusai/InceptionV3/20180526/result3/neckline.csv', 'w')
writer = csv.writer(csv_w)
#model_def = '/home/freedom/caffe/examples/Inception-v3/deploy/InceptionV3_deploy_coatLength.prototxt'
#net_weights = ''
#net = caffe.Net(model_def, net_weights, caffe.TEST)

f = open(testFile, 'r')
fileitems = f.readlines()
for item in tqdm(fileitems):
    os_frame = []
    pic = item.split()[0]
    picPath = os.path.join('/home/freedom/Extraspace/week-rank',pic)
    picKind = item.split()[1]
    frame = cv2.imread(picPath)
    frame = cv2.resize(frame,(420,420))
    os_frame.extend(mirror_crop(frame))
    #os_frame = oversample([frame,], (input_shape[2], input_shape[3]))
    data = np.array([transformer.preprocess('data', x) for x in os_frame])
    net.blobs['data'].reshape(*data.shape)
    net.reshape()
    out = net.forward(blobs=['prob'], data = data)
    score = np.mean(out['prob'],axis = 0)
    writer.writerow([picPath,picKind,convert_array(score)])
f.close()
csv_w.close()


        
    
         
