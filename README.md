# FashionAI2018 服装属性标签识别

## 环境  
caffe+keras  
python 2.7  
numpy 1.14.2  
opencv 3.4.0  
cuda9.0  
cudnn 7.0  

## **文件说明：**  
### **caffe部分**：  
使用InceptionV4举例，该网络在复赛中可以实现94.11的准确率。  
1. **训练网络使用training.sh**  

修改相应的路径，即Log、TOOLS、-weights后面的路径，其中，Inception-V4的pretrain-model请到[caffe-model](https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation)下载 。  
使用方法bash training.sh [solver][Class][GPU]。  
solver ： solver文件路径，本代码中存放于solver文件夹  
class： 类别名，用来命名log日志文件，即neck,collar...  
GPU ： GPU的型号，如0,1,2  
2. **caffe源码处理**

[caffe](https://github.com/BVLC/caffe)，并用文件中的src和include文件夹替换caffe中的src和include。src和include主要修改图片预处理部分，并开始支持多label输入。  
3. **训练网络的solver文件夹**  

修改net:后面的路径，替换为相应的根目录。修改snapshot_prefix路径。  
4. **train文件夹**  

即相应的网络结构，修改训练文件的路径，在data/train/中  
5. **deployV4文件夹**  

测试用的网络结构。  
6. **test.py**  

12-crop测试使用的程序。修改里面对应的路径即可。修改好路径可以直接python test.py运行  
7. **tools文件夹**  

初赛和复赛使用的一些小程序，里面包含复赛使用的双模型融合程序，以及初赛使用的多模型融合程序，以及相应的多属性标签的制作程序。  

**注**：由于有8个属性，如果使用多任务训练，则拥有8个训练标签，不属于当前图片的label置-1，属于当前label的从0开始计数，具体请参考data/train/中的txt  

### **Keras部分：**  
请参考培文大佬源码：  
[Keras代码](https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.57e0311flf0xnz&raceId=231649&postsId=4799)  
直接可用，效果低于InceptionV4，不过也不错。  

### **检测部分：**  
切割的大图部分请参考MaskRCNN代码  
小图请参考CPN代码  






