# License_Plate_Detection_Pytorch
This is a two stage lightweight and robust license plate recognition in MTCNN and LPRNet using Pytorch. [MTCNN](https://arxiv.org/abs/1604.02878v1) is a very well-known real-time detection model primarily designed for human face recognition. It is modified for license plate detection. [LPRNet](https://arxiv.org/abs/1806.10447), another real-time end-to-end DNN, is utilized for the subsquent recognition. This network is attributed by its superior performance with low computational cost without preliminary character segmentation. The [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025) is embeded in this work to allow a better characteristics for recognition. The recognition accuracy is up to **99%** on CCPD base dataset with ~ **80 ms/image** on Nivida Quadro P4000. Here is the illustration of the proposed pipeline:

<img src="test/pipeline.png"  width="800">

## MTCNN
The modified MTCNN structure is presented as below. Only proposal net (Pnet) and output net (Onet) are used in this work since it is found that skipping Rnet will not hurt the accuracy in this case.  The Onet accepts 24(height) x 94(width) BGR image which is consistent with input for LPRNet. 

<img src="test/MTCNN.png"  width="600" style="float: left;">

## LPRNet Performance 
LPRNet coding is heavily followed by [sirius-ai](https://github.com/sirius-ai/LPRNet_Pytorch)'s repo. One exception is that the spatial transformer layer is inserted to increase the accuracy reported on CCPD database as below: 

|   | Base(45k) | DB | FN | Rotate | Tilt | Weather | Challenge |
|  :------:     | :---------: | :---------: |:---------: |:---------: |:---------: |:---------: |:---------: |
|   accuracy %      | 99.1     |  96.3 | 97.3 | 95.1 | 96.4 | 97.1 | 83.2 |

## 依赖
* pip install imutils
* pip install opencv-python==3.4.5.20 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
* pip install opencv-contrib-python==3.4.5.20 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
* pip install torchvision==0.2.2 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
* pip install torch==1.0.1 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com


## 依赖
* pip install imutils -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
* pip install opencv-python==3.4.5.20 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
* pip install opencv-contrib-python==3.4.5.20 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
* pip install torchvision==0.2.2 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
* pip install torch==1.0.1 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com


## 训练 MTCNN
* 下载[CCPD](https://github.com/detectRecog/CCPD)数据集放入至'ccpd'文件中，解压至当前文件夹，并将'CCPD2019'改名为'ccpd_dataset'
* 进入文件夹 'cd MTCNN/data_set/' 运行''python preprocess.py' 分割为训练结果集和验证结果集分布放入 "ccpd_train" 和 "ccpd_val"文件夹中
* 进入文件夹 'cd MTCNN/data_preprocessing/' 修改opt = 'train' 运行'python gen_Pnet_train_data.py'生成Pnet训练数据集
* 进入文件夹 'cd MTCNN/data_preprocessing/' 修改opt = 'val' 运行'python gen_Pnet_train_data.py'生成Pnet验证数据集
* 进入文件夹 'cd MTCNN/data_preprocessing/' 修改opt = 'train' 运行'python assemble_Pnet_imglist.py'合并Pnet训练数据集
* 进入文件夹 'cd MTCNN/train/' 运行'python Train_Pnet.py'训练Pnet模型

* 进入文件夹 'cd MTCNN/data_preprocessing/' 修改opt = 'train' 运行'python gen_Onet_train_data.py'生成Onet训练数据集
* 进入文件夹 'cd MTCNN/data_preprocessing/' 修改opt = 'val' 运行'python gen_Onet_train_data.py'生成Onet验证数据集
* 进入文件夹 'cd MTCNN/data_preprocessing/' 修改opt = 'train' 运行'python assemble_Onet_imglist.py'合并Onet训练数据集
* 进入文件夹 'cd MTCNN/train/' 运行'python Train_Onet.py'训练Onet模型

## Training on LPRNet
* run 'LPRNet/data/preprocess.py' to prepare the dataset
* run 'LPRNet/LPRNet_Train.py' for training 

## Test
* run 'MTCNN/MTCNN.py' for license plate detection
* run 'LPRNet/LPRNet_Test.py' for license plate recognition
* run 'main.py' for both

## Reference
* [MTCNN](https://arxiv.org/abs/1604.02878v1)
* [LPRNet](https://arxiv.org/abs/1806.10447)
* [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025)
* [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)

**Please give me a star if it is helpful for your research**
