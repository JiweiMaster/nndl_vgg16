from JiweiCommonUtil.imgshow import showLineImg
import numpy 
import torch 
import torch.nn as nn


# 使用VGG16的模型
class VGG16(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(VGG16,self).__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            # 64*64
            nn.Conv2d(in_channel,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # 128*128
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # 256*256
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # 512*512
            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # 512*512
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.MaxPool2d(2,2),
            # full linear layer     
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self,x):
        return self.net(x)

if __name__ == "__main__":
    vgg16 = VGG16(3,10)
    print(vgg16)









    

