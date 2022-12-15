import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = [
    "LeNet"
]


class LeNet(nn.Module):
    def __init__(self, rate=1.):
        super(LeNet, self).__init__()  # 调用基类的__init__()函数  
        '''''torch.nn.Conv2d(in_channels, out_channels, kernel_size,  
stride=1, padding=0, dilation=1, groups=1, bias=True)'''

        '''''torch.nn.MaxPool2d(kernel_size, stride=None,  
padding=0, dilation=1, return_indices=False, ceil_mode=False) 
    stride – the stride of the window. Default value is kernel_size'''
        self.conv = nn.Sequential(  # 顺序网络结构  
            nn.Conv2d(1, 6, 5, stride=1, padding=2),  # 卷积层 输入1通道，输出6通道，kernel_size=5*5  
            nn.ReLU(),  # 激活函数  
            nn.MaxPool2d(2, 2),  # 最大池化，kernel_size=2*2，stride=2*2  
            # 输出大小为14*14  
            nn.Conv2d(6, 16, 5, stride=1, padding=2),  # 卷积层 输入6通道，输出16通道，kernel_size=5*5  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 输出大小为7*7  
            nn.Conv2d(16, 120, 5, stride=1, padding=2),  # 卷积层 输入16通道，输出120通道，kernel_size=5*5  
            nn.ReLU(),
        )
        self.fc = nn.Sequential(  # 全连接层  
            nn.Linear(7 * 7 * 120, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            #nn.Linear(84, 10),
            #nn.Sigmoid(),
        )

    def forward(self, x):  # 前向传播  
        out = self.conv(x)
        out = out.view(out.size(0), -1)  # 展平数据为7*7=49的一维向量
        out = self.fc(out)
        return out
