import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
class CNN_extractor(nn.Module):
    def __init__(self):
        super(CNN_extractor, self).__init__()
        self.conv1 = nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1) # 按照公式计算后经过卷积层不改变尺寸
        self.pool = nn.MaxPool2d(2,2)
        self.dp = nn.Dropout(p=0.5)
        self.classifier =  nn.Sequential(nn.Linear(2*56*56*4*4,256),
                                    nn.ReLU(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    nn.Linear(64, 2),
                                    )
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dp(x)
        
        return x

class CNN_trainer(nn.Module):
    def __init__(self):
        super(CNN_trainer, self).__init__()
        self.conv = CNN_extractor()
        self.classifier =  nn.Sequential(nn.Linear(2*56*56*4*4,256),
                                    nn.ReLU(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    nn.Linear(64, 2),
                                    )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 2*56* 56 *4*4)
        pred = self.classifier(x)
        return pred

class Agent(nn.Module):
    def __init__(self):
        super(Agent,self).__init__()     
        self.feature_extract =  nn.Sequential(nn.Linear(2*56*56*4*4,256),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    )
        self.conv1 = nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.feature_extract_bid =  nn.Sequential(nn.Linear(2*56*56*4*4,256),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    )
        self.classifier = nn.Linear(64, 2) 
        self.bidder = nn.Linear(64, 1)
        self.dp = nn.Dropout(p=0.5)
        self.sca_fc = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dp(x)
        # x = self.cnn(x)
        # print(x.shape)/3*2
        x = x.view(-1, 2*56* 56 *4*4)
        out = self.feature_extract(x)
        # out_bid = self.feature_extract_bid(x)
        pred = self.classifier(out)
        
        bid = self.bidder(out)
        bid = self.bn(bid)
        # bid = self.sca_sfc(bid)

        return pred, bid, out

class ResnetAgent(nn.Module):
    def __init__(self):
        super(ResnetAgent,self).__init__()     
        self.conv1 = models.resnet18()
        self.n_feature = self.conv1.fc.in_features
        self.conv1.fc = nn.Sequential(nn.Linear(self.n_feature,256),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    )
        self.pool = nn.MaxPool2d(2,2)
        self.classifier = nn.Linear(64, 2) 
        self.bidder = nn.Linear(64, 1)
        self.dp = nn.Dropout(p=0.5)
        self.sca_fc = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        
        out = self.conv1(x)
        # out_bid = self.feature_extract_bid(x)
        pred = self.classifier(out)
        
        bid = self.bidder(out)
        bid = self.bn(bid)
        # bid = self.sca_sfc(bid)

        return pred, bid, out
