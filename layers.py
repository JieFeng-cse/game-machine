import torch
from torch import nn
import torch.nn.functional as F
class Agent(nn.Module):
    def __init__(self):
        super(Agent,self).__init__()     
        self.feature_extract =  nn.Sequential(nn.Linear(8*56*56*2*2,256),
                                    nn.ReLU(),
                                #     nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    )
        self.conv1 = nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1) # 按照公式计算后经过卷积层不改变尺寸
        self.pool = nn.MaxPool2d(2,2)
        self.feature_extract_bid =  nn.Sequential(nn.Linear(8*56*56*2*2,256),
                                    nn.ReLU(),
                                #     nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    )
        self.classifier = nn.Linear(64, 2) 
        self.bidder = nn.Linear(64+2, 1)
        self.dp = nn.Dropout(p=0.5)
        self.sca_fc = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dp(x)
        x = x.view(-1, 8 * 56* 56 *2*2)
        out = self.feature_extract(x)
        out_bid = self.feature_extract_bid(x)
        pred = self.classifier(out)
        
        bid = self.bidder(torch.cat([out_bid,pred], dim=1))
        bid = self.bn(bid)
        bid = self.sca_fc(bid)

        return pred, bid, out

        
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1) # 按照公式计算后经过卷积层不改变尺寸
        self.pool = nn.MaxPool2d(2,2) # 2*2的池化 池化后size 减半
        # self.conv2 = nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(3*56*56*4*4,256)#两个池化，所以是224/2/2=56
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,2) #suppose the first one is dog, second is cat
        self.agent1 = nn.Sequential(nn.Linear(3*56*56*4*4,256),
                                    nn.ReLU(),
                                #     nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    )
        self.classifier1 = nn.Linear(64, 2)
        self.bidder1 = nn.Linear(64, 1)
        self.agent2 = nn.Sequential(nn.Linear(3*56*56*4*4,256),
                                    nn.ReLU(),
                                #     nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    )
        self.classifier2 = nn.Linear(64, 2)
        self.bidder2 = nn.Linear(64, 1)
        self.agent3 = nn.Sequential(nn.Linear(3*56*56*4*4,256),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    nn.Dropout())
        self.classifier3 = nn.Linear(64, 2)
        self.bidder3 = nn.Linear(64, 1)
        self.dp = nn.Dropout(p=0.5)
        self.sca_fc = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1)
    def forward(self,x):

#         print("input:", x)
        # x = self.pool(F.relu(self.conv1(x)))
#         print("first conv:", x)
        # x = self.pool(F.relu(self.conv2(x)))
#         print("second conv:", x)
        # print(x.shape)
        # x = self.dp(x)
        x = x.view(-1, 3 * 56* 56 *4*4)#将数据平整为一维的 
        out1 = self.agent1(x)
        out2 = self.agent2(x)
        # out3 = self.agent3(x)

        pred1 = self.classifier1(out1)
        pred2 = self.classifier2(out2)
        # pred3 = self.classifier3(out3)

        bid1 = self.bidder1(out1)
        # bid1 = torch.zeros(pred1.shape[0],1)
        bid1 = self.bn(bid1)
        bid1 = self.sca_fc(bid1)

        bid2 = self.bidder2(out2)
        bid2 = self.bn(bid2)
        bid2 = self.sca_fc(bid2)
        # bid3 = self.bidder3(out3)
        # bid3 = self.sca_fc(bid3)

        # print(bid1[0], bid2[0], bid3[0])
        bids = torch.stack([bid1, bid2], dim=1)
        probs = F.softmax(bids, dim=1)
        # probs = 1
        # pred1 = F.softmax(pred1, dim=1)
        # pred2 = F.softmax(pred2, dim=1)
        # pred3 = F.softmax(pred3, dim=1)

        # pred = pred1 * probs[:, 0] + pred2 * probs[:, 1] + pred3 * probs[:, 2]
        # # print(pred1)
        final_bid = bid1 * probs[:, 0] + bid2 * probs[:, 1]
        # pred1 = F.softmax(pred1, dim=1)
        # pred2 = F.softmax(pred2, dim=1)
        # return pred1, bid1, pred2, bid2
        # print(pred.shape)
        # print(final_bid[0])
        # print(probs[0])

        # x = torch.mean(torch.stack([out1, out2, out3]), dim=0)
        # x = F.relu(self.fc1(x))
        # x = self.dp(x)
        # x = F.relu(self.fc2(x))  
        # x = self.fc3(x)  
        # x = F.softmax(x, dim=1)
        # x = torch.argmax(x, dim=1)
        return pred1, pred2, probs, final_bid, bid1, bid2
        # return pred1, probs, final_bid