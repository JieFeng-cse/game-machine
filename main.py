import os 
import pandas as pd 
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms,models,utils
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from layers import MyCNN, Agent
import time
import numpy as np
import torch.nn.functional as F
import math

SEED = 17
train_path = '../datasets/train/'
torch.autograd.set_detect_anomaly(True)

class MyDataset(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)
    def __getitem__(self, idx: int):
        # img to tensor and label to tensor
        img_path = self.path_list[idx]
        if self.train_flag is True:
            if img_path.split('.')[0] == 'cat' : 
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0]) # split 的是str类型要转换为int
        label = torch.as_tensor(label, dtype=torch.int64) # 必须使用long 类型数据，否则后面训练会报错 expect long
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
    def __len__(self) -> int:
        return len(self.path_list)

# def train(epoch, train_loader, device, 
#             model, criterion, optimizer):
#     model.train()
#     loss = 0.0
#     correct_pred = 0
#     total_num = 0
#     total_loss = 0.0
#     i = 0
#     for images, labels in train_loader:
#         i += 1
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         # outputs = model(images)
#         pred, _, bid = model(images)   
#         pred = F.softmax(pred, dim=1)
#         if i == 1:
#             print(pred[0])

#         log_p = pred
#         one_hot = torch.zeros(pred.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
#         loss = -one_hot*log_p
#         loss = loss.sum()
#         loss/=pred.shape[0]
#         loss.backward()
#         total_loss += loss.item()
#         optimizer.step()
#         # accuracy
#         # pred = pred1
#         # bid1 = bid1.repeat(1,2)
#         # bid2 = bid2.repeat(1,2)
#         # pred[bid2>bid1]=pred2[bid2>bid1]
        
#         pred = torch.argmax(pred, dim=1)
#         # print(pred,labels)
#         correct = torch.sum(pred.eq(labels))
#         # print(correct, 'up')
#         correct_pred += correct
#         total_num += labels.shape[0]
#     accuracy = correct_pred.cpu()/float(total_num)
#     return total_loss/i, accuracy
def train_seperate(train_loader, device, agent1, agent2, optimizer1, optimizer2):
    agent1.train()
    agent2.train()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in train_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        pred1, bid1, out1 = agent1(images)
        pred2, bid2, out2 = agent2(images)
        # print(pred1)
        pred1 = F.softmax(pred1, dim=1)
        bids1 = torch.stack([bid1, bid2.detach()], dim=1)
        probs1 = F.softmax(bids1, dim=1)
        c1 = bid1 * probs1[:, 0]                                                                                                                                                       + bid2.detach() * probs1[:, 1]   
        log_p1 = (torch.log(pred1+math.exp(-20)))
        
        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)

        # crite2 = nn.MSELoss()
        tmp = out2.detach().clone()
        # loss1_2 = -crite2(out1, tmp)
        loss1 = -torch.sum(one_hot*log_p1,dim=1).unsqueeze(1)*probs1[:,1] + c1 #TODO how does the gradient go?
        # it seems like if we use bid1 in the () we will get bad results
        loss1 = loss1.sum()
        loss1/=pred1.shape[0]
        # loss1 += loss1_2
        with torch.autograd.detect_anomaly():
            loss1.backward()
        # if i == 1:
        #     for name, parms in agent1.named_parameters():
        #         print('-->name:', name, ' --> grad_value:', parms.grad)
        nn.utils.clip_grad_norm_(agent1.parameters(), max_norm=1)
        optimizer1.step()

        pred2 = F.softmax(pred2, dim=1)
        bids2 = torch.stack([bid1.detach(), bid2], dim=1)
        probs2 = F.softmax(bids2, dim=1)
        c2 = bid1.detach() * probs2[:, 0] + bid2 * probs2[:, 1] 
        log_p2 = (torch.log(pred2+math.exp(-20)))
        one_hot = torch.zeros(pred2.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        loss2 = -torch.sum(one_hot*log_p2,dim=1).unsqueeze(1)*probs2[:,0] + c2
        # print(log_p1[0].detach().cpu())
        loss2 = loss2.sum()
        loss2/=pred2.shape[0]
        with torch.autograd.detect_anomaly():
            loss2.backward()
        optimizer2.step()

        total_loss += loss1.item()+loss2.item()
        probs = probs1.detach()
        pred = pred1*probs[:,0] + pred2*probs[:,1]
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    return total_loss/i, accuracy

def eval_seperate(eval_loader, device, agent1, agent2, optimizer1, optimizer2):
    agent1.eval()
    agent2.eval()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in eval_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        pred1, bid1, _ = agent1(images)
        pred2, bid2, _ = agent2(images)
        if i == 1:
            print(bid1[:2].detach(), bid2[:2].detach())

        pred1 = F.softmax(pred1, dim=1)
        bids1 = torch.stack([bid1, bid2.detach()], dim=1)
        probs1 = F.softmax(bids1, dim=1)
        c1 = bid1 * probs1[:, 0] + bid2.detach() * probs1[:, 1]   
        log_p1 = (torch.log(pred1+math.exp(-20)))*probs1[:,0]
        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        loss1 = -one_hot*log_p1 + bid2.detach()*probs1[:,0]
        loss1 = loss1.sum()
        loss1/=pred1.shape[0]

        pred2 = F.softmax(pred2, dim=1)
        bids2 = torch.stack([bid1.detach(), bid2], dim=1)
        probs2 = F.softmax(bids2, dim=1)
        c2 = bid1.detach() * probs2[:, 0] + bid2 * probs2[:, 1] 
        log_p2 = (torch.log(pred2+math.exp(-20)))*probs2[:,1]
        one_hot = torch.zeros(pred2.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        loss2 = -one_hot*log_p2 + bid2.detach()*probs2[:,1]
        loss2 = loss2.sum()
        loss2/=pred2.shape[0]

        total_loss += loss1.item()+loss2.item()
        probs = probs1.detach()
        pred = pred1*probs[:,0] + pred2*probs[:,1]
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    return total_loss/i, accuracy

        

def train(epoch, train_loader, device, 
            model, criterion, optimizer):
    model.train()
    loss = 0.0
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in train_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # outputs = model(images)
        pred1, pred2, probs, final_bid, bid1, bid2 = model(images)   
        pred1 = F.softmax(pred1, dim=1)
        pred2 = F.softmax(pred2, dim=1)
            
        log_p1 = (torch.log(pred1))*probs[:,0]
        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        loss1 = -one_hot*log_p1 + bid1*probs[:,0]
        loss1 = loss1.sum()
        loss1/=pred1.shape[0]
        loss1.backward(retain_graph=True)

        log_p2 = (torch.log(pred2))*probs[:,1]
        one_hot = torch.zeros(pred2.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        loss2 = -one_hot*log_p2 + bid2*probs[:,1]
        loss2 = loss2.sum()
        loss2/=pred2.shape[0]
        # loss = loss1 + loss2 #- torch.abs(probs[:,0]-probs[:,1]).mean()
        loss2.backward()
        total_loss += loss1.item()+loss2.item()
        optimizer.step()
        # accuracy
        pred = pred1*probs[:,0] + pred2*probs[:,1]
        # bid1 = bid1.repeat(1,2)
        # bid2 = bid2.repeat(1,2)
        # pred[bid2>bid1]=pred2[bid2>bid1]
        
        pred = torch.argmax(pred, dim=1)
        # print(pred,labels)
        correct = torch.sum(pred.eq(labels))
        # print(correct, 'up')
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    return total_loss/i, accuracy

def eval(epoch, eval_loader, device, 
            model, criterion, optimizer):
    model.eval()
    loss = 0.0
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in eval_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # outputs = model(images)
        pred1, pred2, probs, final_bid, bid1, bid2 = model(images)   
        if i == 1:
            print(bid1[:2],bid2[:2])
        pred1 = F.softmax(pred1, dim=1)
        pred2 = F.softmax(pred2, dim=1)
            
        log_p1 = (torch.log(pred1) - final_bid)*probs[:,0]
        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        loss1 = -one_hot*log_p1
        loss1 = loss1.sum()
        loss1/=pred1.shape[0]

        log_p2 = (torch.log(pred2) - final_bid)*probs[:,1]
        one_hot = torch.zeros(pred2.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)# KL divergence?
        loss2 = -one_hot*log_p2
        loss2 = loss2.sum()
        loss2/=pred2.shape[0]
        loss = loss1 + loss2# - torch.abs(probs[:,0]-probs[:,1]).mean()
        total_loss += loss.item()
        # accuracy
        pred = pred1
        bid1 = bid1.repeat(1,2)
        bid2 = bid2.repeat(1,2)
        pred[bid2>bid1]=pred2[bid2>bid1]
        
        pred = torch.argmax(pred, dim=1)
        # print(pred,labels)
        correct = torch.sum(pred.eq(labels))
        # print(correct, 'up')
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    return total_loss/i, accuracy

# def eval(epoch, eval_loader, device, 
#             model, criterion, optimizer):
#     model.eval()
#     loss = 0.0
#     correct_pred = 0
#     total_num = 0
#     total_loss = 0.0
#     i = 0
#     for images, labels in eval_loader:
#         i += 1
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         # outputs = model(images)
#         # loss = criterion(outputs,labels)
#         # pred1, bid1, pred2, bid2 = model(images)
#         # print(bid1[0],bid2[0])
#         # loss1 = -torch.mean((bid1>bid2)*((labels.unsqueeze(-1)*pred1[:,1].unsqueeze(-1) + (1 - labels.unsqueeze(-1))*pred1[:,0].unsqueeze(-1)) - bid2))
#         # loss2 = -torch.mean((bid1<bid2)*((labels.unsqueeze(-1)*pred2[:,1].unsqueeze(-1) + (1 - labels.unsqueeze(-1))*pred2[:,0].unsqueeze(-1)) - bid1))
#         # loss = loss1 + loss2
#         pred, _, bid = model(images)   
#         if i ==1:
#             print(bid[0])
#         # # print(pred.shape) 
#         # # tmp = labels.unsqueeze(-1)*pred[:,1].unsqueeze(-1) + (1 - labels.unsqueeze(-1))*pred[:,0].unsqueeze(-1)#+ (1 - labels.unsqueeze(-1))*pred[:,0] - bid
#         # # print(pred)
#         # # print(tmp,bid)
#         pred = F.softmax(pred, dim=1)
#         loss = -torch.mean(labels.unsqueeze(-1)*pred[:,1].unsqueeze(-1) + (1 - labels.unsqueeze(-1))*pred[:,0].unsqueeze(-1))
#         # loss1 = criterion(pred, labels)
#         # loss = loss1 + torch.mean(bid)
#         total_loss += loss.item()
#         # accuracy
#         # pred = pred1
#         # bid1 = bid1.repeat(1,2)
#         # bid2 = bid2.repeat(1,2)
#         # pred[bid2>bid1]=pred2[bid2>bid1]

#         pred = torch.argmax(pred, dim=1)
#         correct = torch.sum(pred.eq(labels))
#         correct_pred += correct
#         total_num += labels.shape[0]
#     accuracy = correct_pred.cpu()/float(total_num)
#     return total_loss/i, accuracy

def main(epochs):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    device = torch.device('cuda:0')
    train_ds = MyDataset(train_path)

    full_ds = train_ds
    train_size = int(0.8 * len(full_ds))
    validate_size = len(full_ds) - train_size
    new_train_ds, validate_ds = torch.utils.data.random_split(full_ds,[train_size, validate_size])

    new_train_loader = torch.utils.data.DataLoader(new_train_ds, batch_size=64,
                                                shuffle=True, pin_memory=True, num_workers=8)

    validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=64,
                                                shuffle=True, pin_memory=True, num_workers=8)

    criterion = nn.CrossEntropyLoss()

    # model = MyCNN()
    # model = model.to(device)
    agent1 = Agent()
    agent2 = Agent()
    agent1 = agent1.to(device)
    agent2 = agent2.to(device)
    
    optimizer1 = torch.optim.Adam(agent1.parameters(),lr=1e-3)
    optimizer2 = torch.optim.Adam(agent2.parameters(),lr=1e-3)

    best_acc = 0.0
    for e in range(epochs):
        epoch_start_time = time.time()
        # train_loss, train_acc = train(e, new_train_loader, device, model, criterion, optimizer)
        # val_loss, val_acc = eval(e, validate_loader, device, model, criterion, optimizer)
        train_loss, train_acc = train_seperate(new_train_loader, device, agent1, agent2, optimizer1, optimizer2)
        val_loss, val_acc = eval_seperate(validate_loader, device, agent1, agent2, optimizer1, optimizer2)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | train_acc {:5.5f} | val_loss {:5.5f} | val_acc {:5.5f} |'.format(
                        e, (time.time()-epoch_start_time), train_loss, train_acc, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            # model_pth = './model'
            # model_pth = os.path.join(model_pth, str(val_acc)+'model.pt')
            # torch.save(model,model_pth)
            print("model saved")
def test():
    device = torch.device('cuda:0')
    pth = 'model/tensor(1.)model.pt'
    test_model = torch.load(pth)
    test_img = '../datasets/test/1.jpg'
    trans_func = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    img = Image.open(test_img)
    img = trans_func(img)
    img = img.unsqueeze(0).to(device)
    print(img.shape)
    out = test_model(img)
    print(out)
 


if __name__ == "__main__":
    main(100)
    # test()



# 
# for i, item in enumerate(tqdm(train_ds)):
#     # pass
#     print(item[0].shape)
#     break
# img_PIL_Tensor = train_ds[1][0]
# new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
# plt.imshow(new_img_PIL)
# plt.show()