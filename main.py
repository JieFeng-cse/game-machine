import os 
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
import torch
from torch.serialization import save
from torch.utils.data import Dataset
from torch.distributions.normal import Normal
from torchvision import transforms,models,utils
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from layers import Agent, ResnetAgent
import time
import numpy as np
import torch.nn.functional as F
import math
# SEED = 27
train_path = './datasets/train/'
val_path = './datasets/val/'
overfit_path = './datasets/overfit/'
torch.autograd.set_detect_anomaly(True)

class MyDataset(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size = (224,224)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation((0,360)),
                # transforms.GaussianBlur((3,3)),
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

class MyDatasetval(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(size = (224,224)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

def train_seperate(e, train_loader, device, agent1, agent2, optimizer1, optimizer2, writer):
    agent1.train()
    agent2.train()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    if e > 100:
        flag = 0.0
    for images, labels in train_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
    
        pred1, bid1, out1 = agent1(images)
        pred2, bid2, out2 = agent2(images)

        # loss1 = F.cross_entropy(pred1, labels)
        pred1 = F.softmax(pred1, dim=1)
        # bids1 = torch.stack([bid1*beta, bid2.detach()*beta], dim=1)
        # probs1 = F.softmax(bids1, dim=1)
        # c1 = bid1 * probs1[:, 0] #+ bid2.detach() * probs1[:, 1]   
        log_p1 = (torch.log(pred1+math.exp(-70)))
        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)

        # loss1 = -torch.sum(one_hot*log_p1,dim=1).unsqueeze(1) * probs1[:,1] + c1 + F.smooth_l1_loss(bid1, bid2.detach())*flag
        sigma = 0.1*torch.ones_like(bid1).to(device)*torch.sqrt(torch.tensor([2])).to(device)
        ones_t = torch.ones_like(bid1).to(device)
        dis1_1 = Normal(bid1-bid2.detach(), sigma)
        dis1_2_fp = Normal(bid1-bid2.detach()+0.01*ones_t, sigma)
        dis1_2_sp = Normal(bid1-bid2.detach()-0.01*ones_t, sigma)
        p_a1_loss = dis1_1.cdf(torch.zeros_like(bid1).to(device))
        a_1 = torch.sum(one_hot*log_p1,dim=1).unsqueeze(1)
        # loss1 = -torch.sum(one_hot*log_p1,dim=1).unsqueeze(1) * p_a1_loss + torch.exp(bid1+0.5)*(1-dis1_2.cdf(torch.zeros_like(bid1)).to(device))
        fp_loss_1 = torch.exp(bid1+0.005)*(1-dis1_2_fp.cdf(torch.zeros_like(bid1)).to(device))
        sp_loss_1 = torch.exp(bid2.detach()+0.005)*(1-dis1_2_sp.cdf(torch.zeros_like(bid1)).to(device)) 
        # loss1 = -a_1 * p_a1_loss + fp_loss_1*0.1+sp_loss_1*0.9 
        loss1 = -a_1 * p_a1_loss + sp_loss_1*0.9 + fp_loss_1*0.1 #0.3*F.smooth_l1_loss(extra_bid1, torch.zeros_like(extra_bid1).to(device))
        # if i==1:
        #     print(format(bid1[0].item(), '.3f'),format(bid2[0].item(), '.3f'))
        #     # print(format(bid1[1].item(), '.3f'),format(bid2[1].item(), '.3f'))
        #     tmp_bid = torch.exp(bid2.detach()+0.5)[0].item()
        #     tmp_prob = (1-dis1_2.cdf(torch.zeros_like(bid1)))[0].item()
        #     print("agent 1: ", format(float(p_a1_loss[0]), '.3f'), format(tmp_bid, '.3f'), format(tmp_prob, '.3f'), format(tmp_bid * tmp_prob, '.3f'))
        loss1 = loss1.sum()
        loss1/=pred1.shape[0]

        # loss2 = F.cross_entropy(pred2, labels)
        pred2 = F.softmax(pred2, dim=1)
        # bids2 = torch.stack([bid1.detach()*beta, bid2*beta], dim=1)
        # probs2 = F.softmax(bids2, dim=1)
        # c2 =  bid2 * probs2[:, 1] #bid1.detach() * probs2[:, 0] + bid2 * probs2[:, 1] 
        log_p2 = (torch.log(pred2+math.exp(-70)))
        one_hot = torch.zeros(pred2.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        # loss2 = -torch.sum(one_hot*log_p2,dim=1).unsqueeze(1) * probs2[:,0] + c2 + F.smooth_l1_loss(bid1.detach(), bid2)*flag
        dis2_1 = Normal(bid2-bid1.detach(), sigma)
        dis2_2_fp = Normal(bid2-bid1.detach()+0.01*ones_t, sigma) #fp means first price
        dis2_2_sp = Normal(bid2-bid1.detach()-0.01*ones_t, sigma)
        p_a2_loss = dis2_1.cdf(torch.zeros_like(bid2).to(device))
        a_2 = torch.sum(one_hot*log_p2,dim=1).unsqueeze(1)
        # loss2 = -torch.sum(one_hot*log_p2,dim=1).unsqueeze(1) * p_a2_loss + torch.exp(bid2+0.5)*(1-dis2_2.cdf(torch.zeros_like(bid2)).to(device))
        fp_loss_2 = torch.exp(bid2+0.005)*(1-dis2_2_fp.cdf(torch.zeros_like(bid2)).to(device))
        sp_loss_2 = torch.exp(bid1.detach()+0.005)*(1-dis2_2_sp.cdf(torch.zeros_like(bid2)).to(device)) 
        # loss2 = -a_2 * p_a2_loss + fp_loss_2*0.1 + sp_loss_2*0.9
        loss2 = -a_2 * p_a2_loss + sp_loss_2*0.9 + fp_loss_2*0.1#0.3*F.smooth_l1_loss(extra_bid2, torch.zeros_like(extra_bid2).to(device))
        # if i==1:
        #     tmp_bid = torch.exp(bid1.detach()+0.5)[0].item()
        #     tmp_prob = (1-dis2_2.cdf(torch.zeros_like(bid1)))[0].item()
        #     print("agent 2: ", format(float(p_a2_loss[0]), '.3f'), format(tmp_bid, '.3f'), format(tmp_prob, '.3f'), format(tmp_bid * tmp_prob, '.3f'))
        loss2 = loss2.sum()
        loss2/=pred2.shape[0]        
       
        loss1.backward()        
        optimizer1.step()

        loss2.backward()
        optimizer2.step()
        
        # if i == 1:
        #     for name, parms in agent1.named_parameters(): 
        #         if 'feature' in name:
        #             continue
        #         if 'conv' in name:
        #             continue
        #         print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
        #         ' -->grad_value:',parms.grad)
        total_loss += loss1.item()+loss2.item()
        # probs = probs1.detach()
        pred = pred1.detach() * p_a1_loss.detach() + pred2.detach()*p_a2_loss.detach()
        # pred = pred1#*0.5 + pred2*0.5
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    
    accuracy = correct_pred.cpu()/float(total_num)
    writer.add_scalar('loss/train', total_loss, e)
    writer.add_scalar('accuracy/train', accuracy, e)
    # log = np.vstack(log)
    # save_path = f'./log/{e}_{SEED}cnn_log.txt'
    # np.savetxt(save_path, log, fmt='%1.4e')
    return total_loss/i, accuracy

def eval_seperate(e, eval_loader, device, agent1, agent2, writer):
    agent1.eval()
    agent2.eval()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    beta = 1000
    i = 0
    for images, labels in eval_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        pred1, bid1, _ = agent1(images)
        pred2, bid2, _ = agent2(images)
        if i==1:
            print(bid1[0].detach(),bid2[0].detach(),pred1[0].detach())
            print(bid1[1].detach(),bid2[1].detach(),pred1[1].detach())

        # loss1 = F.cross_entropy(pred1, labels)
        pred1 = F.softmax(pred1, dim=1)
        bids1 = torch.stack([bid1*beta, bid2.detach()*beta], dim=1)
        probs1 = F.softmax(bids1, dim=1)
        c1 = bid1 * probs1[:, 0] #+ bid2.detach() * probs1[:, 1]   
        log_p1 = (torch.log(pred1+math.exp(-70)))
        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        # loss1 = -torch.sum(one_hot*log_p1,dim=1).unsqueeze(1) * probs1[:,1] + c1 
        sigma = 0.1*torch.ones_like(bid1).to(device)*torch.sqrt(torch.tensor([2])).to(device)
        ones_t = torch.ones_like(bid1).to(device)
        dis1_1 = Normal(bid1-bid2.detach(), sigma)
        # dis1_2 = Normal(bid1-bid2.detach()+ones_t, sigma)
        dis1_2 = Normal(bid1-bid2.detach()-0.01*ones_t, sigma)
        p_a1_loss = dis1_1.cdf(torch.zeros_like(bid1).to(device))
        # loss1 = -torch.sum(one_hot*log_p1,dim=1).unsqueeze(1) * p_a1_loss + torch.exp(bid1+0.5)*(1-dis1_2.cdf(torch.zeros_like(bid1)).to(device))
        loss1 = -torch.sum(one_hot*log_p1,dim=1).unsqueeze(1) * p_a1_loss + torch.exp(bid2.detach()+0.005)*(1-dis1_2.cdf(torch.zeros_like(bid1)).to(device))
        
        loss1 = loss1.sum()
        loss1/=pred1.shape[0]

        pred2 = F.softmax(pred2, dim=1)
        bids2 = torch.stack([bid1.detach()*beta, bid2*beta], dim=1)
        probs2 = F.softmax(bids2, dim=1)
        # c2 = bid2 * probs2[:, 1] 
        log_p2 = (torch.log(pred2+math.exp(-70)))
        one_hot = torch.zeros(pred2.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        # loss2 = -torch.sum(one_hot*log_p2,dim=1).unsqueeze(1) * probs2[:,0] + c2
        dis2_1 = Normal(bid2-bid1.detach(), sigma)
        # dis2_2 = Normal(bid2-bid1.detach()+ones_t, sigma)
        dis2_2 = Normal(bid2-bid1.detach()-0.01*ones_t, sigma)
        p_a2_loss = dis2_1.cdf(torch.zeros_like(bid2).to(device))
        # loss2 = -torch.sum(one_hot*log_p2,dim=1).unsqueeze(1) * p_a2_loss + torch.exp(bid2+0.5)*(1-dis2_2.cdf(torch.zeros_like(bid2)).to(device))
        loss2 = -torch.sum(one_hot*log_p2,dim=1).unsqueeze(1) * p_a2_loss + torch.exp(bid1.detach()+0.005)*(1-dis2_2.cdf(torch.zeros_like(bid2)).to(device))
        
        loss2 = loss2.sum()
        loss2/=pred2.shape[0]

        total_loss += loss1.item()+loss2.item()
        # probs = probs1.detach()
        pred = pred1* p_a1_loss.detach() + pred2*p_a2_loss.detach()
        # pred = pred1#*0.5 + pred2*0.5
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    writer.add_scalar('loss/val', total_loss, e)
    writer.add_scalar('accuracy/val', accuracy, e)
    return total_loss/i, accuracy


def main(epochs, seed):
    global SEED
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    print(SEED)
    device = torch.device('cuda:0')
    train_ds = MyDataset(train_path)
    val_ds = MyDatasetval(val_path)
    writer = SummaryWriter()

    # full_ds = train_ds
    # train_size = int(0.8 * len(full_ds))
    # validate_size = len(full_ds) - train_size
    # new_train_ds, validate_ds = torch.utils.data.random_split(full_ds,[train_size, validate_size])

    new_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64,
                                                shuffle=True, pin_memory=True, num_workers=8)

    validate_loader = torch.utils.data.DataLoader(val_ds, batch_size=64,
                                                shuffle=True, pin_memory=True, num_workers=8)

    agent1 = ResnetAgent()
    agent2 = ResnetAgent()
    agent1 = agent1.to(device)
    agent2 = agent2.to(device)

    # agent1=torch.load(pth1)
    # agent2=torch.load(pth2)
    # agent1.train()
    # agent2.train()
    
    optimizer1 = torch.optim.Adam(agent1.parameters(),lr=1e-4)
    optimizer2 = torch.optim.Adam(agent2.parameters(),lr=1e-4)

    best_acc = 0.0
    for e in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_seperate(e, new_train_loader, device, agent1, agent2, optimizer1, optimizer2, writer)
        val_loss, val_acc = eval_seperate(e, validate_loader, device, agent1, agent2, writer)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | train_acc {:5.5f} | val_loss {:5.5f} | val_acc {:5.5f} |'.format(
                        e, (time.time()-epoch_start_time), train_loss, train_acc, val_loss, val_acc))
        if not (e+1)%500 or val_acc>=best_acc:
            if val_acc>=best_acc:
                best_acc = val_acc
                model_pth = './model'
                if not os.path.exists(model_pth):
                    os.mkdir(model_pth)
                model1_pth = os.path.join(model_pth, f'cnnmodel1t{SEED}_sto_resnet_best.pt') #nl means no penalty loss
                model2_pth = os.path.join(model_pth, f'cnnmodel2t{SEED}_sto_resnet_best.pt')
                torch.save(agent1,model1_pth)
                torch.save(agent2,model2_pth)
                print("model saved")
            # else:
            #     model_pth = './model'
            #     if not os.path.exists(model_pth):
            #         os.mkdir(model_pth)
            #     model1_pth = os.path.join(model_pth, f'cnnmodel1t{SEED}_sto_know_{e}.pt') #nl means no penalty loss
            #     model2_pth = os.path.join(model_pth, f'cnnmodel2t{SEED}_sto_know_{e}.pt')
            #     torch.save(agent1,model1_pth)
            #     torch.save(agent2,model2_pth)
    print(best_acc)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def test(enum,noise):
    device = torch.device('cuda:0')
    pth1 = f'model/cnnmodel1t27_sto_{enum}.pt'
    pth2 = f'model/cnnmodel2t27_sto_{enum}.pt'
    test_agent1=torch.load(pth1)
    test_agent2=torch.load(pth2)
    test_agent1.eval()
    test_agent2.eval()
    test_img = 'datasets copy/train/cat.1.jpg'
    trans_func = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                # AddGaussianNoise(0., 1.)
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    img = Image.open(test_img)
    img = trans_func(img) + noise
    img = img.unsqueeze(0).to(device)
    # print(img.shape)
    pred1, bid1, _ = test_agent1(img)
    pred2, bid2, _ = test_agent2(img)
    pred1 = F.softmax(pred1)
    pred2 = F.softmax(pred2)
    print(pred1[0,1].detach().cpu(), pred2[0,1].detach().cpu(), torch.exp(bid1[0,0]).detach().cpu(), torch.exp(bid2[0,0]).detach().cpu())
    return pred1[0,1].detach().cpu(), pred2[0,1].detach().cpu(), torch.exp(bid1[0,0]).detach().cpu(), torch.exp(bid2[0,0]).detach().cpu()
 
def test2(e_num):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    device = torch.device('cuda:0')
    train_ds = MyDatasetval(val_path)
    # pth1=f'model/50cnnmodel1t{SEED}nl.pt'
    # pth2=f'model/50cnnmodel2t{SEED}nl.pt'
    pth1 = f'model/cnnmodel1t{SEED}_sto_know_{e_num}.pt'
    pth2 = f'model/cnnmodel2t{SEED}_sto_know_{e_num}.pt'
    pth1 = f'model/cnnmodel1t{SEED}_sto_bound_best.pt'
    pth2 = f'model/cnnmodel2t{SEED}_sto_bound_best.pt'
    # pth1='model/model1.pt'
    # pth2='model/model2.pt'
    test_agent1=torch.load(pth1)
    test_agent2=torch.load(pth2)

    # full_ds = train_ds
    # train_size = int(0.8 * len(full_ds))
    # validate_size = len(full_ds) - train_size
    # new_train_ds, validate_ds = torch.utils.data.random_split(full_ds,[train_size, validate_size])

    validate_loader = torch.utils.data.DataLoader(train_ds, batch_size=2,
                                                shuffle=True, pin_memory=True, num_workers=8)
    save_pth = f'50cnn_log{SEED}_sto_know_best.txt'
    test_write(save_pth, validate_loader,device,test_agent1,test_agent2)

def test_write(save_pth, eval_loader, device, agent1, agent2):
    agent1.eval()
    agent2.eval()
    correct_pred = 0
    total_num = 0

    i=0
    log=[]
    for images, labels in eval_loader:
        i+=1
        images = images.to(device)
        # if i<100:
        #     images += torch.randn(images.shape).cuda()
        labels = labels.to(device)
        pred1, bid1, _ = agent1(images)
        pred2, bid2, _ = agent2(images)

        pred1 = F.softmax(pred1, dim=1)
        pred2 = F.softmax(pred2, dim=1)

        one_hot = torch.zeros(pred1.shape,dtype=float,device='cuda:0').scatter_(1, torch.unsqueeze(labels, dim=1),1)
        log_pred1= torch.log(pred1+math.exp(-70))
        log_pred2= torch.log(pred2+math.exp(-70))
        
        v1 = torch.sum(one_hot*log_pred1,dim=1).unsqueeze(1)
        v2 = torch.sum(one_hot*log_pred2,dim=1).unsqueeze(1)
        sigma = 0.1*torch.ones_like(bid1).to(device)*torch.sqrt(torch.tensor([2])).to(device)
        
        dis1_1 = Normal(bid1-bid2.detach(), sigma)
        p_a1_loss = dis1_1.cdf(torch.zeros_like(bid1).to(device))
        dis2_1 = Normal(bid2-bid1.detach(), sigma)
        p_a2_loss = dis2_1.cdf(torch.zeros_like(bid2).to(device))

        pred = pred1*p_a1_loss + pred2*p_a2_loss
        pred = torch.argmax(pred, dim=1)
        tmp = torch.cat([v1,torch.exp(bid1),v2,torch.exp(bid2),labels.unsqueeze(-1),pred.unsqueeze(-1)],dim=1)
        # tmp = torch.cat([v1,(bid1),v2,(bid2),labels.unsqueeze(-1),pred.unsqueeze(-1)],dim=1)
        log.append(tmp.detach().cpu().numpy())
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    print(accuracy, SEED)
    log = np.vstack(log)
    np.savetxt(save_pth,log, fmt='%1.4e')
    
if __name__ == "__main__":
    seed_set = set([0,2,4,13,18,25,26,28,42,53,73,82,102,114,115,132,139,150,154,168,172,173,186])
    # for i in range(10):
    #     seed = np.random.randint(0,200,1)[0]
    #     while True:
    #         if int(seed) in seed_set:
    #             seed = np.random.randint(0,200,1)[0]
    #         else:
    #             seed_set.add(int(seed))
    #             break
    seed = 49#27 #0.7918 1710 with added loss, 0.78600, 1692 without added loss, 0.7916, half, 0.7858, single agent no bid
    # for seed in [13,25,51,101]:
    #     main(500, seed)
    main(20000, seed) # normal 1500,0.7768; 0.7858
    # seeds = [0,18,25,26,28,42,53,102,114,115,132,172]
    global SEED
    SEED = seed
    # p1_l = []
    # p2_l = []
    # b1_l = []
    # b2_l = []
    # x = []
    # noise = torch.randn([3,224,224])*0.0#5
    # for i in range(20): #38
    #     p1,p2,b1,b2 = test(enum=i*500+499,noise=noise)
    #     x.append(i*500+499)
    #     p1_l.append(float(p1))
    #     p2_l.append(float(p2))
    #     b1_l.append(float(b1))
    #     b2_l.append(float(b2))
    # plt.plot(x,p1_l,label='Prediction of agent 1')
    # plt.plot(x,p2_l,label='Prediction of agent 2')
    # plt.plot(x,b1_l, marker='o', linestyle='dashed',label='Bid of agent 1')
    # plt.plot(x,b2_l, marker='o', linestyle='dashed',label='Bid of agent 2')
    # plt.xlabel('Checkpoint number')
    # plt.ylabel('Bid or Prediction')
    # plt.legend()
    # plt.show()
    
    

    # test2(499)
    # for i in range(50):
    #     test2(5*i)



# 
# for i, item in enumerate(tqdm(train_ds)):
#     # pass
#     print(item[0].shape)
#     break
# img_PIL_Tensor = train_ds[1][0]
# new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
# plt.imshow(new_img_PIL)
# plt.show()