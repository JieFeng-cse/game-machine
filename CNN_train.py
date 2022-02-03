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
from layers import CNN_trainer, Agent
import time
import numpy as np
import torch.nn.functional as F
import math

SEED = 17
train_path = './datasets/train/'
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

def train_seperate(e, train_loader, device, agent, optimizer):
    agent.train()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in train_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
    
        pred = agent(images)

        loss1 = F.cross_entropy(pred, labels)

        with torch.autograd.detect_anomaly():
            loss1.backward()
        optimizer.step()

        total_loss += loss1.item()
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    return total_loss/i, accuracy

def eval_seperate(e, eval_loader, device, agent, optimizer):
    agent.eval()
    correct_pred = 0
    total_num = 0
    total_loss = 0.0
    i = 0
    for images, labels in eval_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
    
        pred = agent(images)

        loss1 = F.cross_entropy(pred, labels)

        total_loss += loss1.item()
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred.eq(labels))
        correct_pred += correct
        total_num += labels.shape[0]
    accuracy = correct_pred.cpu()/float(total_num)
    return total_loss/i, accuracy


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

    agent1 = CNN_trainer()
    agent1 = agent1.to(device)
    
    optimizer1 = torch.optim.Adam(agent1.parameters(),lr=1e-3)

    best_acc = 0.0
    for e in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_seperate(e, new_train_loader, device, agent1, optimizer1)
        val_loss, val_acc = eval_seperate(e, validate_loader, device, agent1, optimizer1)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | train_acc {:5.5f} | val_loss {:5.5f} | val_acc {:5.5f} |'.format(
                        e, (time.time()-epoch_start_time), train_loss, train_acc, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            model_pth = './model'
            if not os.path.exists(model_pth):
                os.mkdir(model_pth)
            model1_pth = os.path.join(model_pth, 'cnn_extractor.pt')
            torch.save(agent1.conv.state_dict(),model1_pth)
            print("model saved")

if __name__ == "__main__":
    main(100)