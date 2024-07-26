import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import timm

import json

import os,PIL,random,pathlib
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

data_dir = './data/'
data_dir_path = pathlib.Path(data_dir)

data_paths = list(data_dir_path.glob('*'))
classeNames = [str(path).split("\\")[1] for path in data_paths]

# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据预处理 - 添加数据增强
train_transforms = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 亮度和对比度调整
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用 MixNet 预训练模型
class ModifiedMixNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedMixNet, self).__init__()
        # 加载预训练的 MixNet 模型
        self.mixnet = timm.create_model('mixnet_s', pretrained=True)
        # 替换 MixNet 的分类器层
        self.mixnet.classifier = nn.Linear(self.mixnet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.mixnet(x)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 当验证损失在patience个epoch内没有改善时，训练将停止
            verbose (bool): 如果为True，当早停被触发时，将输出一条消息
            delta (float): 表示损失的最小变化，小于此值将认为没有改善
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta

        loss_path = 'MixNet_best_loss.json'
        if os.path.isfile(loss_path):
            print("[INFO]发现已保存的loss，将继续比较")
            with open(loss_path, 'r') as f:
                self.val_loss_min = float(f.read())
                self.best_score = -self.val_loss_min
            print("[INFO]上一次loss为",self.val_loss_min)
        else:
            print("[INFO]没有发现已保存的loss，初始化为无穷大")
            self.val_loss_min = np.Inf
            self.best_score = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'[INFO]早停法计数: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型'''
        if self.verbose:
            print(f'[INFO]验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}).  正在保存模型...')
        torch.save(model.state_dict(), 'MixNet_best_checkpoint.pt')
        # 保存loss记录
        loss_path = 'MixNet_best_loss.json'
        print("[INFO]更新loss记录")
        with open(loss_path, 'w') as f:
            f.write(str(val_loss))
        self.val_loss_min = val_loss

def test(model, test_loader, loss_model):
    print("[INFO]开始测试")
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_model(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"[INFO]测试结果: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct,test_loss

def train(model,train_loader,loss_model,optimizer):
    print("[INFO]开始训练")
    model=model.to(device)
    model.train()
    
    for i, (images, labels) in enumerate(train_loader, 0):

        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_model(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:    
            print('[%4d] loss: %.3f' % (i, loss))

def main():
    # 加载数据集
    print(classeNames)

    total_data = datasets.ImageFolder(data_dir, transform=train_transforms)

    # 保存 class_to_idx 映射
    class_to_idx = total_data.class_to_idx
    with open('MixNet_class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f)

    train_size = int(0.8 * len(total_data))
    test_size = len(total_data) - train_size
    train_dataset, test_dataset = random_split(total_data, [train_size, test_size])
    
    
    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(classeNames)
    model = ModifiedMixNet(num_classes).to(device)
    # Adam 学习率优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 图像分类任务，并且是多分类任务，选用CrossEntropyLoss(交叉熵损失函数)
    loss_model = nn.CrossEntropyLoss()

    # 检查是否有已保存的模型
    model_path = 'MixNet_best_checkpoint.pt'
    if os.path.isfile(model_path):
        print("[INFO]发现已保存的模型，将继续训练")
        model.load_state_dict(torch.load(model_path))
    else:
        print("[INFO]没有发现已保存的模型，将从头开始训练")
    
    # 训练和测试循环
    test_acc_list = []
    epochs = 30
    early_stopping = EarlyStopping(patience=7, verbose=True)
    # 使用学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for t in range(epochs):
        print(f"Epoch {t+1} -------------------------------")

        train(model,train_loader,loss_model,optimizer)
        test_acc,test_loss = test(model, test_loader, loss_model)
        test_acc_list.append(test_acc)

        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("[Warn]早停法触发，训练终止")
            break

        scheduler.step()  # 更新学习率
    
    #torch.save(model.state_dict(), 'model_state_dict_resnet.pth')
    print("[INFO]All Epochs Are Done!")

    x = [i for i in range(1,t+2)]

    plt.plot(x, test_acc_list, label="Accuracy", alpha=0.8)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()    
    plt.show()

if __name__ == '__main__':
    main()
