import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import json

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu" #单图片推理完全没必要GPU，反而慢了...

classeNames = ['无花果', '柠檬', '榴莲', '橘子', '番荔枝', '石榴', '草莓', '菠萝', '青苹果', '香蕉']

with open('MixNet_class_to_idx.json', 'r') as f:
    class_to_idx = json.load(f)
idx_to_class = {int(i): c for c, i in class_to_idx.items()}

# 使用 MixNet 预训练模型
class ModifiedMixNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedMixNet, self).__init__()
        # 加载预训练的 MixNet 模型
        self.mixnet = timm.create_model('mixnet_s', pretrained=False)#我们读取自己的权重，不需要再次加载预训练
        # 替换 MixNet 的分类器层
        self.mixnet.classifier = nn.Linear(self.mixnet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.mixnet(x)

# 创建模型实例
num_classes = len(classeNames)  # 使用与训练时相同的类别数量
model = ModifiedMixNet(num_classes).to(device)

# 加载模型权重
model.load_state_dict(torch.load('MixNet_best_checkpoint.pt'))
model=model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),  # 转换为RGB
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载并预处理图像
img = Image.open("橘子.jpeg")
img = transform(img).unsqueeze(0)  # 增加一个批次维度

# 进行预测
with torch.no_grad():
    outputs = model(img.to(device))
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)

predicted_class = idx_to_class[predicted.item()]

for idx, prob in enumerate(probabilities[0]):
    print(f"{idx_to_class[idx]}: {prob.item() * 100:.2f}%")

print(f"-----------\n最终结果: {predicted_class}")
