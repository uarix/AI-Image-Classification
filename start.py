import streamlit as st
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
import json

# 加载预训练的 Mask R-CNN 模型
weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = maskrcnn_resnet50_fpn_v2(weights=weights)
model.eval()
colors = {'🍌香蕉': 'yellow', '🍎苹果': 'red', '🍊橘子': 'orange'}
classeNames = ['无花果', '柠檬', '榴莲', '橘子', '番荔枝', '石榴', '草莓', '菠萝', '青苹果', '香蕉']
with open('ResNet_class_to_idx.json', 'r') as f:
    class_to_idx = json.load(f)
idx_to_class = {int(i): c for c, i in class_to_idx.items()}
# 图像预处理
def transform_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)

def classify_image(image):
    
    

    # 定义模型结构
    class ModifiedResNet(nn.Module):
        def __init__(self, num_classes):
            super(ModifiedResNet, self).__init__()
            weights = ResNet18_Weights.DEFAULT
            self.resnet = resnet18(weights=weights)
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.resnet.fc.in_features, num_classes)
            )

        def forward(self, x):
            return self.resnet(x)

    # 创建模型实例
    num_classes = len(classeNames)  # 使用与训练时相同的类别数量
    model = ModifiedResNet(num_classes)

    # 加载模型权重
    model.load_state_dict(torch.load('ResNet_best_checkpoint.pt'))
    model.eval()

    # 图像预处理
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),  # 转换为RGB
        T.Resize([256, 256]),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载并预处理图像
    img = image
    img = transform(img).unsqueeze(0)  # 增加一个批次维度

    # 进行预测
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    predicted_class = idx_to_class[predicted.item()]
    return predicted_class,probabilities[0]

# 实例分割并计数
def instance_segmentation(image, threshold=0.15):
    image_tensor = transform_image(image.convert('RGB'))

    with torch.no_grad():
        prediction = model(image_tensor)

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='simhei.ttf', size=20)
    counts = {'🍌香蕉': 0, '🍎苹果': 0, '🍊橘子': 0}
    

    for element in range(len(prediction[0]['boxes'])):
        score = prediction[0]['scores'][element].item()
        if score > threshold:
            label = prediction[0]['labels'][element].item()
            box = prediction[0]['boxes'][element].cpu().numpy()
            label_name = None
            if label == 52:  # 香蕉
                label_name = '🍌香蕉'
            elif label == 53:  # 苹果
                label_name = '🍎苹果'
            elif label == 55:  # 橘子
                label_name = '🍊橘子'

            if label_name:
                counts[label_name] += 1
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=colors[label_name], width=3)
                draw.text((box[2], box[1]), label_name, fill=(0, 255, 0), font=font)

    return image, counts

# 创建交互式统计图
def create_chart(counts):
    df = pd.DataFrame(list(counts.items()), columns=['水果', '数量'])
    fig = px.bar(df, x='数量', y='水果', orientation='h', color='水果', height=400)
    return fig

# 设置侧边栏
st.sidebar.title("🔍 图像识别")
option = st.sidebar.radio('选择功能', ('识别水果数量', '识别水果种类'))
uploaded_file = st.sidebar.file_uploader("📤 上传图像文件", type=["png", "jpg", "jpeg"])

# Streamlit 应用主体
st.title("AI 水果识别与计数")

# 如果没有上传图片，展示项目介绍
if uploaded_file is None:
    st.write("""
        ## 🌟 项目简介

        **图像识别** 是一个基于深度学习的图像分析工具，旨在通过高级计算机视觉模型识别和计数图像中的物体。
        
        - **🔬 技术**: 应用使用了 Mask R-CNN 模型，一个在图像识别和分割领域表现出色的深度学习模型。
        - **🍇 功能**: 自动检测上传的图片中的特定物体（如香蕉、苹果、橘子），并计数每种物体的数量。
        - **🎯 用途**: 可用于零售业库存管理、农业研究、教育目的等多种场景。

        上传图片试试吧！
    """)
else:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='上传的图像', use_column_width=True)
    if option == '识别水果数量':
        processed_image, counts = instance_segmentation(image)

        # 在侧边栏展示各种水果数量
        st.sidebar.markdown("### 📊 水果数量统计")
        for fruit, count in counts.items():
            st.sidebar.markdown(f"<span style='color:{colors[fruit]}; font-size:20px; font-weight:bold;'>**{fruit}**: {count}</span>", unsafe_allow_html=True)
        # 展示统计结果
        chart = create_chart(counts)
        st.plotly_chart(chart)

        # 展示处理后的图像
        st.image(processed_image, caption='识别结果', use_column_width=True)
    else:  # '识别水果种类'
        predicted_class,probabilities = classify_image(image)

        # 在侧边栏展示各种水果数量
        st.sidebar.markdown("### 📊 水果种类置信度")
        for idx, prob in enumerate(probabilities):
            st.sidebar.markdown(f"<span style='font-size:15px; font-weight:bold;'>**{idx_to_class[idx]}**: {prob.item() * 100:.2f}%</span>", unsafe_allow_html=True)
        
        # 展示预测的水果种类
        st.write(f"### 🍇 预测的水果种类: {predicted_class}")
        st.write(f"### 📈 置信度:")
        df_probabilities = pd.DataFrame({
            '水果': [idx_to_class[idx] for idx, _ in enumerate(probabilities)],
            '置信度 (%)': [prob.item() * 100 for prob in probabilities]
        })
        fig = px.bar(df_probabilities, x='置信度 (%)', y='水果', orientation='h', color='水果', height=400)
        st.plotly_chart(fig)

