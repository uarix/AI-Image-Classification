import torch

# 加载预训练的 YOLOv5s 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# 图片的路径或URL
imgs = ['8.jpg']  # 你可以更换为本地图片路径或其他图片URL

# 进行推理
results = model(imgs)

# 打印结果
results.print()  # 打印结果到控制台
results.show()  # 结果图片显示

# 提取预测结果
predicted_data = results.pandas().xyxy[0]  # 提取为 pandas DataFrame 格式
print(predicted_data)
