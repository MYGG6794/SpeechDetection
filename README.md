# 实时说话检测器

这是一个基于计算机视觉的实时说话检测程序，可以检测摄像头画面中的人是否在说话，并通过WebSocket实时发送检测结果。

## 特性

- 实时唇形检测和说话状态识别
- 可视化调节界面，包括检测阈值、检测区域和嘴部开合比例
- WebSocket服务器实时推送检测结果
- 低资源占用，高性能运行

## 安装

1. 确保您的系统已安装Python 3.8或更高版本
2. 创建并激活虚拟环境：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```
3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行程序：
```bash
python speech_detector.py
```

2. 调节参数：
- 检测阈值：调节检测的灵敏度
- 检测区域比例：调节检测区域的大小
- 最小嘴部开合比例：调节判定为说话状态的嘴部开合程度

3. WebSocket接口：
- 默认端口：8765
- 连接地址：ws://localhost:8765
- 数据格式：
```json
{
    "is_speaking": true/false,
    "timestamp": "2025-06-09T10:00:00.000Z"
}
```

## 系统要求

- Windows/Linux/MacOS
- Python 3.8+
- 摄像头
- 4GB及以上内存