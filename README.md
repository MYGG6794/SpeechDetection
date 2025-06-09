# 实时说话检测器

这是一个基于计算机视觉的实时说话检测程序，可以通过摄像头实时检测用户是否在说话。

## 功能特点

- 实时人脸检测和跟踪
- 实时嘴部状态检测
- WebSocket服务器支持，可以与其他应用程序集成
- 可调节的检测参数
- 实时状态显示和日志记录
- 性能优化，支持高帧率运行

## 系统要求

- Windows 10 或更高版本
- 摄像头
- 运行环境:
  - Python 3.13 或更高版本（如果从源码运行）
  - OpenCV-Python 4.8.0.74
  - NumPy 2.3.0
  - PyQt6 6.9.1
  - WebSockets 15.0.1

## 安装说明

### 使用已编译的可执行文件

1. 下载最新的发布版本
2. 解压文件到任意目录
3. 运行 `dist\SpeechDetector.exe`

### 从源码运行

1. 克隆仓库：
```powershell
git clone [repository-url]
cd SpeechDetection
```

2. 安装依赖：
```powershell
pip install -r requirements.txt
```

3. 运行程序：
```powershell
python speech_detector.py
```

### 打包说明

如果要自行打包程序:

1. 安装打包工具:
```powershell
pip install pyinstaller
```

2. 执行打包命令:
```powershell
pyinstaller --noconfirm --clean --onefile --name SpeechDetector --add-data "models/*;models/" --noconsole --collect-all numpy --collect-all cv2 speech_detector.py
```

3. 打包完成后，可执行文件 `SpeechDetector.exe` 将位于 `dist` 目录下

## 使用说明

1. 启动程序后，会自动打开摄像头并开始检测
2. 调整参数滑动条可以优化检测效果：
   - 检测阈值：调整检测的灵敏度
   - 检测区域比例：调整人脸检测区域大小
   - 最小嘴部开合比例：调整说话判定的阈值

3. WebSocket集成：
   - WebSocket服务器运行在 `ws://localhost:8765`
   - 发送的数据格式为 JSON：
     ```json
     {
       "is_speaking": true/false,
       "timestamp": "ISO格式时间戳"
     }
     ```

## 性能优化说明

- 使用帧率控制确保稳定运行
- 实现人脸位置追踪，减少重复检测
- UI更新频率限制，降低系统负载
- 支持可调节的检测精度

## 更新日志

### 2025.6.9
- 优化性能，提高检测准确度
- 修复程序关闭时卡住的问题
- 改进UI更新逻辑，提升流畅度
- 添加帧率控制
- 优化人脸检测算法

## 问题反馈

如有问题或建议，请提交 Issue。