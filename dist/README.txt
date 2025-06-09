语音检测器使用说明
================

基本使用
--------
1. 直接双击运行 SpeechDetector.exe 即可启动程序
2. 程序会自动检测摄像头并开始进行面部和嘴部动作检测
3. 使用界面上的滑块可以调整检测的灵敏度：
   - 检测阈值：值越大，需要更明显的嘴部动作才会触发说话状态
   - ROI比例：调整感兴趣区域的大小
   - 最小高宽比：调整触发说话状态所需的最小嘴部开合程度
4. 界面会实时显示当前的检测状态和参数

WebSocket接口说明
--------------
本程序提供了WebSocket接口，可以让其他应用程序实时获取说话状态：

1. 连接信息：
   - WebSocket URL: ws://localhost:8765
   - 协议：ws (WebSocket)
   - 端口：8765

2. 数据格式：
   服务器每100ms发送一次JSON格式的状态数据：
   ```json
   {
     "is_speaking": true|false,    // 当前是否在说话
     "timestamp": "2025-06-09T10:30:45.123456"  // ISO格式的时间戳
   }
   ```

3. 示例代码：
   JavaScript:
   ```javascript
   const ws = new WebSocket('ws://localhost:8765');
   ws.onmessage = function(event) {
     const data = JSON.parse(event.data);
     console.log('说话状态:', data.is_speaking);
     console.log('时间:', data.timestamp);
   };
   ```

   Python:
   ```python
   import websockets
   import asyncio
   import json

   async def connect():
       async with websockets.connect('ws://localhost:8765') as websocket:
           while True:
               data = json.loads(await websocket.recv())
               print('说话状态:', data['is_speaking'])
               print('时间:', data['timestamp'])

   asyncio.run(connect())
   ```

注意事项
-------
- 请确保摄像头正常工作并且光线充足
- 面对摄像头，保持适当距离（建议30-50厘米）
- 如果检测不稳定，可以调整滑块参数来优化效果
- WebSocket服务器只接受本地连接(localhost)
- 每个WebSocket客户端连接后会自动接收实时状态更新

故障排除
-------
1. 如果看不到摄像头画面：
   - 检查摄像头是否正确连接
   - 确认其他程序没有占用摄像头
   
2. 如果无法连接WebSocket：
   - 确认程序正在运行
   - 检查端口8765是否被其他程序占用
   - 确认防火墙设置不会阻止本地连接

技术支持
-------
如有问题，请访问项目的GitHub页面提交问题：
https://github.com/yourusername/SpeechDetection
