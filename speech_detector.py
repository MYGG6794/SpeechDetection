# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                           QVBoxLayout, QHBoxLayout, QSlider, QLabel, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import asyncio
import websockets
import json
from datetime import datetime

def resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建临时文件夹tempdir，并放置运行文件的脚本于_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class SpeechDetector(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    speech_signal = pyqtSignal(bool)
    log_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.detection_threshold = 0.5
        self.roi_scale = 1.0
        self.min_mouth_aspect_ratio = 0.35
        
        # 性能优化参数
        self.frame_skip = 1  # 每隔多少帧进行一次完整检测
        self.frame_count = 0
        self.target_fps = 30  # 目标帧率
        self.frame_interval = 1.0 / self.target_fps
        self.last_face_position = None  # 缓存上一次检测到的人脸位置
        
        # 状态检测参数
        self.speaking_frames = 0
        self.silent_frames = 0
        self.current_speaking_state = False
        self.SPEAKING_THRESHOLD_FRAMES = 2
        self.SILENT_THRESHOLD_FRAMES = 30

        # 加载分类器
        face_cascade_path = resource_path("models/haarcascade_frontalface_default.xml")
        mouth_cascade_path = resource_path("models/haarcascade_smile.xml")
        
        self.face_cascade = cv2.CascadeClassifier()
        self.mouth_cascade = cv2.CascadeClassifier()
        
        if not self.face_cascade.load(face_cascade_path):
            raise RuntimeError(f"无法加载人脸检测器: {face_cascade_path}")
        if not self.mouth_cascade.load(mouth_cascade_path):
            raise RuntimeError(f"无法加载嘴部检测器: {mouth_cascade_path}")
    
    def detect_faces(self, frame):
        # 缩小图像以提高性能
        scale = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 使用上一次的人脸位置作为ROI
        if self.last_face_position is not None:
            x, y, w, h = self.last_face_position
            # 扩大搜索区域
            x = max(0, int(x * scale - w * 0.2))
            y = max(0, int(y * scale - h * 0.2))
            roi_w = min(small_frame.shape[1] - x, int(w * scale * 1.4))
            roi_h = min(small_frame.shape[0] - y, int(h * scale * 1.4))
            roi = gray[y:y+roi_h, x:x+roi_w]
            
            faces = self.face_cascade.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(15, 15),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # 调整回原始坐标
                faces = [(int((x + x1)/scale), int((y + y1)/scale),
                         int(w1/scale), int(h1/scale)) for (x1, y1, w1, h1) in faces]
                self.last_face_position = faces[0]
                return faces
        
        # 如果没有找到人脸，进行全图搜索
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # 调整回原始坐标
            faces = [(int(x/scale), int(y/scale),
                     int(w/scale), int(h/scale)) for (x, y, w, h) in faces]
            self.last_face_position = faces[0]
        else:
            self.last_face_position = None
            
        return faces

    def detect_mouth_state(self, gray, face):
        x, y, w, h = face
        # 只关注脸部下半部分
        h_start = int(h * 0.6)
        face_roi = gray[y+h_start:y+h, x:x+w]
        
        # 使用直方图均衡化增强对比度
        face_roi = cv2.equalizeHist(face_roi)
        
        mouths = self.mouth_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=20,
            minSize=(25, 15),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(mouths) == 0:
            return False, None
            
        mouth = max(mouths, key=lambda m: m[2] * m[3])
        mx, my, mw, mh = mouth
        
        my += h_start
        aspect_ratio = float(mh) / mw
        speaking_threshold = self.min_mouth_aspect_ratio
        
        is_speaking = aspect_ratio > speaking_threshold
        
        log_info = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "aspect_ratio": float(f"{aspect_ratio:.3f}"),
            "threshold": float(f"{speaking_threshold:.3f}"),
            "is_speaking": is_speaking,
            "mouth_width": mw,
            "mouth_height": mh
        }
        
        self.log_signal.emit(log_info)
        return is_speaking, (mx, my, mw, mh)
        
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_frame_time = datetime.now()
        
        while self.running:
            current_time = datetime.now()
            elapsed = (current_time - last_frame_time).total_seconds()
            
            if elapsed < self.frame_interval:
                continue
                
            last_frame_time = current_time
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            self.frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 每隔frame_skip帧进行一次完整检测
            if self.frame_count % self.frame_skip == 0:
                faces = self.detect_faces(frame)
                
                is_speaking = False
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    speaking, mouth_rect = self.detect_mouth_state(gray, (x, y, w, h))
                    is_speaking = is_speaking or speaking
                    
                    if mouth_rect is not None:
                        mx, my, mw, mh = mouth_rect
                        cv2.rectangle(frame, 
                                    (x + mx, y + my), 
                                    (x + mx + mw, y + my + mh),
                                    (0, 255, 0) if speaking else (0, 0, 255),
                                    2)
                
                if is_speaking:
                    self.speaking_frames += 1
                    self.silent_frames = 0
                    if self.speaking_frames >= self.SPEAKING_THRESHOLD_FRAMES:
                        self.current_speaking_state = True
                else:
                    self.silent_frames += 1
                    if self.silent_frames >= self.SILENT_THRESHOLD_FRAMES:
                        self.current_speaking_state = False
                        self.speaking_frames = 0
            
            self.frame_signal.emit(frame)
            self.speech_signal.emit(bool(self.current_speaking_state))
            
        cap.release()

class WebSocketServer(QThread):
    status_signal = pyqtSignal(str)  # 添加状态信号
    
    def __init__(self, port=8765):
        super().__init__()
        self.port = port
        self.is_speaking = False
        self.running = True
        self.server = None
        self.clients = set()
        self.loop = None
        
    def set_speaking_state(self, state):
        self.is_speaking = state
        
    async def handler(self, websocket):
        # 添加新的客户端连接
        self.clients.add(websocket)
        self.status_signal.emit(f"新客户端连接，当前连接数: {len(self.clients)}")
        
        try:
            while self.running:
                if not self.running:
                    break
                data = {
                    "is_speaking": self.is_speaking,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(data))
                await asyncio.sleep(0.1)  # 控制发送频率
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # 移除断开的客户端
            self.clients.remove(websocket)
            self.status_signal.emit(f"客户端断开，当前连接数: {len(self.clients)}")
                
    async def run_server(self):
        self.server = await websockets.serve(self.handler, "localhost", self.port)
        self.loop = asyncio.get_event_loop()
        await self.server.wait_closed()
            
    def run(self):
        asyncio.run(self.run_server())
        
    async def close_connections(self):
        # 关闭所有客户端连接
        if self.clients:
            await asyncio.gather(*[client.close() for client in self.clients])
            self.clients.clear()
        
    def stop(self):
        self.running = False
        if self.server:
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.close_connections(), self.loop)
                self.server.close()
                # 等待服务器完全关闭
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.server.wait_closed(), self.loop)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时说话检测器")
        self.setGeometry(100, 100, 1000, 800)
        
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 创建水平布局来放置视频和日志
        h_layout = QVBoxLayout()
        
        # 创建视频预览标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        h_layout.addWidget(self.video_label)
        
        # 创建状态标签
        self.status_label = QLabel("等待检测...")
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_layout.addWidget(self.status_label)
        
        layout.addLayout(h_layout)
        
        # 创建 WebSocket 状态显示区域
        ws_status_layout = QVBoxLayout()
        ws_status_label = QLabel("WebSocket 服务器状态")
        ws_status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        ws_status_layout.addWidget(ws_status_label)
        
        self.ws_url_label = QLabel("WebSocket URL: ws://localhost:8765")
        self.ws_url_label.setStyleSheet("font-family: Consolas;")
        ws_status_layout.addWidget(self.ws_url_label)
        
        self.ws_status_text = QLabel("等待客户端连接...")
        self.ws_status_text.setStyleSheet("color: gray;")
        ws_status_layout.addWidget(self.ws_status_text)
        
        layout.addLayout(ws_status_layout)
        
        # 创建日志显示区域，使用QPlainTextEdit替代QTextEdit以提高性能
        log_label = QLabel("实时检测数据:")
        log_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(log_label)

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setFixedHeight(100)
        self.debug_text.setStyleSheet("font-family: Consolas, Courier; font-size: 12px;")
        layout.addWidget(self.debug_text)

        # 状态变化日志
        status_log_label = QLabel("状态变化日志:")
        status_log_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(status_log_label)

        self.log_text = QLabel("检测日志:")
        self.log_text.setStyleSheet("font-family: Consolas, Courier;")
        self.log_text.setWordWrap(True)
        layout.addWidget(self.log_text)
        
        # 创建滑动条
        self.create_slider("检测阈值", 1, 100, 50, layout, lambda x: self.detector.set_detection_threshold(x))
        self.create_slider("检测区域比例", 50, 200, 100, layout, lambda x: self.detector.set_roi_scale(x))
        self.create_slider("最小嘴部开合比例", 1, 100, 20, layout, lambda x: self.detector.set_min_mouth_aspect_ratio(x))
        
        # 初始化检测器和WebSocket服务器
        self.detector = SpeechDetector()
        self.ws_server = WebSocketServer()
        
        # 连接信号
        self.detector.speech_signal.connect(self.ws_server.set_speaking_state)
        self.detector.frame_signal.connect(self.update_video_frame)
        self.detector.speech_signal.connect(self.update_status)
        self.detector.log_signal.connect(self.update_debug_text)
        self.ws_server.status_signal.connect(self.ws_status_text.setText)
        
        # UI更新优化
        self.last_update_time = datetime.now()
        self.update_interval = 0.1  # 100ms
        self.last_state = False
        self.log_entries = []
        
        # 启动线程
        self.detector.start()
        self.ws_server.start()
        
    def create_slider(self, name, min_val, max_val, default, layout, slot):
        # 创建水平布局来放置标签、滑动条和数值
        slider_layout = QHBoxLayout()
        
        # 创建标签
        label = QLabel(name)
        label.setMinimumWidth(150)  # 确保标签有足够的宽度
        slider_layout.addWidget(label)
        
        # 创建滑动条
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        
        # 创建数值标签
        value_label = QLabel(f"{default / 100:.2f}")
        value_label.setMinimumWidth(50)  # 确保数值标签有足够的宽度
        
        # 连接滑动条的值变化信号
        def on_value_changed(value):
            value_label.setText(f"{value / 100:.2f}")
            slot(value)
            
        slider.valueChanged.connect(on_value_changed)
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        
    def update_video_frame(self, frame):
        current_time = datetime.now()
        if (current_time - self.last_update_time).total_seconds() < self.update_interval:
            return
            
        self.last_update_time = current_time
        
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_status(self, is_speaking):
        if is_speaking == self.last_state:
            return
            
        status_text = "检测到说话" if is_speaking else "未检测到说话"
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(
            f"font-size: 24px; font-weight: bold; color: {'green' if is_speaking else 'red'}"
        )
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        state_log = f"[{timestamp}] {'开始说话' if is_speaking else '停止说话'}"
        self.log_entries.insert(0, state_log)
        
        if len(self.log_entries) > 10:
            self.log_entries.pop()
            
        self.log_text.setText("\n".join(self.log_entries))
        self.last_state = is_speaking
        
    def update_debug_text(self, log_info):
        current_time = datetime.now()
        if (current_time - self.last_update_time).total_seconds() < self.update_interval:
            return
            
        debug_text = f"时间: {log_info['timestamp']}\n"
        debug_text += f"嘴部高宽比: {log_info['aspect_ratio']:.3f}\n"
        debug_text += f"当前阈值: {log_info['threshold']:.3f}\n"
        debug_text += f"嘴部尺寸: {log_info['mouth_width']}x{log_info['mouth_height']}\n"
        debug_text += f"检测状态: {'说话中' if log_info['is_speaking'] else '未说话'}"
        
        self.debug_text.setText(debug_text)
        
    def closeEvent(self, event):
        # 优化关闭流程
        self.detector.running = False
        self.ws_server.stop()
        
        # 给线程一些时间来清理
        if not self.detector.wait(1000):  # 等待最多1秒
            self.detector.terminate()
        if not self.ws_server.wait(1000):  # 等待最多1秒
            self.ws_server.terminate()
            
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
