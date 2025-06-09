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
    log_signal = pyqtSignal(dict)  # 添加日志信号
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.detection_threshold = 0.5
        self.roi_scale = 1.0
        self.min_mouth_aspect_ratio = 0.35  # 增加默认阈值
        
        # 添加状态持续时间计数器
        self.speaking_frames = 0
        self.silent_frames = 0
        self.current_speaking_state = False
        self.SPEAKING_THRESHOLD_FRAMES = 2  # 至少检测到2帧说话才触发
        self.SILENT_THRESHOLD_FRAMES = 30   # 至少30帧未检测到说话才认为停止说话

        # 使用resource_path获取正确的模型文件路径
        face_cascade_path = resource_path("models/haarcascade_frontalface_default.xml")
        mouth_cascade_path = resource_path("models/haarcascade_smile.xml")
        
        self.face_cascade = cv2.CascadeClassifier()
        self.mouth_cascade = cv2.CascadeClassifier()
        
        if not self.face_cascade.load(face_cascade_path):
            raise RuntimeError(f"无法加载人脸检测器: {face_cascade_path}")
        if not self.mouth_cascade.load(mouth_cascade_path):
            raise RuntimeError(f"无法加载嘴部检测器: {mouth_cascade_path}")
        
    def set_detection_threshold(self, value):
        self.detection_threshold = value / 100.0
        
    def set_roi_scale(self, value):
        self.roi_scale = value / 100.0
        
    def set_min_mouth_aspect_ratio(self, value):
        self.min_mouth_aspect_ratio = value / 100.0
        
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [tuple(map(int, face)) for face in faces]
        
    def detect_mouth_state(self, gray, face):
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        
        # 优化 ROI 区域 - 主要关注下半部分人脸
        h_start = int(h * 0.6)  # 从脸部60%处开始检测
        face_roi = face_roi[h_start:h, :]
        
        # 检测嘴部
        mouths = self.mouth_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=20,
            minSize=(25, 15),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(mouths) == 0:
            return False, None
            
        # 获取最大的检测结果作为嘴部区域
        mouth = max(mouths, key=lambda m: m[2] * m[3])
        mx, my, mw, mh = mouth
        
        # 调整y坐标以匹配原始人脸坐标
        my += h_start
        
        # 计算嘴部纵横比 (高宽比，值越大说明嘴巴越张开)
        aspect_ratio = float(mh) / mw
        
        # 计算当前阈值
        speaking_threshold = self.min_mouth_aspect_ratio
        
        # 根据高宽比判断说话状态（张嘴时，高度增加，比例变大）
        is_speaking = aspect_ratio > speaking_threshold
        
        # 添加日志信息
        log_info = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "aspect_ratio": float(f"{aspect_ratio:.3f}"),
            "threshold": float(f"{speaking_threshold:.3f}"),
            "is_speaking": is_speaking,
            "mouth_width": mw,
            "mouth_height": mh
        }
        
        # 发送日志信息
        self.log_signal.emit(log_info)
            
        return is_speaking, (mx, my, mw, mh)
        
    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(frame)
            
            is_speaking = False
            for (x, y, w, h) in faces:
                # 绘制人脸框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # 检测嘴部状态
                speaking, mouth_rect = self.detect_mouth_state(gray, (x, y, w, h))
                is_speaking = is_speaking or speaking
                
                if mouth_rect is not None:
                    mx, my, mw, mh = mouth_rect
                    cv2.rectangle(frame, 
                                (x + mx, y + my), 
                                (x + mx + mw, y + my + mh),
                                (0, 255, 0) if speaking else (0, 0, 255),
                                2)
            
            # 更新说话状态计数器
            if is_speaking:
                self.speaking_frames += 1
                self.silent_frames = 0
                if self.speaking_frames >= self.SPEAKING_THRESHOLD_FRAMES:
                    self.current_speaking_state = True
            else:
                self.silent_frames += 1
                if self.silent_frames >= self.SILENT_THRESHOLD_FRAMES:
                    self.current_speaking_state = False
                    self.speaking_frames = 0  # 重置说话帧计数器
            
            # 发送每一帧和当前状态
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
        
    def set_speaking_state(self, state):
        self.is_speaking = state
        
    async def handler(self, websocket):
        # 添加新的客户端连接
        self.clients.add(websocket)
        self.status_signal.emit(f"新客户端连接，当前连接数: {len(self.clients)}")
        
        try:
            while self.running:
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
        await self.server.wait_closed()
            
    def run(self):
        asyncio.run(self.run_server())
        
    def stop(self):
        self.running = False
        if self.server:
            self.server.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时说话检测器")
        self.setGeometry(100, 100, 1000, 800)  # 增加窗口大小
        
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
        
        # 创建日志显示区域
        log_label = QLabel("实时检测数据:")
        log_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(log_label)

        # 创建用于显示实时检测数据的文本框
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setFixedHeight(100)
        self.debug_text.setStyleSheet("font-family: Consolas, Courier; font-size: 12px;")
        layout.addWidget(self.debug_text)

        # 创建状态变化日志标签
        status_log_label = QLabel("状态变化日志:")
        status_log_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(status_log_label)

        # 创建用于显示状态变化的文本标签
        self.log_text = QLabel("检测日志:")
        self.log_text.setStyleSheet("font-family: Consolas, Courier;")
        self.log_text.setWordWrap(True)  # 允许文本换行
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
        self.detector.log_signal.connect(self.update_debug_text)  # 连接检测日志信号
        self.ws_server.status_signal.connect(self.ws_status_text.setText)
        
        # 用于日志记录
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
        # 将OpenCV的BGR图像转换为Qt可显示的格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        
        # 根据标签大小缩放图像
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_status(self, is_speaking):
        # 更新状态标签
        status_text = "检测到说话" if is_speaking else "未检测到说话"
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(
            f"font-size: 24px; font-weight: bold; color: {'green' if is_speaking else 'red'}"
        )
        
        # 记录检测状态变化
        if is_speaking != self.last_state:
            timestamp = datetime.now().strftime("%H:%M:%S")
            state_log = f"[{timestamp}] {'开始说话' if is_speaking else '停止说话'}"
            self.log_entries.insert(0, state_log)
            
            # 保持最近的10条日志
            if len(self.log_entries) > 10:
                self.log_entries.pop()
            
        self.last_state = is_speaking
        
    def update_debug_text(self, log_info):
        # 更新实时检测数据显示
        debug_text = f"时间: {log_info['timestamp']}\n"
        debug_text += f"嘴部高宽比: {log_info['aspect_ratio']:.3f}\n"
        debug_text += f"当前阈值: {log_info['threshold']:.3f}\n"
        debug_text += f"嘴部尺寸: {log_info['mouth_width']}x{log_info['mouth_height']}\n"
        debug_text += f"检测状态: {'说话中' if log_info['is_speaking'] else '未说话'}"
        
        self.debug_text.setText(debug_text)
        
    def closeEvent(self, event):
        self.detector.running = False
        self.detector.wait()
        self.ws_server.stop()
        self.ws_server.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
