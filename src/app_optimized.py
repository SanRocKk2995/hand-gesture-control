
"""
Hand Gesture Control - User Friendly Application
Phiên bản thân thiện người dùng:
- Giao diện đơn giản, dễ sử dụng
- Chế độ nhà phát triển ẩn các tính năng debug
- Tối ưu hiệu năng tự động
"""

import sys
import os
import time

# Thêm đường dẫn
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, base_path)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QMessageBox,
    QTabWidget, QScrollArea, QLineEdit, QSlider, QCheckBox,
    QGroupBox, QTextEdit, QSpinBox, QComboBox, QSizePolicy,
    QStackedWidget, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QIcon, QFont

import json
import cv2
import numpy as np

# === KeyBindButton - Widget bắt phím ===
class KeyBindButton(QPushButton):
    """Button cho phép người dùng bấm phím để bind"""
    key_bound = pyqtSignal(str)  # Signal khi có phím mới
    
    def __init__(self, current_key="", parent=None):
        super().__init__(parent)
        self.bound_key = current_key
        self.is_recording = False
        self.update_display()
        self.setMinimumWidth(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("""
            QPushButton {
                background: #2a2a4a;
                border: 2px solid #4f46e5;
                border-radius: 6px;
                padding: 6px 10px;
                color: white;
                text-align: left;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #3a3a5a;
            }
            QPushButton:focus {
                border-color: #22c55e;
                background: #1a3a2a;
            }
        """)
        self.clicked.connect(self.start_recording)
    
    def update_display(self):
        if self.is_recording:
            self.setText("⌨️ Nhấn phím...")
        elif self.bound_key:
            self.setText(f"🔑 {self.bound_key}")
        else:
            self.setText("Click để gán phím")
    
    def start_recording(self):
        self.is_recording = True
        self.update_display()
        self.setFocus()
    
    def stop_recording(self):
        self.is_recording = False
        self.update_display()
    
    def keyPressEvent(self, event):
        if not self.is_recording:
            super().keyPressEvent(event)
            return
        
        # Bỏ qua chỉ modifier keys
        if event.key() in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return
        
        # Xây dựng tổ hợp phím
        modifiers = event.modifiers()
        key_parts = []
        
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            key_parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            key_parts.append("alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            key_parts.append("shift")
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            key_parts.append("win")
        
        # Lấy tên phím
        key = event.key()
        key_name = self.get_key_name(key)
        
        if key_name:
            key_parts.append(key_name)
            self.bound_key = "+".join(key_parts)
            self.key_bound.emit(self.bound_key)
        
        self.stop_recording()
    
    def get_key_name(self, key):
        """Chuyển đổi Qt key thành tên phím"""
        key_map = {
            Qt.Key.Key_Space: "space",
            Qt.Key.Key_Return: "enter",
            Qt.Key.Key_Enter: "enter",
            Qt.Key.Key_Escape: "escape",
            Qt.Key.Key_Tab: "tab",
            Qt.Key.Key_Backspace: "backspace",
            Qt.Key.Key_Delete: "delete",
            Qt.Key.Key_Home: "home",
            Qt.Key.Key_End: "end",
            Qt.Key.Key_PageUp: "pageup",
            Qt.Key.Key_PageDown: "pagedown",
            Qt.Key.Key_Insert: "insert",
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
            Qt.Key.Key_F1: "f1",
            Qt.Key.Key_F2: "f2",
            Qt.Key.Key_F3: "f3",
            Qt.Key.Key_F4: "f4",
            Qt.Key.Key_F5: "f5",
            Qt.Key.Key_F6: "f6",
            Qt.Key.Key_F7: "f7",
            Qt.Key.Key_F8: "f8",
            Qt.Key.Key_F9: "f9",
            Qt.Key.Key_F10: "f10",
            Qt.Key.Key_F11: "f11",
            Qt.Key.Key_F12: "f12",
            Qt.Key.Key_Print: "print_screen",
            Qt.Key.Key_Pause: "pause",
            Qt.Key.Key_CapsLock: "capslock",
            Qt.Key.Key_NumLock: "numlock",
            Qt.Key.Key_ScrollLock: "scrolllock",
            Qt.Key.Key_VolumeUp: "volume_up",
            Qt.Key.Key_VolumeDown: "volume_down",
            Qt.Key.Key_VolumeMute: "volume_mute",
            Qt.Key.Key_MediaPlay: "play_pause",
            Qt.Key.Key_MediaNext: "next_track",
            Qt.Key.Key_MediaPrevious: "prev_track",
            Qt.Key.Key_MediaStop: "stop",
        }
        
        if key in key_map:
            return key_map[key]
        
        # Phím chữ/số
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            return chr(key).lower()
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            return chr(key)
        
        # Các phím đặc biệt khác
        special = {
            Qt.Key.Key_Minus: "-",
            Qt.Key.Key_Equal: "=",
            Qt.Key.Key_BracketLeft: "[",
            Qt.Key.Key_BracketRight: "]",
            Qt.Key.Key_Semicolon: ";",
            Qt.Key.Key_Apostrophe: "'",
            Qt.Key.Key_Comma: ",",
            Qt.Key.Key_Period: ".",
            Qt.Key.Key_Slash: "/",
            Qt.Key.Key_Backslash: "\\",
            Qt.Key.Key_QuoteLeft: "`",
        }
        return special.get(key, "")
    
    def focusOutEvent(self, event):
        if self.is_recording:
            self.stop_recording()
        super().focusOutEvent(event)
    
    def get_bound_key(self):
        return self.bound_key
    
    def set_bound_key(self, key):
        self.bound_key = key
        self.update_display()


# Lazy imports - chỉ import khi cần
HandDetector = None
CommandMapper = None
OptimizedGestureRecognizer = None
PerformanceOptimizer = None


def lazy_import_detector():
    """Lazy import HandDetector"""
    global HandDetector
    if HandDetector is None:
        try:
            from src.hand_detector import HandDetector as HD
            HandDetector = HD
        except ImportError:
            from hand_detector import HandDetector as HD
            HandDetector = HD
    return HandDetector


def lazy_import_mapper():
    """Lazy import CommandMapper"""
    global CommandMapper
    if CommandMapper is None:
        try:
            from src.command_mapper import CommandMapper as CM
            CommandMapper = CM
        except ImportError:
            from command_mapper import CommandMapper as CM
            CommandMapper = CM
    return CommandMapper


def lazy_import_recognizer():
    """Lazy import OptimizedGestureRecognizer"""
    global OptimizedGestureRecognizer, PerformanceOptimizer
    if OptimizedGestureRecognizer is None:
        try:
            from src.optimized_recognizer import OptimizedGestureRecognizer as OGR
            from src.optimized_recognizer import PerformanceOptimizer as PO
            OptimizedGestureRecognizer = OGR
            PerformanceOptimizer = PO
        except ImportError:
            from optimized_recognizer import OptimizedGestureRecognizer as OGR
            from optimized_recognizer import PerformanceOptimizer as PO
            OptimizedGestureRecognizer = OGR
            PerformanceOptimizer = PO
    return OptimizedGestureRecognizer, PerformanceOptimizer


class OptimizedCameraThread(QThread):
    """Thread xử lý camera tối ưu hiệu năng"""
    frame_ready = pyqtSignal(np.ndarray)
    gesture_detected = pyqtSignal(str, str)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    performance_stats = pyqtSignal(dict)
    
    def __init__(self, config_path="gesture_config.json"):
        super().__init__()
        self.config_path = config_path
        self.running = False
        self.paused = False
        self.camera_index = 0
        self.show_landmarks = True
        self.show_fps = True
        self.show_gesture = True
        
    def load_config(self):
        """Load cấu hình từ file"""
        config_file = os.path.join(base_path, self.config_path)
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"gestures": {}, "settings": {}}
    
    def run(self):
        """Main loop tối ưu"""
        self.running = True
        self.status_update.emit("Đang khởi động...")
        
        # Load config
        config = self.load_config()
        settings = config.get("settings", {})
        
        # === PERFORMANCE OPTIMIZATION ===
        OGR, PO = lazy_import_recognizer()
        optimizer = PO(config)
        opt_settings = optimizer.get_optimized_settings()
        
        self.status_update.emit(f"Tối ưu: {len(opt_settings['enabled_gestures'])} cử chỉ")
        
        # Khởi tạo camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error_occurred.emit("Không thể mở camera!")
            return
        
        # Set camera properties - có thể giảm để tiết kiệm
        cam_width = settings.get("camera_width", 640)
        cam_height = settings.get("camera_height", 480)
        
        # Giảm resolution nếu ít gesture
        if len(opt_settings['enabled_gestures']) <= 3:
            cam_width = min(cam_width, 480)
            cam_height = min(cam_height, 360)
            self.status_update.emit(f"Giảm resolution: {cam_width}x{cam_height}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Giới hạn FPS camera
        
        # Lazy load modules
        try:
            self.status_update.emit("Đang tải Hand Detector...")
            HD = lazy_import_detector()
            detector = HD(
                max_hands=opt_settings['max_hands'],
                detection_confidence=opt_settings['detection_confidence'],
                tracking_confidence=opt_settings['tracking_confidence'],
                fist_threshold=settings.get('fist_threshold', 0.4),
                require_face=settings.get('require_face', False)
            )
            
            self.status_update.emit("Đang tải Gesture Recognizer...")
            recognizer = OGR(
                enabled_gestures=opt_settings['enabled_gestures'],
                fist_threshold=settings.get('fist_threshold', 0.4)
            )
            
            self.status_update.emit("Đang tải Command Mapper...")
            CM = lazy_import_mapper()
            mapper = CM(config.get("gestures", {}))
            
        except Exception as e:
            self.error_occurred.emit(f"Lỗi khởi tạo: {str(e)}")
            cap.release()
            return
        
        self.status_update.emit("Đang chạy (Tối ưu)")
        
        # Performance tracking
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        process_times = []
        
        # Frame skip settings
        process_every = settings.get('process_every_n_frames', 2)
        last_gesture = None
        gesture_cooldown = 0
        cooldown_frames = settings.get('gesture_cooldown', 15)
        require_looking = settings.get('require_looking', False)
        mouse_control_enabled = settings.get('mouse_control', False)
        
        # Lấy kích thước màn hình cho điều khiển chuột
        frame_w, frame_h = cam_width, cam_height
        
        while self.running:
            if self.paused:
                self.msleep(100)
                continue
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # === OPTIMIZATION: Skip frames ===
            should_process = (frame_count % process_every == 0)
            
            if should_process:
                process_start = time.time()
                
                # Detect hands
                frame, results = detector.find_hands(frame, draw=self.show_landmarks)
                
                # Get landmarks
                landmarks_list = detector.get_landmarks(frame, results)
                
                if landmarks_list and len(landmarks_list) >= 21:
                    # Giảm cooldown
                    if gesture_cooldown > 0:
                        gesture_cooldown -= 1
                    
                    # Lấy vị trí ngón trỏ (landmark 8)
                    index_finger = landmarks_list[8]  # [id, x, y, z]
                    hand_x, hand_y = index_finger[1], index_finger[2]
                    
                    # === ĐIỀU KHIỂN CHUỘT ===
                    # Kiểm tra nếu setting bật hoặc cử chỉ hiện tại có action "mouse_control"
                    if mouse_control_enabled:
                        mapper.move_mouse(hand_x, hand_y, frame_w, frame_h)
                        
                        # Vẽ điểm điều khiển
                        cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)
                        cv2.putText(frame, "MOUSE", (hand_x + 15, hand_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Check if looking at camera (if required)
                    can_execute = True
                    if require_looking:
                        is_looking = detector.is_looking_at_camera(frame)
                        can_execute = is_looking
                        if not is_looking and self.show_gesture:
                            cv2.putText(frame, "Hay nhin vao camera!", 
                                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Nhận diện cử chỉ (chỉ check gesture được bật)
                    result = recognizer.recognize_from_list(landmarks_list)
                    
                    if result and result.name != 'unknown' and can_execute:
                        # Lấy action từ config
                        gesture_data = config.get("gestures", {}).get(result.name, {})
                        action_str = gesture_data.get("action", "") if isinstance(gesture_data, dict) else ""
                        
                        # Nếu action là mouse_control, di chuyển chuột theo tay
                        if action_str == "mouse_control":
                            mapper.move_mouse(hand_x, hand_y, frame_w, frame_h)
                            cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)
                            cv2.putText(frame, f"{result.name} -> MOUSE", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif gesture_cooldown == 0:
                            # Execute command bình thường
                            action = mapper.execute_gesture(result.name)
                            if action:
                                self.gesture_detected.emit(result.name, action)
                                gesture_cooldown = cooldown_frames
                                last_gesture = result.name
                            
                            # Draw gesture
                            if self.show_gesture:
                                cv2.putText(frame, f"{result.name} ({result.confidence:.0%})", 
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        elif self.show_gesture:
                            cv2.putText(frame, f"{result.name} (cooldown)", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 2)
                    elif result and result.name != 'unknown' and self.show_gesture:
                        # Show gesture but not executed
                        color = (0, 255, 255) if not can_execute else (128, 128, 128)
                        cv2.putText(frame, f"{result.name} (waiting)", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                
                process_time = (time.time() - process_start) * 1000
                process_times.append(process_time)
                if len(process_times) > 30:
                    process_times.pop(0)
            
            # FPS calculation
            if fps_counter >= 10:
                elapsed = time.time() - fps_start_time
                current_fps = fps_counter / elapsed if elapsed > 0 else 0
                fps_counter = 0
                fps_start_time = time.time()
                
                # Send performance stats
                avg_process = sum(process_times) / len(process_times) if process_times else 0
                self.performance_stats.emit({
                    'fps': current_fps,
                    'process_time_ms': avg_process,
                    'enabled_gestures': len(opt_settings['enabled_gestures']),
                    'frame_skip': process_every
                })
            
            # Draw FPS và info
            if self.show_fps:
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Skip: {process_every} | Gestures: {len(opt_settings['enabled_gestures'])}", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Emit frame
            self.frame_ready.emit(frame)
            
            # Sleep để giảm CPU (đủ cho ~60 FPS)
            self.msleep(1)
        
        cap.release()
        detector.close()
        self.status_update.emit("Đã dừng")
    
    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """Cửa sổ chính - Phiên bản thân thiện người dùng"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤚 Hand Gesture Control")
        self.setMinimumSize(600, 400)
        self.resize(900, 600)
        
        self.camera_thread = None
        self.is_running = False
        self.is_compact_mode = False
        self.developer_mode = False  # Chế độ nhà phát triển
        
        self.config_path = os.path.join(base_path, "gesture_config.json")
        self.config = self.load_config()
        
        # Load developer mode từ config
        self.developer_mode = self.config.get("settings", {}).get("developer_mode", False)
        
        self.setup_ui()
        self.apply_styles()
        self.update_developer_mode_ui()
    
    def resizeEvent(self, event):
        """Xử lý responsive khi resize"""
        super().resizeEvent(event)
    
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"gestures": {}, "settings": {}}
    
    def save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.log("💾 Đã lưu cấu hình")
            return True
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể lưu: {e}")
            return False
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === HEADER BAR ===
        header_bar = QFrame()
        header_bar.setObjectName("headerBar")
        header_bar.setFixedHeight(60)
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # Logo và tiêu đề
        title_label = QLabel("🤚 Hand Gesture Control")
        title_label.setObjectName("appTitle")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Status indicator
        self.status_indicator = QLabel("⚪ Chưa chạy")
        self.status_indicator.setObjectName("statusIndicator")
        header_layout.addWidget(self.status_indicator)
        
        # Developer mode toggle
        self.dev_mode_btn = QPushButton("🔧")
        self.dev_mode_btn.setObjectName("devModeBtn")
        self.dev_mode_btn.setFixedSize(40, 40)
        self.dev_mode_btn.setToolTip("Bật/Tắt chế độ nhà phát triển")
        self.dev_mode_btn.clicked.connect(self.toggle_developer_mode)
        header_layout.addWidget(self.dev_mode_btn)
        
        main_layout.addWidget(header_bar)
        
        # === CONTENT AREA ===
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # === LEFT: Camera Panel (chỉ hiện khi developer mode) ===
        self.left_panel = QFrame()
        self.left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("📷 Camera Preview")
        header.setObjectName("panelHeader")
        left_layout.addWidget(header)
        
        self.camera_label = QLabel()
        self.camera_label.setObjectName("cameraDisplay")
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Camera chưa bật")
        self.camera_label.setScaledContents(False)
        left_layout.addWidget(self.camera_label, stretch=1)
        
        # Debug info (chỉ hiện developer mode)
        self.debug_frame = QFrame()
        debug_layout = QHBoxLayout(self.debug_frame)
        debug_layout.setContentsMargins(0, 10, 0, 0)
        
        self.perf_label = QLabel("")
        self.perf_label.setObjectName("perfLabel")
        debug_layout.addWidget(self.perf_label)
        debug_layout.addStretch()
        left_layout.addWidget(self.debug_frame)
        
        content_layout.addWidget(self.left_panel, stretch=2)
        
        # === RIGHT: Main Settings Panel ===
        self.right_panel = QFrame()
        self.right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)
        
        # === SIMPLE MODE: Nút điều khiển lớn ===
        self.simple_controls = QFrame()
        simple_layout = QVBoxLayout(self.simple_controls)
        simple_layout.setSpacing(20)
        
        # Gesture status display - lớn và rõ ràng
        self.gesture_display = QLabel("✋")
        self.gesture_display.setObjectName("gestureDisplay")
        self.gesture_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_display.setFont(QFont("Segoe UI Emoji", 72))
        simple_layout.addWidget(self.gesture_display)
        
        self.gesture_name_label = QLabel("Sẵn sàng nhận diện cử chỉ")
        self.gesture_name_label.setObjectName("gestureNameLabel")
        self.gesture_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_name_label.setFont(QFont("Segoe UI", 14))
        simple_layout.addWidget(self.gesture_name_label)
        
        self.action_label = QLabel("")
        self.action_label.setObjectName("actionLabel")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        simple_layout.addWidget(self.action_label)
        
        simple_layout.addStretch()
        
        # Big Start/Stop Button
        self.start_btn = QPushButton("▶  Bắt Đầu")
        self.start_btn.setObjectName("bigStartBtn")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.start_btn.clicked.connect(self.toggle_camera)
        simple_layout.addWidget(self.start_btn)
        
        # Pause button
        self.pause_btn = QPushButton("⏸  Tạm Dừng")
        self.pause_btn.setObjectName("pauseBtn")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        simple_layout.addWidget(self.pause_btn)
        
        right_layout.addWidget(self.simple_controls)
        
        # === TABS: Cấu hình (đơn giản hơn) ===
        self.tabs = QTabWidget()
        self.tabs.setObjectName("settingsTabs")
        self.tabs.addTab(self.create_gesture_tab(), "🤚 Cử Chỉ")
        self.tabs.addTab(self.create_simple_settings_tab(), "⚙️ Cài Đặt")
        
        # Tab developer (chỉ hiện khi bật developer mode)
        self.dev_performance_tab = self.create_performance_tab()
        self.dev_settings_tab = self.create_advanced_settings_tab()
        self.dev_log_tab = self.create_log_tab()
        
        right_layout.addWidget(self.tabs)
        
        # Save button
        save_btn = QPushButton("💾 Lưu Cấu Hình")
        save_btn.setObjectName("saveBtn")
        save_btn.clicked.connect(self.save_and_apply)
        right_layout.addWidget(save_btn)
        
        content_layout.addWidget(self.right_panel, stretch=1)
        
        main_layout.addWidget(content_widget, stretch=1)
    
    def toggle_developer_mode(self):
        """Bật/tắt chế độ nhà phát triển"""
        self.developer_mode = not self.developer_mode
        self.config.setdefault("settings", {})["developer_mode"] = self.developer_mode
        self.update_developer_mode_ui()
        
        if self.developer_mode:
            self.log("🔧 Đã bật chế độ nhà phát triển")
        else:
            self.log("🔧 Đã tắt chế độ nhà phát triển")
    
    def update_developer_mode_ui(self):
        """Cập nhật UI theo chế độ developer"""
        if self.developer_mode:
            # Hiện camera panel và các tab debug
            self.left_panel.show()
            self.dev_mode_btn.setStyleSheet("""
                QPushButton {
                    background: #22c55e;
                    border: none;
                    border-radius: 8px;
                    color: white;
                    font-size: 18px;
                }
            """)
            
            # Thêm các tab developer
            if self.tabs.indexOf(self.dev_performance_tab) == -1:
                self.tabs.addTab(self.dev_performance_tab, "⚡ Hiệu Năng")
                self.tabs.addTab(self.dev_settings_tab, "🔧 Nâng Cao")
                self.tabs.addTab(self.dev_log_tab, "📝 Log")
            
            # Hiện debug info
            self.debug_frame.show()
            self.perf_label.show()
            
            # Update window title
            self.setWindowTitle("🤚 Hand Gesture Control [Developer Mode]")
            self.resize(1200, 700)
        else:
            # Ẩn camera panel
            self.left_panel.hide()
            self.dev_mode_btn.setStyleSheet("""
                QPushButton {
                    background: #3a3a5a;
                    border: none;
                    border-radius: 8px;
                    color: #888;
                    font-size: 18px;
                }
                QPushButton:hover {
                    background: #4a4a6a;
                    color: white;
                }
            """)
            
            # Xóa các tab developer
            for i in range(self.tabs.count() - 1, 1, -1):
                self.tabs.removeTab(i)
            
            # Ẩn debug info
            self.debug_frame.hide()
            
            # Update window title
            self.setWindowTitle("🤚 Hand Gesture Control")
            self.resize(500, 600)
    
    def create_simple_settings_tab(self):
        """Tab cài đặt đơn giản cho người dùng thường"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        settings = self.config.get("settings", {})
        
        # Độ nhạy - đơn giản
        sens_group = QGroupBox("🎯 Độ Nhạy Nhận Diện")
        sens_layout = QVBoxLayout(sens_group)
        
        sens_info = QLabel("Điều chỉnh độ chính xác khi nhận diện cử chỉ tay")
        sens_info.setWordWrap(True)
        sens_info.setStyleSheet("color: #888; font-size: 11px;")
        sens_layout.addWidget(sens_info)
        
        self.detection_slider = QSlider(Qt.Orientation.Horizontal)
        self.detection_slider.setRange(30, 100)
        self.detection_slider.setValue(int(settings.get("detection_confidence", 0.7) * 100))
        self.detection_label = QLabel(f"{self.detection_slider.value()}%")
        self.detection_slider.valueChanged.connect(lambda v: self.detection_label.setText(f"{v}%"))
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Thấp"))
        slider_layout.addWidget(self.detection_slider, stretch=1)
        slider_layout.addWidget(QLabel("Cao"))
        slider_layout.addWidget(self.detection_label)
        sens_layout.addLayout(slider_layout)
        
        layout.addWidget(sens_group)
        
        # Điều khiển chuột
        mouse_group = QGroupBox("🖱️ Điều Khiển Chuột")
        mouse_layout = QVBoxLayout(mouse_group)
        
        mouse_info = QLabel("Di chuyển con trỏ chuột bằng cử chỉ tay của bạn")
        mouse_info.setWordWrap(True)
        mouse_info.setStyleSheet("color: #888; font-size: 11px;")
        mouse_layout.addWidget(mouse_info)
        
        self.mouse_control_check = QCheckBox("Bật điều khiển chuột bằng tay")
        self.mouse_control_check.setChecked(settings.get("mouse_control", False))
        mouse_layout.addWidget(self.mouse_control_check)
        
        layout.addWidget(mouse_group)
        
        # Theo dõi mắt
        eye_group = QGroupBox("👁️ Theo Dõi Mắt")
        eye_layout = QVBoxLayout(eye_group)
        
        eye_info = QLabel("Chỉ thực thi cử chỉ khi bạn đang nhìn vào camera (an toàn hơn)")
        eye_info.setWordWrap(True)
        eye_info.setStyleSheet("color: #888; font-size: 11px;")
        eye_layout.addWidget(eye_info)
        
        self.require_face_check = QCheckBox("Yêu cầu phát hiện khuôn mặt")
        self.require_face_check.setChecked(settings.get("require_face", False))
        eye_layout.addWidget(self.require_face_check)
        
        self.require_eye_check = QCheckBox("Yêu cầu đang nhìn vào camera")
        self.require_eye_check.setChecked(settings.get("require_looking", False))
        eye_layout.addWidget(self.require_eye_check)
        
        layout.addWidget(eye_group)
        
        # Khởi động cùng Windows (placeholder)
        startup_group = QGroupBox("🚀 Khởi Động")
        startup_layout = QVBoxLayout(startup_group)
        
        self.auto_start_check = QCheckBox("Tự động chạy khi khởi động Windows")
        self.auto_start_check.setChecked(settings.get("auto_start", False))
        startup_layout.addWidget(self.auto_start_check)
        
        self.minimize_to_tray_check = QCheckBox("Thu nhỏ xuống khay hệ thống khi đóng")
        self.minimize_to_tray_check.setChecked(settings.get("minimize_to_tray", False))
        startup_layout.addWidget(self.minimize_to_tray_check)
        
        layout.addWidget(startup_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def create_advanced_settings_tab(self):
        """Tab cài đặt nâng cao - chỉ hiện ở developer mode"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        settings = self.config.get("settings", {})
        
        # Fist threshold
        fist_group = QGroupBox("✊ Ngưỡng Nắm Đấm")
        fist_layout = QVBoxLayout(fist_group)
        
        self.fist_slider = QSlider(Qt.Orientation.Horizontal)
        self.fist_slider.setRange(20, 60)
        self.fist_slider.setValue(int(settings.get("fist_threshold", 0.4) * 100))
        self.fist_label = QLabel(f"{self.fist_slider.value() / 100:.2f}")
        self.fist_slider.valueChanged.connect(lambda v: self.fist_label.setText(f"{v / 100:.2f}"))
        
        fist_layout.addWidget(self.fist_slider)
        fist_layout.addWidget(self.fist_label)
        layout.addWidget(fist_group)
        
        # Display options
        disp_group = QGroupBox("🖥️ Hiển Thị Camera")
        disp_layout = QVBoxLayout(disp_group)
        
        self.show_landmarks_check = QCheckBox("Hiện khung xương tay")
        self.show_landmarks_check.setChecked(settings.get("show_landmarks", True))
        disp_layout.addWidget(self.show_landmarks_check)
        
        self.show_fps_check = QCheckBox("Hiện FPS")
        self.show_fps_check.setChecked(settings.get("show_fps", True))
        disp_layout.addWidget(self.show_fps_check)
        
        self.show_gesture_check = QCheckBox("Hiện tên cử chỉ trên camera")
        self.show_gesture_check.setChecked(settings.get("show_gesture", True))
        disp_layout.addWidget(self.show_gesture_check)
        
        layout.addWidget(disp_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def create_gesture_tab(self):
        """Tab cử chỉ với checkbox bật/tắt"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        
        # Info label
        info = QLabel("💡 Chỉ bật các cử chỉ bạn cần để tối ưu hiệu năng!")
        info.setStyleSheet("color: #fbbf24; font-size: 12px; padding: 10px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        gestures = self.config.get("gestures", {})
        
        gesture_list = [
            # === Cử chỉ tĩnh ===
            ("fist", "✊ Nắm đấm"),
            ("open_palm", "🖐️ Xòe tay"),
            ("pointing", "👆 Chỉ tay"),
            ("peace", "✌️ Hòa bình"),
            ("thumbs_up", "👍 Ngón cái lên"),
            ("thumbs_down", "👎 Ngón cái xuống"),
            ("ok", "👌 OK"),
            ("rock", "🤘 Rock"),
            ("three", "3️⃣ Số ba"),
            ("four", "4️⃣ Số bốn"),
            ("call", "🤙 Gọi điện"),
            ("loose_fist", "✊ Nắm hờ"),
            # === Cử chỉ vuốt ===
            ("swipe_up", "🖐⬆ Vuốt lên"),
            ("swipe_down", "🖐⬇ Vuốt xuống"),
            ("swipe_left", "🖐⬅ Vuốt trái"),
            ("swipe_right", "🖐➡ Vuốt phải"),
            # === Cử chỉ khác ===
            ("pinch", "🤏 Nhéo/Kẹp"),
            ("wave", "👋 Vẫy tay"),
            ("zoom_in", "🔍+ Phóng to"),
            ("zoom_out", "🔍- Thu nhỏ"),
        ]
        
        self.gesture_inputs = {}
        self.gesture_checks = {}
        
        for gesture_id, gesture_name in gesture_list:
            frame = QFrame()
            frame.setObjectName("gestureCard")
            frame_layout = QHBoxLayout(frame)
            frame_layout.setContentsMargins(8, 4, 8, 4)
            frame_layout.setSpacing(8)
            
            # Checkbox bật/tắt
            check = QCheckBox()
            current = gestures.get(gesture_id, {})
            is_enabled = current.get('enabled', False) if isinstance(current, dict) else bool(current)
            check.setChecked(is_enabled)
            check.setFixedWidth(25)
            frame_layout.addWidget(check)
            self.gesture_checks[gesture_id] = check
            
            # Label - responsive width
            label = QLabel(gesture_name)
            label.setMinimumWidth(80)
            label.setMaximumWidth(150)
            frame_layout.addWidget(label, stretch=1)
            
            # Key bind button - bấm để gán phím
            current_action = ""
            if isinstance(current, dict):
                current_action = current.get("action", "")
            elif current:
                current_action = str(current)
            
            key_btn = KeyBindButton(current_action)
            key_btn.setMinimumWidth(120)
            key_btn.setMaximumWidth(200)
            key_btn.setToolTip("Click vào rồi bấm phím/tổ hợp phím bạn muốn gán")
            frame_layout.addWidget(key_btn, stretch=2)
            self.gesture_inputs[gesture_id] = key_btn
            
            # Dropdown cho hành động đặc biệt (chuột, media...)
            special_combo = QComboBox()
            special_combo.setFixedWidth(45)
            special_combo.setToolTip("Chọn hành động đặc biệt")
            special_combo.setStyleSheet("""
                QComboBox {
                    background: #2a2a4a;
                    border: 2px solid #6366f1;
                    border-radius: 4px;
                    padding: 4px;
                    color: white;
                }
                QComboBox:hover {
                    background: #3a3a5a;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 20px;
                }
                QComboBox QAbstractItemView {
                    background: #1a1a2e;
                    color: white;
                    selection-background-color: #4f46e5;
                }
            """)
            
            # Thêm các hành động đặc biệt
            special_actions = [
                ("", "📋"),
                # Chuột
                ("mouse_control", "🖐️ Di chuột theo tay"),
                ("click", "🖱️ Click"),
                ("right_click", "🖱️ Click phải"),
                ("double_click", "🖱️ Double click"),
                ("middle_click", "🖱️ Click giữa"),
                ("scroll_up", "⬆️ Cuộn lên"),
                ("scroll_down", "⬇️ Cuộn xuống"),
                ("scroll_left", "⬅️ Cuộn trái"),
                ("scroll_right", "➡️ Cuộn phải"),
                ("mouse_drag", "✊ Kéo thả"),
                # Di chuyển chuột
                ("mouse_up", "🖱️⬆️ Chuột lên"),
                ("mouse_down", "🖱️⬇️ Chuột xuống"),
                ("mouse_left", "🖱️⬅️ Chuột trái"),
                ("mouse_right", "🖱️➡️ Chuột phải"),
                ("mouse_center", "🎯 Chuột về giữa"),
                # Media
                ("volume_up", "🔊 Vol+"),
                ("volume_down", "🔉 Vol-"),
                ("volume_mute", "🔇 Mute"),
                ("play_pause", "⏯️ Play/Pause"),
                ("next_track", "⏭️ Next"),
                ("prev_track", "⏮️ Prev"),
                ("stop", "⏹️ Stop"),
                # Độ sáng
                ("brightness_up", "🔆 Sáng+"),
                ("brightness_down", "🔅 Sáng-"),
                # Phím tắt phổ biến
                ("print_screen", "📸 Chụp màn hình"),
                ("alt+tab", "🔄 Alt+Tab"),
                ("alt+f4", "❌ Đóng cửa sổ"),
                ("win+d", "🖥️ Desktop"),
                ("win+e", "📁 Explorer"),
                ("win+l", "🔒 Khóa máy"),
                ("ctrl+c", "📋 Copy"),
                ("ctrl+v", "📋 Paste"),
                ("ctrl+z", "↩️ Undo"),
                ("ctrl+s", "💾 Save"),
            ]
            for key, name in special_actions:
                special_combo.addItem(name, key)
            
            # Khi chọn action đặc biệt, cập nhật vào key_btn
            def on_special_selected(index, btn=key_btn, combo=special_combo):
                action = combo.itemData(index)
                if action:
                    btn.set_bound_key(action)
                    combo.setCurrentIndex(0)  # Reset về icon
            
            special_combo.currentIndexChanged.connect(on_special_selected)
            frame_layout.addWidget(special_combo)
            
            # Nút xóa - nhỏ gọn
            clear_btn = QPushButton("✕")
            clear_btn.setFixedSize(28, 28)
            clear_btn.setToolTip("Xóa phím đã gán")
            clear_btn.setStyleSheet("""
                QPushButton {
                    background: #4a1a1a;
                    border: none;
                    border-radius: 4px;
                    color: #ff6b6b;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #6a2a2a;
                }
            """)
            clear_btn.clicked.connect(lambda checked, btn=key_btn: btn.set_bound_key(""))
            frame_layout.addWidget(clear_btn)
            
            layout.addWidget(frame)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def create_performance_tab(self):
        """Tab cài đặt hiệu năng - Developer mode"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        settings = self.config.get("settings", {})
        
        # Frame skip
        skip_group = QGroupBox("🔄 Xử Lý Frame")
        skip_layout = QVBoxLayout(skip_group)
        
        skip_info = QLabel("Bỏ qua frame để giảm CPU. Số càng cao = tiết kiệm hơn nhưng chậm phản hồi.")
        skip_info.setWordWrap(True)
        skip_info.setStyleSheet("color: #888; font-size: 11px;")
        skip_layout.addWidget(skip_info)
        
        skip_h = QHBoxLayout()
        skip_h.addWidget(QLabel("Xử lý mỗi:"))
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 5)
        self.frame_skip_spin.setValue(settings.get('process_every_n_frames', 2))
        skip_h.addWidget(self.frame_skip_spin)
        skip_h.addWidget(QLabel("frame"))
        skip_h.addStretch()
        skip_layout.addLayout(skip_h)
        
        layout.addWidget(skip_group)
        
        # Cooldown
        cool_group = QGroupBox("⏱️ Cooldown Cử Chỉ")
        cool_layout = QVBoxLayout(cool_group)
        
        cool_info = QLabel("Thời gian chờ giữa 2 lần thực thi cùng cử chỉ (tránh spam).")
        cool_info.setWordWrap(True)
        cool_info.setStyleSheet("color: #888; font-size: 11px;")
        cool_layout.addWidget(cool_info)
        
        cool_h = QHBoxLayout()
        cool_h.addWidget(QLabel("Cooldown:"))
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(5, 60)
        self.cooldown_spin.setValue(settings.get('gesture_cooldown', 15))
        cool_h.addWidget(self.cooldown_spin)
        cool_h.addWidget(QLabel("frames"))
        cool_h.addStretch()
        cool_layout.addLayout(cool_h)
        
        layout.addWidget(cool_group)
        
        # Low performance mode
        low_group = QGroupBox("🔋 Chế Độ Tiết Kiệm")
        low_layout = QVBoxLayout(low_group)
        
        self.low_perf_check = QCheckBox("Bật chế độ tiết kiệm (giảm resolution, tăng skip)")
        self.low_perf_check.setChecked(settings.get('low_performance_mode', False))
        low_layout.addWidget(self.low_perf_check)
        
        layout.addWidget(low_group)
        
        # Stats display
        stats_group = QGroupBox("📊 Thống Kê Hiệu Năng")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Chưa có dữ liệu. Bấm 'Bắt Đầu' để xem.")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def create_gesture_tab(self):
        """Tab cử chỉ - danh sách cử chỉ và phím tắt"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        
        # Info label
        info = QLabel("💡 Bật các cử chỉ bạn muốn sử dụng và gán phím tắt cho chúng")
        info.setStyleSheet("color: #fbbf24; font-size: 12px; padding: 10px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        gestures = self.config.get("gestures", {})
        
        gesture_list = [
            ("fist", "✊ Nắm đấm"),
            ("open_palm", "🖐️ Xòe tay"),
            ("pointing", "👆 Chỉ tay"),
            ("peace", "✌️ Hòa bình"),
            ("thumbs_up", "👍 Ngón cái lên"),
            ("thumbs_down", "👎 Ngón cái xuống"),
            ("ok", "👌 OK"),
            ("rock", "🤘 Rock"),
            ("three", "3️⃣ Số ba"),
            ("four", "4️⃣ Số bốn"),
            ("call", "🤙 Gọi điện"),
            ("swipe_up", "⬆️ Vuốt lên"),
            ("swipe_down", "⬇️ Vuốt xuống"),
            ("swipe_left", "⬅️ Vuốt trái"),
            ("swipe_right", "➡️ Vuốt phải"),
            ("pinch", "🤏 Nhéo"),
        ]
        
        self.gesture_inputs = {}
        self.gesture_checks = {}
        
        for gesture_id, gesture_name in gesture_list:
            frame = QFrame()
            frame.setObjectName("gestureCard")
            frame_layout = QHBoxLayout(frame)
            frame_layout.setContentsMargins(10, 6, 10, 6)
            frame_layout.setSpacing(10)
            
            # Checkbox bật/tắt
            check = QCheckBox()
            current = gestures.get(gesture_id, {})
            is_enabled = current.get('enabled', False) if isinstance(current, dict) else bool(current)
            check.setChecked(is_enabled)
            check.setFixedWidth(25)
            frame_layout.addWidget(check)
            self.gesture_checks[gesture_id] = check
            
            # Label
            label = QLabel(gesture_name)
            label.setMinimumWidth(100)
            frame_layout.addWidget(label, stretch=1)
            
            # Key bind button
            current_action = ""
            if isinstance(current, dict):
                current_action = current.get("action", "")
            elif current:
                current_action = str(current)
            
            key_btn = KeyBindButton(current_action)
            key_btn.setMinimumWidth(120)
            key_btn.setToolTip("Click để gán phím")
            frame_layout.addWidget(key_btn, stretch=2)
            self.gesture_inputs[gesture_id] = key_btn
            
            # Quick actions dropdown
            quick_combo = QComboBox()
            quick_combo.setFixedWidth(40)
            quick_combo.setToolTip("Hành động nhanh")
            
            quick_actions = [
                ("", "⚡"),
                ("click", "🖱️ Click"),
                ("right_click", "🖱️ Click phải"),
                ("volume_up", "🔊 Tăng âm"),
                ("volume_down", "🔉 Giảm âm"),
                ("play_pause", "⏯️ Play/Pause"),
                ("alt+tab", "🔄 Alt+Tab"),
                ("win+d", "🖥️ Desktop"),
                ("ctrl+c", "📋 Copy"),
                ("ctrl+v", "📋 Paste"),
            ]
            for key, name in quick_actions:
                quick_combo.addItem(name, key)
            
            def on_quick_selected(index, btn=key_btn, combo=quick_combo):
                action = combo.itemData(index)
                if action:
                    btn.set_bound_key(action)
                    combo.setCurrentIndex(0)
            
            quick_combo.currentIndexChanged.connect(on_quick_selected)
            frame_layout.addWidget(quick_combo)
            
            # Clear button
            clear_btn = QPushButton("✕")
            clear_btn.setFixedSize(24, 24)
            clear_btn.setToolTip("Xóa")
            clear_btn.setStyleSheet("""
                QPushButton { background: transparent; border: none; color: #ff6b6b; font-weight: bold; }
                QPushButton:hover { color: #ff4444; }
            """)
            clear_btn.clicked.connect(lambda checked, btn=key_btn: btn.set_bound_key(""))
            frame_layout.addWidget(clear_btn)
            
            layout.addWidget(frame)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def create_log_tab(self):
        """Tab log - chỉ developer mode"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("logText")
        layout.addWidget(self.log_text)
        
        clear_btn = QPushButton("🗑️ Xóa Log")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(clear_btn)
        
        return widget
    
    def log(self, message):
        if hasattr(self, 'log_text'):
            self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def apply_styles(self):
        self.setStyleSheet("""
            * { background-color: transparent; }
            QMainWindow { background: #0f0f1a; }
            QWidget { background: transparent; color: white; }
            
            /* Header */
            #headerBar { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a1a2e, stop:1 #16213e);
                border-bottom: 1px solid #333;
            }
            #appTitle { color: white; }
            #statusIndicator { color: #888; font-size: 13px; margin-right: 15px; }
            #devModeBtn { background: #3a3a5a; border: none; border-radius: 8px; }
            
            /* Panels */
            #leftPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #16213e, stop:1 #1a1a2e);
                border-right: 1px solid #333;
            }
            #rightPanel { background: #0f0f1a; }
            #panelHeader { color: #fff; font-size: 16px; font-weight: bold; padding: 8px; }
            
            /* Camera display */
            #cameraDisplay {
                background: #000; border: 2px solid #333; border-radius: 10px;
                color: #555; font-size: 14px;
            }
            
            /* Gesture display */
            #gestureDisplay { color: white; background: transparent; }
            #gestureNameLabel { color: #aaa; }
            #actionLabel { color: #4ade80; font-size: 13px; }
            
            /* Buttons */
            QPushButton {
                background: #3730a3; color: white; border: none;
                border-radius: 10px; padding: 12px 20px;
                font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background: #4f46e5; }
            QPushButton:disabled { background: #333; color: #666; }
            
            #bigStartBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #22c55e, stop:1 #16a34a);
                border-radius: 15px; font-size: 18px;
            }
            #bigStartBtn:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ade80, stop:1 #22c55e); }
            
            #pauseBtn { background: #4a4a6a; }
            #pauseBtn:hover { background: #5a5a7a; }
            
            #saveBtn { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8b5cf6, stop:1 #7c3aed); }
            
            /* Tabs */
            QTabWidget { background: transparent; }
            QTabWidget::pane { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; }
            QTabBar::tab { 
                background: #252545; color: #888; 
                padding: 10px 20px; border-radius: 8px 8px 0 0; 
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #4f46e5; color: white; }
            QTabBar::tab:hover { background: #3a3a5a; color: white; }
            
            /* Groups */
            QGroupBox { 
                color: white; font-weight: bold; 
                border: 1px solid #333; border-radius: 10px; 
                margin-top: 15px; padding: 15px; padding-top: 25px;
                background: #1a1a2e;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; left: 15px; 
                padding: 0 8px; color: #aaa;
            }
            
            /* Gesture cards */
            #gestureCard { 
                background: #1e1e3a; border: 1px solid #333; 
                border-radius: 8px; 
            }
            #gestureCard:hover { border-color: #4f46e5; background: #252550; }
            
            /* Inputs */
            QLineEdit, QSpinBox {
                background: #252545; border: 1px solid #444;
                border-radius: 8px; padding: 10px; color: white;
            }
            QLineEdit:focus, QSpinBox:focus { border-color: #4f46e5; }
            
            QComboBox {
                background: #252545; border: 1px solid #444;
                border-radius: 6px; padding: 6px 10px; color: white;
            }
            QComboBox:hover { border-color: #4f46e5; }
            QComboBox::drop-down { border: none; width: 25px; }
            QComboBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
            }
            QComboBox QAbstractItemView {
                background: #1a1a2e; border: 1px solid #444;
                color: white; selection-background-color: #4f46e5;
            }
            
            /* Slider */
            QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4f46e5; width: 18px; margin: -6px 0; border-radius: 9px; }
            QSlider::sub-page:horizontal { background: #4f46e5; border-radius: 3px; }
            
            /* Checkbox */
            QCheckBox { color: white; spacing: 10px; }
            QCheckBox::indicator { 
                width: 22px; height: 22px; border-radius: 6px; 
                border: 2px solid #444; background: #252545;
            }
            QCheckBox::indicator:checked { background: #4f46e5; border-color: #4f46e5; }
            QCheckBox::indicator:hover { border-color: #666; }
            
            /* Log */
            #logText { 
                background: #0a0a15; color: #4ade80; 
                border: 1px solid #333; border-radius: 8px; 
                font-family: 'Consolas', monospace; font-size: 11px;
                padding: 10px;
            }
            
            /* Scroll */
            QScrollArea { border: none; }
            QScrollBar:vertical { background: #1a1a2e; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: #555; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            
            /* Performance label */
            #perfLabel { color: #60a5fa; font-size: 11px; }
        """)
    
    def toggle_camera(self):
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        self.update_config_from_ui()
        self.save_config()
        
        # Count enabled gestures
        enabled = sum(1 for g, c in self.gesture_checks.items() if c.isChecked())
        self.log(f"⚡ Khởi động với {enabled} cử chỉ được bật")
        
        self.camera_thread = OptimizedCameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.gesture_detected.connect(self.on_gesture_detected)
        self.camera_thread.status_update.connect(self.on_status_update)
        self.camera_thread.error_occurred.connect(self.on_error)
        self.camera_thread.performance_stats.connect(self.on_performance_stats)
        
        # Developer mode settings
        if self.developer_mode:
            self.camera_thread.show_landmarks = self.show_landmarks_check.isChecked()
            self.camera_thread.show_fps = self.show_fps_check.isChecked()
            self.camera_thread.show_gesture = self.show_gesture_check.isChecked()
        else:
            self.camera_thread.show_landmarks = False
            self.camera_thread.show_fps = False
            self.camera_thread.show_gesture = True
        
        self.camera_thread.start()
        
        self.is_running = True
        self.start_btn.setText("⏹  Dừng")
        self.start_btn.setStyleSheet("""
            #bigStartBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ef4444, stop:1 #dc2626);
            }
        """)
        self.pause_btn.setEnabled(True)
        self.status_indicator.setText("🟢 Đang chạy")
        self.status_indicator.setStyleSheet("color: #4ade80;")
    
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.is_running = False
        self.start_btn.setText("▶  Bắt Đầu")
        self.start_btn.setStyleSheet("")
        self.pause_btn.setEnabled(False)
        self.camera_label.setText("Camera đã dừng")
        self.status_indicator.setText("⚪ Đã dừng")
        self.status_indicator.setStyleSheet("color: #888;")
        self.gesture_display.setText("✋")
        self.gesture_name_label.setText("Sẵn sàng nhận diện cử chỉ")
        self.action_label.setText("")
        self.log("⏹ Đã dừng")
    
    def toggle_pause(self):
        if self.camera_thread:
            self.camera_thread.paused = not self.camera_thread.paused
            if self.camera_thread.paused:
                self.pause_btn.setText("▶  Tiếp Tục")
                self.status_indicator.setText("🟡 Tạm dừng")
                self.status_indicator.setStyleSheet("color: #fbbf24;")
            else:
                self.pause_btn.setText("⏸  Tạm Dừng")
                self.status_indicator.setText("🟢 Đang chạy")
                self.status_indicator.setStyleSheet("color: #4ade80;")
    
    def update_frame(self, frame):
        if not self.developer_mode:
            return  # Không hiện camera khi không ở developer mode
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        
        label_size = self.camera_label.size()
        scaled = QPixmap.fromImage(qimg).scaled(
            label_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)
    
    def on_gesture_detected(self, gesture, action):
        # Cập nhật UI thân thiện
        gesture_icons = {
            'fist': '✊', 'open_palm': '🖐️', 'pointing': '👆',
            'peace': '✌️', 'thumbs_up': '👍', 'thumbs_down': '👎',
            'ok': '👌', 'rock': '🤘', 'three': '3️⃣', 'four': '4️⃣',
            'call': '🤙', 'pinch': '🤏', 'swipe_up': '⬆️',
            'swipe_down': '⬇️', 'swipe_left': '⬅️', 'swipe_right': '➡️'
        }
        
        icon = gesture_icons.get(gesture, '🤚')
        self.gesture_display.setText(icon)
        self.gesture_name_label.setText(gesture.replace('_', ' ').title())
        self.action_label.setText(f"→ {action}")
        
        self.log(f"🤚 {gesture} → {action}")
        
        # Reset sau 2 giây
        QTimer.singleShot(2000, lambda: self.reset_gesture_display())
    
    def reset_gesture_display(self):
        if self.is_running:
            self.gesture_display.setText("👀")
            self.gesture_name_label.setText("Đang theo dõi...")
            self.action_label.setText("")
    
    def on_status_update(self, status):
        self.log(f"ℹ️ {status}")
    
    def on_error(self, error):
        self.log(f"❌ {error}")
        QMessageBox.critical(self, "Lỗi", error)
        self.stop_camera()
    
    def on_performance_stats(self, stats):
        if self.developer_mode:
            self.perf_label.setText(f"⚡ {stats['process_time_ms']:.1f}ms | FPS: {stats['fps']:.0f}")
            if hasattr(self, 'stats_label'):
                self.stats_label.setText(
                    f"FPS: {stats['fps']:.1f}\n"
                    f"Thời gian xử lý: {stats['process_time_ms']:.1f}ms\n"
                    f"Cử chỉ đang bật: {stats['enabled_gestures']}\n"
                    f"Frame skip: {stats['frame_skip']}"
                )
    
    def update_config_from_ui(self):
        # Gestures
        gestures = {}
        for gesture_id, check in self.gesture_checks.items():
            key_btn = self.gesture_inputs[gesture_id]
            action = key_btn.get_bound_key()
            gestures[gesture_id] = {
                "action": action if action else "",
                "enabled": check.isChecked()
            }
        self.config["gestures"] = gestures
        
        # Settings
        settings = self.config.get("settings", {})
        settings.update({
            "detection_confidence": self.detection_slider.value() / 100,
            "mouse_control": self.mouse_control_check.isChecked(),
            "auto_start": self.auto_start_check.isChecked(),
            "minimize_to_tray": self.minimize_to_tray_check.isChecked(),
            "developer_mode": self.developer_mode,
            "max_hands": 1,
            "tracking_confidence": 0.5,
            "camera_width": 640,
            "camera_height": 480,
        })
        
        # Developer mode settings
        if self.developer_mode and hasattr(self, 'frame_skip_spin'):
            settings.update({
                "process_every_n_frames": self.frame_skip_spin.value(),
                "gesture_cooldown": self.cooldown_spin.value(),
                "low_performance_mode": self.low_perf_check.isChecked(),
                "show_landmarks": self.show_landmarks_check.isChecked(),
                "show_fps": self.show_fps_check.isChecked(),
                "show_gesture": self.show_gesture_check.isChecked(),
                "fist_threshold": self.fist_slider.value() / 100,
                "require_face": self.require_face_check.isChecked(),
                "require_looking": self.require_eye_check.isChecked(),
            })
        
        self.config["settings"] = settings
    
    def save_and_apply(self):
        self.update_config_from_ui()
        if self.save_config():
            enabled = sum(1 for g, c in self.gesture_checks.items() if c.isChecked())
            self.log(f"✅ Đã lưu! {enabled} cử chỉ được bật")
            
            if self.is_running:
                self.log("🔄 Khởi động lại để áp dụng...")
                self.stop_camera()
                QTimer.singleShot(500, self.start_camera)
    
    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
