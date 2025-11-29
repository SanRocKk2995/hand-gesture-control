
"""
Hand Gesture Control - Optimized Application
Phiên bản tối ưu hiệu năng:
- Chỉ nhận diện cử chỉ được bật trong config
- Skip frame để giảm CPU
- Cache kết quả
- Lazy loading
- Giảm resolution khi cần
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
    QGroupBox, QTextEdit, QSpinBox, QComboBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QKeySequence

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
    """Cửa sổ chính - Phiên bản tối ưu"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Control (Optimized)")
        self.setMinimumSize(800, 500)  # Responsive - cho phép nhỏ hơn
        self.resize(1200, 700)  # Kích thước mặc định
        
        self.camera_thread = None
        self.is_running = False
        self.is_compact_mode = False  # Responsive mode
        
        self.config_path = os.path.join(base_path, "gesture_config.json")
        self.config = self.load_config()
        
        self.setup_ui()
        self.apply_styles()
    
    def resizeEvent(self, event):
        """Xử lý responsive khi resize"""
        super().resizeEvent(event)
        width = event.size().width()
        
        # Compact mode khi cửa sổ nhỏ
        compact = width < 1000
        
        if compact != self.is_compact_mode:
            self.is_compact_mode = compact
            self.update_responsive_layout()
        
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
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === LEFT: Camera ===
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("📷 Camera (Optimized)")
        header.setObjectName("panelHeader")
        left_layout.addWidget(header)
        
        self.camera_label = QLabel()
        self.camera_label.setObjectName("cameraDisplay")
        self.camera_label.setMinimumSize(320, 240)  # Responsive min size
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Camera chưa bật\n\nBấm 'Bắt Đầu' để chạy")
        self.camera_label.setScaledContents(False)  # Giữ tỷ lệ
        left_layout.addWidget(self.camera_label, stretch=1)
        
        # Status
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 10, 0, 0)
        
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)
        
        self.gesture_label = QLabel("")
        self.gesture_label.setObjectName("gestureLabel")
        status_layout.addWidget(self.gesture_label)
        
        self.perf_label = QLabel("")
        self.perf_label.setObjectName("perfLabel")
        status_layout.addWidget(self.perf_label)
        
        status_layout.addStretch()
        left_layout.addWidget(status_frame)
        
        # Buttons
        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setSpacing(15)
        
        self.start_btn = QPushButton("▶ Bắt Đầu")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.toggle_camera)
        btn_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ Tạm Dừng")
        self.pause_btn.setObjectName("pauseBtn")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.pause_btn)
        
        left_layout.addWidget(btn_frame)
        self.left_panel = left_panel  # Lưu reference cho responsive
        main_layout.addWidget(left_panel, stretch=2)
        
        # === RIGHT: Settings ===
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        self.right_panel = right_panel  # Lưu reference cho responsive
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        header2 = QLabel("⚙️ Cấu Hình & Tối Ưu")
        header2.setObjectName("panelHeader")
        right_layout.addWidget(header2)
        
        tabs = QTabWidget()
        tabs.setObjectName("settingsTabs")
        tabs.addTab(self.create_gesture_tab(), "🤚 Cử Chỉ")
        tabs.addTab(self.create_performance_tab(), "⚡ Hiệu Năng")
        tabs.addTab(self.create_settings_tab(), "🔧 Cài Đặt")
        tabs.addTab(self.create_log_tab(), "📝 Log")
        
        right_layout.addWidget(tabs)
        
        save_btn = QPushButton("💾 Lưu Cấu Hình")
        save_btn.setObjectName("saveBtn")
        save_btn.clicked.connect(self.save_and_apply)
        right_layout.addWidget(save_btn)
        
        main_layout.addWidget(right_panel, stretch=1)
    
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
        """Tab cài đặt hiệu năng"""
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
    
    def create_settings_tab(self):
        """Tab cài đặt chung"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        settings = self.config.get("settings", {})
        
        # Detection
        det_group = QGroupBox("🎯 Độ Nhạy")
        det_layout = QVBoxLayout(det_group)
        
        self.detection_slider = QSlider(Qt.Orientation.Horizontal)
        self.detection_slider.setRange(30, 100)
        self.detection_slider.setValue(int(settings.get("detection_confidence", 0.7) * 100))
        self.detection_label = QLabel(f"{self.detection_slider.value()}%")
        self.detection_slider.valueChanged.connect(lambda v: self.detection_label.setText(f"{v}%"))
        
        det_layout.addWidget(QLabel("Độ chính xác phát hiện:"))
        det_layout.addWidget(self.detection_slider)
        det_layout.addWidget(self.detection_label)
        layout.addWidget(det_group)
        
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
        
        # Display
        disp_group = QGroupBox("🖥️ Hiển Thị")
        disp_layout = QVBoxLayout(disp_group)
        
        self.show_landmarks_check = QCheckBox("Hiện khung xương tay")
        self.show_landmarks_check.setChecked(settings.get("show_landmarks", True))
        disp_layout.addWidget(self.show_landmarks_check)
        
        self.show_fps_check = QCheckBox("Hiện FPS")
        self.show_fps_check.setChecked(settings.get("show_fps", True))
        disp_layout.addWidget(self.show_fps_check)
        
        self.show_gesture_check = QCheckBox("Hiện tên cử chỉ")
        self.show_gesture_check.setChecked(settings.get("show_gesture", True))
        disp_layout.addWidget(self.show_gesture_check)
        
        layout.addWidget(disp_group)
        
        # Eye tracking / Face detection
        eye_group = QGroupBox("👁️ Theo Dõi Mắt")
        eye_layout = QVBoxLayout(eye_group)
        
        eye_info = QLabel("Chỉ thực thi cử chỉ khi người dùng đang nhìn vào camera")
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
        
        # Mouse control
        mouse_group = QGroupBox("🖱️ Điều Khiển Chuột")
        mouse_layout = QVBoxLayout(mouse_group)
        
        mouse_info = QLabel("Di chuyển chuột theo vị trí ngón trỏ của bạn")
        mouse_info.setWordWrap(True)
        mouse_info.setStyleSheet("color: #888; font-size: 11px;")
        mouse_layout.addWidget(mouse_info)
        
        self.mouse_control_check = QCheckBox("Bật điều khiển chuột bằng tay")
        self.mouse_control_check.setChecked(settings.get("mouse_control", False))
        mouse_layout.addWidget(self.mouse_control_check)
        
        layout.addWidget(mouse_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def create_log_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("logText")
        layout.addWidget(self.log_text)
        
        clear_btn = QPushButton("Xóa Log")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(clear_btn)
        
        return widget
    
    def log(self, message):
        self.log_text.append(message)
    
    def apply_styles(self):
        self.setStyleSheet("""
            * { 
                background-color: transparent;
            }
            QMainWindow { background: #1a1a2e; }
            QWidget { background: transparent; color: white; }
            #leftPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #16213e, stop:1 #1a1a2e);
                border-right: 1px solid #333;
            }
            #rightPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a1a2e, stop:1 #16213e);
            }
            #panelHeader { color: #fff; font-size: 18px; font-weight: bold; padding: 10px; }
            #cameraDisplay {
                background: #000; border: 2px solid #333; border-radius: 10px;
                color: #666; font-size: 16px;
            }
            #statusLabel { color: #4ade80; font-size: 13px; }
            #gestureLabel { color: #fbbf24; font-size: 13px; font-weight: bold; margin-left: 15px; }
            #perfLabel { color: #60a5fa; font-size: 11px; margin-left: 15px; }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4f46e5, stop:1 #3730a3);
                color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6366f1, stop:1 #4f46e5); }
            QPushButton:disabled { background: #333; color: #666; }
            #startBtn { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #22c55e, stop:1 #16a34a); }
            #pauseBtn { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f59e0b, stop:1 #d97706); }
            #saveBtn { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8b5cf6, stop:1 #7c3aed); }
            QTabWidget { background: transparent; }
            QTabWidget::pane { background: #1e1e3a; border: 1px solid #333; border-radius: 8px; }
            QTabBar::tab { background: #2a2a4a; color: #aaa; padding: 10px 15px; border-radius: 8px 8px 0 0; margin-right: 2px; }
            QTabBar::tab:selected { background: #4f46e5; color: white; }
            QGroupBox { 
                color: white; font-weight: bold; border: 1px solid #444; 
                border-radius: 8px; margin-top: 10px; padding-top: 15px;
                background: #252545;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            #gestureCard { background: #252545; border: 1px solid #444; border-radius: 8px; padding: 5px; }
            QFrame { background: transparent; }
            QLabel { color: white; background: transparent; }
            QLineEdit {
                background: #2a2a4a; border: 1px solid #555;
                border-radius: 6px; padding: 8px; color: white;
            }
            QLineEdit:focus { border-color: #4f46e5; background: #353560; }
            QLineEdit::placeholder { color: #666; }
            QSpinBox {
                background: #2a2a4a; border: 1px solid #555;
                border-radius: 6px; padding: 8px; color: white;
            }
            QSpinBox:focus { border-color: #4f46e5; }
            QComboBox {
                background: #2a2a4a; border: 1px solid #555;
                border-radius: 6px; padding: 8px; color: white;
                min-height: 20px;
            }
            QComboBox:hover { border-color: #4f46e5; }
            QComboBox::drop-down {
                border: none; width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #888;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a4a; border: 1px solid #555;
                color: white; selection-background-color: #4f46e5;
                outline: none;
            }
            QSlider::groove:horizontal { height: 8px; background: #333; border-radius: 4px; }
            QSlider::handle:horizontal { background: #4f46e5; width: 20px; margin: -6px 0; border-radius: 10px; }
            QSlider::sub-page:horizontal { background: #4f46e5; border-radius: 4px; }
            QCheckBox { color: white; spacing: 8px; background: transparent; }
            QCheckBox::indicator { 
                width: 20px; height: 20px; border-radius: 4px; 
                border: 2px solid #555; background: #2a2a4a;
            }
            QCheckBox::indicator:checked { background: #4f46e5; border-color: #4f46e5; }
            #logText { background: #111; color: #4ade80; border: 1px solid #333; border-radius: 8px; font-family: Consolas; }
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: #222; width: 10px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 5px; min-height: 30px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
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
        
        self.camera_thread.show_landmarks = self.show_landmarks_check.isChecked()
        self.camera_thread.show_fps = self.show_fps_check.isChecked()
        self.camera_thread.show_gesture = self.show_gesture_check.isChecked()
        
        self.camera_thread.start()
        
        self.is_running = True
        self.start_btn.setText("⏹ Dừng")
        self.start_btn.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ef4444, stop:1 #dc2626);")
        self.pause_btn.setEnabled(True)
    
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.is_running = False
        self.start_btn.setText("▶ Bắt Đầu")
        self.start_btn.setStyleSheet("")
        self.pause_btn.setEnabled(False)
        self.camera_label.setText("Camera đã dừng")
        self.log("⏹ Đã dừng")
    
    def toggle_pause(self):
        if self.camera_thread:
            self.camera_thread.paused = not self.camera_thread.paused
            self.pause_btn.setText("▶ Tiếp Tục" if self.camera_thread.paused else "⏸ Tạm Dừng")
    
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        
        # Responsive: scale theo kích thước label hiện tại
        label_size = self.camera_label.size()
        scaled = QPixmap.fromImage(qimg).scaled(
            label_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)
    
    def update_responsive_layout(self):
        """Cập nhật layout khi thay đổi kích thước"""
        if self.is_compact_mode:
            # Compact: Ẩn một số phần tử, thu gọn
            self.perf_label.hide()
            self.left_panel.setMinimumWidth(300)
        else:
            # Full: Hiện tất cả
            self.perf_label.show()
            self.left_panel.setMinimumWidth(400)
    
    def on_gesture_detected(self, gesture, action):
        self.gesture_label.setText(f"🤚 {gesture} → {action}")
        self.log(f"🤚 {gesture} → {action}")
        QTimer.singleShot(2000, lambda: self.gesture_label.setText(""))
    
    def on_status_update(self, status):
        self.status_label.setText(status)
        self.log(f"ℹ️ {status}")
    
    def on_error(self, error):
        self.log(f"❌ {error}")
        QMessageBox.critical(self, "Lỗi", error)
        self.stop_camera()
    
    def on_performance_stats(self, stats):
        self.perf_label.setText(f"⚡ {stats['process_time_ms']:.1f}ms | Skip:{stats['frame_skip']}")
        self.stats_label.setText(
            f"FPS: {stats['fps']:.1f}\n"
            f"Thời gian xử lý: {stats['process_time_ms']:.1f}ms\n"
            f"Cử chỉ đang bật: {stats['enabled_gestures']}\n"
            f"Frame skip: {stats['frame_skip']}"
        )
    
    def update_config_from_ui(self):
        # Gestures với enabled flag
        gestures = {}
        for gesture_id, check in self.gesture_checks.items():
            key_btn = self.gesture_inputs[gesture_id]
            action = key_btn.get_bound_key()  # Lấy key từ KeyBindButton
            gestures[gesture_id] = {
                "action": action if action else "",
                "enabled": check.isChecked()
            }
        self.config["gestures"] = gestures
        
        # Settings
        self.config["settings"] = {
            "detection_confidence": self.detection_slider.value() / 100,
            "fist_threshold": self.fist_slider.value() / 100,
            "show_landmarks": self.show_landmarks_check.isChecked(),
            "show_fps": self.show_fps_check.isChecked(),
            "show_gesture": self.show_gesture_check.isChecked(),
            "process_every_n_frames": self.frame_skip_spin.value(),
            "gesture_cooldown": self.cooldown_spin.value(),
            "low_performance_mode": self.low_perf_check.isChecked(),
            "max_hands": 1,
            "tracking_confidence": 0.5,
            "camera_width": 640,
            "camera_height": 480,
            "require_face": self.require_face_check.isChecked(),
            "require_looking": self.require_eye_check.isChecked(),
            "mouse_control": self.mouse_control_check.isChecked()
        }
    
    def save_and_apply(self):
        self.update_config_from_ui()
        self.save_config()
        
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
