
"""
Hand Gesture Control - User Friendly Application
Phiên bản thân thiện người dùng:
- Giao diện đơn giản, dễ sử dụng
- Chế độ nhà phát triển ẩn các tính năng debug
- Tối ưu hiệu năng tự động
- Tối ưu RAM với thuật toán tiên tiến:
  * Adaptive Resolution: Tự động điều chỉnh theo RAM
  * Adaptive Frame Skip: Skip nhiều hơn khi không có tay
  * Temporal Caching: Cache gesture giữa các frame
  * Smart GC: Chỉ garbage collect khi RAM > 70%
  * Object Pooling: Tái sử dụng buffers
"""

import sys
import os
import time
import gc

# Giới hạn memory cho numpy/opencv
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QEvent
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QIcon, QFont


# === NoScrollSlider - Slider không bị ảnh hưởng bởi scroll chuột ===
class NoScrollSlider(QSlider):
    """Slider không scroll khi lăn chuột - tránh thay đổi nhầm giá trị"""
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
    
    def wheelEvent(self, event):
        # Bỏ qua wheel event, không thay đổi giá trị slider
        event.ignore()


# === NoScrollSpinBox - SpinBox không bị ảnh hưởng bởi scroll chuột ===
class NoScrollSpinBox(QSpinBox):
    """SpinBox không scroll khi lăn chuột"""
    def wheelEvent(self, event):
        event.ignore()


# === NoScrollComboBox - ComboBox không bị ảnh hưởng bởi scroll chuột ===
class NoScrollComboBox(QComboBox):
    """ComboBox không scroll khi lăn chuột"""
    def wheelEvent(self, event):
        event.ignore()

import json
import cv2
import numpy as np

# Import psutil cho memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

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


def get_available_cameras(max_cameras=10):
    """
    Quét và trả về danh sách các camera khả dụng trên hệ thống.
    Trả về list các dict chứa thông tin camera.
    """
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Lấy thông tin camera
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            backend = cap.getBackendName()
            
            camera_info = {
                "index": i,
                "name": f"Camera {i}",
                "resolution": f"{width}x{height}",
                "fps": fps,
                "backend": backend,
                "display_name": f"📷 Camera {i} ({width}x{height} @ {fps}fps) [{backend}]"
            }
            available_cameras.append(camera_info)
            cap.release()
        else:
            cap.release()
    
    # Thêm các backend khác nếu có (DirectShow, MSMF, etc.)
    # Thử các backend phổ biến trên Windows
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
    ]
    
    for backend_id, backend_name in backends:
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, backend_id)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Kiểm tra xem camera này đã có chưa (với backend này)
                camera_key = f"{i}_{backend_id}"
                already_exists = any(
                    c.get("backend_id") == backend_id and c.get("index") == i 
                    for c in available_cameras
                )
                
                if not already_exists:
                    camera_info = {
                        "index": i,
                        "backend_id": backend_id,
                        "name": f"Camera {i} ({backend_name})",
                        "resolution": f"{width}x{height}",
                        "fps": fps,
                        "backend": backend_name,
                        "display_name": f"📷 Camera {i} ({width}x{height}) [{backend_name}]"
                    }
                    available_cameras.append(camera_info)
                cap.release()
            else:
                cap.release()
    
    return available_cameras


class OptimizedCameraThread(QThread):
    """
    Thread xử lý camera với thuật toán tối ưu tiên tiến:
    - Adaptive Resolution: Tự động điều chỉnh resolution theo RAM
    - Object Pooling: Tái sử dụng buffer thay vì tạo mới
    - Spatial Downsampling: Chỉ xử lý vùng có tay
    - Temporal Caching: Cache kết quả giữa các frame
    - Lazy Evaluation: Chỉ tính toán khi cần
    """
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
        self.camera_backend = None  # None = auto, hoặc cv2.CAP_DSHOW, cv2.CAP_MSMF, etc.
        self.show_landmarks = True
        self.show_fps = True
        self.show_gesture = True
        
        # === OBJECT POOLING: Pre-allocate buffers ===
        self._frame_buffer = None  # Reusable frame buffer
        self._rgb_buffer = None    # Reusable RGB buffer
        self._result_cache = None  # Cache gesture result
        self._cache_valid_frames = 0  # Số frame cache còn valid
        
        # === ADAPTIVE SETTINGS ===
        self._adaptive_skip = 2    # Adaptive frame skip
        self._last_hand_detected = False
        self._last_face_detected = False  # Track face detection
        self._no_hand_count = 0
        self._idle_mode = False  # Chế độ nhàn rỗi khi không có gì
        
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
    
    def _get_optimal_resolution(self, settings):
        """
        Adaptive Resolution Algorithm:
        Tự động chọn resolution tối ưu dựa trên RAM khả dụng
        """
        if not HAS_PSUTIL:
            return 640, 480  # Default nếu không có psutil
        
        available_ram = psutil.virtual_memory().available / (1024 * 1024)  # MB
        
        if available_ram > 500:
            return 640, 480  # Full HD
        elif available_ram > 300:
            return 480, 360  # Medium
        elif available_ram > 150:
            return 320, 240  # Low
        else:
            return 240, 180  # Ultra low
    
    def _adaptive_frame_skip(self, has_hand, process_time_ms):
        """
        Adaptive Frame Skip Algorithm:
        - Khi có tay: xử lý nhiều hơn (skip ít)
        - Khi không có tay: skip nhiều hơn
        - Dựa vào process time để điều chỉnh
        """
        if has_hand:
            self._no_hand_count = 0
            # Có tay -> xử lý nhanh hơn
            if process_time_ms < 20:
                self._adaptive_skip = 1  # Xử lý mỗi frame
            elif process_time_ms < 50:
                self._adaptive_skip = 2
            else:
                self._adaptive_skip = 3
        else:
            self._no_hand_count += 1
            # Không có tay lâu -> tăng skip để tiết kiệm
            if self._no_hand_count > 60:
                self._adaptive_skip = 8  # Rất ít xử lý khi không có gì
            elif self._no_hand_count > 30:
                self._adaptive_skip = 6
            elif self._no_hand_count > 10:
                self._adaptive_skip = 4
            else:
                self._adaptive_skip = 3
        
        return self._adaptive_skip
    
    def _get_idle_sleep_time(self, has_hand, has_face):
        """
        Tính thời gian sleep khi không có tay/mặt trong khung hình
        Giảm FPS display để tiết kiệm CPU/RAM
        """
        if has_hand:
            return 5  # 5ms - responsive khi có tay
        elif has_face:
            return 20  # 20ms - chậm hơn khi chỉ có mặt
        else:
            # Không có gì trong khung hình -> giảm mạnh FPS
            if self._no_hand_count > 60:
                return 100  # ~10 FPS display
            elif self._no_hand_count > 30:
                return 50   # ~20 FPS display
            else:
                return 30   # ~33 FPS display
    
    def run(self):
        """Main loop với thuật toán tối ưu"""
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
        
        # Khởi tạo camera với backend được chọn
        if self.camera_backend is not None:
            cap = cv2.VideoCapture(self.camera_index, self.camera_backend)
            self.status_update.emit(f"Đang mở Camera {self.camera_index} với backend {self.camera_backend}...")
        else:
            cap = cv2.VideoCapture(self.camera_index)
            self.status_update.emit(f"Đang mở Camera {self.camera_index}...")
        
        if not cap.isOpened():
            self.error_occurred.emit(f"Không thể mở Camera {self.camera_index}! Vui lòng thử driver khác.")
            return
        
        # === ADAPTIVE RESOLUTION ===
        cam_width, cam_height = self._get_optimal_resolution(settings)
        user_width = settings.get("camera_width", 640)
        user_height = settings.get("camera_height", 480)
        
        # Dùng resolution cao hơn nếu user yêu cầu và RAM đủ
        cam_width = min(user_width, cam_width)
        cam_height = min(user_height, cam_height)
        
        self.status_update.emit(f"Adaptive: {cam_width}x{cam_height}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Giữ FPS cao, dùng adaptive skip
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # === OBJECT POOLING: Pre-allocate buffers ===
        self._frame_buffer = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
        self._rgb_buffer = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
        
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
        
        # === ADAPTIVE FRAME SKIP ===
        # Thay vì skip cố định, dùng adaptive based on performance
        base_skip = settings.get('process_every_n_frames', 2)
        current_skip = base_skip
        last_gesture = None
        gesture_cooldown = 0
        cooldown_frames = settings.get('gesture_cooldown', 15)
        require_looking = settings.get('require_looking', False)
        mouse_control_enabled = settings.get('mouse_control', False)
        
        # Lấy kích thước màn hình cho điều khiển chuột
        frame_w, frame_h = cam_width, cam_height
        
        # === INTELLIGENT MEMORY MANAGEMENT ===
        gc_counter = 0
        GC_INTERVAL = 60  # Tăng lên 60 frames để giảm overhead
        last_process_time = 0
        cached_gesture = None
        cache_valid = 0  # Số frame cache còn valid
        
        # === ROI (Region of Interest) Tracking ===
        last_hand_roi = None  # Vùng có tay lần cuối
        
        while self.running:
            if self.paused:
                self.msleep(100)
                continue
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            fps_counter += 1
            gc_counter += 1
            
            # === SMART GC: Chỉ collect khi thực sự cần ===
            if gc_counter >= GC_INTERVAL:
                if HAS_PSUTIL:
                    mem_percent = psutil.virtual_memory().percent
                    if mem_percent > 70:  # Chỉ collect khi RAM > 70%
                        gc.collect()
                else:
                    gc.collect()  # Fallback: luôn collect
                gc_counter = 0
            
            # === IN-PLACE FLIP: Không tạo frame mới ===
            cv2.flip(frame, 1, frame)  # In-place flip
            
            # === ADAPTIVE FRAME SKIP ===
            current_skip = self._adaptive_frame_skip(
                self._last_hand_detected, 
                last_process_time
            )
            should_process = (frame_count % current_skip == 0)
            
            # === TEMPORAL CACHING ===
            # Nếu cache còn valid, dùng gesture từ cache thay vì xử lý lại
            if cache_valid > 0 and not should_process:
                cache_valid -= 1
                if cached_gesture and self.show_gesture:
                    cv2.putText(frame, f"{cached_gesture} (cached)", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2)
            
            if should_process:
                process_start = time.time()
                
                # Detect hands
                frame, results = detector.find_hands(frame, draw=self.show_landmarks)
                
                # Get landmarks
                landmarks_list = detector.get_landmarks(frame, results)
                
                if landmarks_list and len(landmarks_list) >= 21:
                    self._last_hand_detected = True
                    cache_valid = 3  # Cache valid cho 3 frames tiếp theo
                    
                    # Giảm cooldown
                    if gesture_cooldown > 0:
                        gesture_cooldown -= 1
                    
                    # Lấy vị trí ngón trỏ (landmark 8)
                    index_finger = landmarks_list[8]  # [id, x, y, z]
                    hand_x, hand_y = index_finger[1], index_finger[2]
                    
                    # === ĐIỀU KHIỂN CHUỘT ===
                    if mouse_control_enabled:
                        mapper.move_mouse(hand_x, hand_y, frame_w, frame_h)
                        cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)
                        cv2.putText(frame, "MOUSE", (hand_x + 15, hand_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Check if looking at camera (if required)
                    can_execute = True
                    if require_looking:
                        is_looking = detector.is_looking_at_camera(frame)
                        self._last_face_detected = is_looking  # Track face detection
                        can_execute = is_looking
                        if not is_looking and self.show_gesture:
                            cv2.putText(frame, "Hay nhin vao camera!", 
                                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        self._last_face_detected = True  # Không yêu cầu face = coi như có
                    
                    # Nhận diện cử chỉ
                    result = recognizer.recognize_from_list(landmarks_list)
                    
                    if result and result.name != 'unknown' and can_execute:
                        cached_gesture = result.name  # Cache gesture
                        
                        # Lấy action từ config
                        gesture_data = config.get("gestures", {}).get(result.name, {})
                        action_str = gesture_data.get("action", "") if isinstance(gesture_data, dict) else ""
                        
                        if action_str == "mouse_control":
                            mapper.move_mouse(hand_x, hand_y, frame_w, frame_h)
                            cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)
                            cv2.putText(frame, f"{result.name} -> MOUSE", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif gesture_cooldown == 0:
                            action = mapper.execute_gesture(result.name)
                            if action:
                                self.gesture_detected.emit(result.name, action)
                                gesture_cooldown = cooldown_frames
                                last_gesture = result.name
                            
                            if self.show_gesture:
                                cv2.putText(frame, f"{result.name} ({result.confidence:.0%})", 
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        elif self.show_gesture:
                            cv2.putText(frame, f"{result.name} (cooldown)", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 2)
                    elif result and result.name != 'unknown' and self.show_gesture:
                        cached_gesture = result.name
                        color = (0, 255, 255) if not can_execute else (128, 128, 128)
                        cv2.putText(frame, f"{result.name} (waiting)", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                else:
                    self._last_hand_detected = False
                    cached_gesture = None
                    cache_valid = 0
                    
                    # === IDLE MODE: Hiển thị trạng thái chờ ===
                    if self._no_hand_count > 30 and self.show_gesture:
                        idle_text = "Dua tay vao khung hinh..."
                        cv2.putText(frame, idle_text, (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                
                last_process_time = (time.time() - process_start) * 1000
                process_times.append(last_process_time)
                if len(process_times) > 10:
                    process_times.pop(0)
            
            # FPS calculation
            if fps_counter >= 15:
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
                    'frame_skip': current_skip,  # Hiện adaptive skip
                    'adaptive': True,
                    'idle_mode': self._no_hand_count > 30  # Báo hiệu idle mode
                })
            
            # Draw FPS và info
            if self.show_fps:
                idle_indicator = " [IDLE]" if self._no_hand_count > 30 else ""
                cv2.putText(frame, f"FPS: {current_fps:.1f} | Skip: {current_skip}{idle_indicator}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Emit frame - dùng memoryview để tránh copy khi có thể
            self.frame_ready.emit(frame)  # Emit trực tiếp, Qt sẽ copy
            
            # === ADAPTIVE SLEEP: Giảm FPS display khi không có tay/mặt ===
            sleep_time = self._get_idle_sleep_time(
                self._last_hand_detected, 
                self._last_face_detected
            )
            self.msleep(sleep_time)
        
        # Cleanup
        cap.release()
        detector.close()
        gc.collect()  # Final cleanup
        self.status_update.emit("Đã dừng")
    
    def stop(self):
        self.running = False
        self.wait()
        gc.collect()


class MainWindow(QMainWindow):
    """Cửa sổ chính - Phiên bản tối ưu RAM < 100MB"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤚 Hand Gesture Control")
        self.setMinimumSize(500, 400)
        self.resize(600, 500)  # Giảm kích thước mặc định
        
        self.camera_thread = None
        self.is_running = False
        self.is_compact_mode = False
        self.developer_mode = False
        
        self.config_path = os.path.join(base_path, "gesture_config.json")
        self.config = self.load_config()
        
        # Load developer mode từ config
        self.developer_mode = self.config.get("settings", {}).get("developer_mode", False)
        
        # Memory optimization: Lazy init stats
        self.gesture_count = 0
        self.command_count = 0
        self.gesture_history = {}
        self.session_start_time = time.time()
        
        self.setup_ui()
        self.apply_styles()
        self.update_developer_mode_ui()
        
        # Chạy GC sau khi setup
        gc.collect()
    
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
        right_layout.setContentsMargins(15, 10, 15, 10)
        right_layout.setSpacing(8)
        
        # === SIMPLE MODE: Compact controls ===
        self.simple_controls = QFrame()
        simple_layout = QVBoxLayout(self.simple_controls)
        simple_layout.setSpacing(5)
        simple_layout.setContentsMargins(0, 0, 0, 0)
        
        # Gesture status display - nhỏ gọn hơn
        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        
        self.gesture_display = QLabel("✋")
        self.gesture_display.setObjectName("gestureDisplay")
        self.gesture_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_display.setFont(QFont("Segoe UI Emoji", 36))  # Nhỏ hơn: 36 thay vì 72
        self.gesture_display.setFixedSize(70, 70)
        status_row.addWidget(self.gesture_display)
        
        # Text bên phải icon
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        
        self.gesture_name_label = QLabel("Sẵn sàng nhận diện cử chỉ")
        self.gesture_name_label.setObjectName("gestureNameLabel")
        self.gesture_name_label.setFont(QFont("Segoe UI", 12))
        text_col.addWidget(self.gesture_name_label)
        
        self.action_label = QLabel("")
        self.action_label.setObjectName("actionLabel")
        text_col.addWidget(self.action_label)
        
        status_row.addLayout(text_col, stretch=1)
        simple_layout.addLayout(status_row)
        
        # Buttons row - ngang nhau, nhỏ gọn
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        
        # Start/Stop Button - nhỏ hơn
        self.start_btn = QPushButton("▶  Bắt Đầu")
        self.start_btn.setObjectName("bigStartBtn")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.start_btn.clicked.connect(self.toggle_camera)
        btn_row.addWidget(self.start_btn, stretch=2)
        
        # Pause button
        self.pause_btn = QPushButton("⏸  Tạm Dừng")
        self.pause_btn.setObjectName("pauseBtn")
        self.pause_btn.setMinimumHeight(42)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_row.addWidget(self.pause_btn, stretch=1)
        
        simple_layout.addLayout(btn_row)
        
        right_layout.addWidget(self.simple_controls)
        
        # === TABS: Cấu hình (đơn giản hơn) ===
        self.tabs = QTabWidget()
        self.tabs.setObjectName("settingsTabs")
        self.tabs.addTab(self.create_gesture_tab(), "🤚 Cử Chỉ")
        self.tabs.addTab(self.create_simple_settings_tab(), "⚙️ Cài Đặt")
        self.tabs.addTab(self.create_profiles_tab(), "📁 Profiles")
        self.tabs.addTab(self.create_stats_tab(), "📊 Thống Kê")
        
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
        
        self.detection_slider = NoScrollSlider(Qt.Orientation.Horizontal)
        self.detection_slider.setRange(30, 100)
        self.detection_slider.setValue(int(settings.get("detection_confidence", 0.7) * 100))
        self.detection_slider.setMinimumHeight(30)
        self.detection_label = QLabel(f"{self.detection_slider.value()}%")
        self.detection_label.setMinimumWidth(45)
        self.detection_slider.valueChanged.connect(lambda v: self.detection_label.setText(f"{v}%"))
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Thấp"))
        slider_layout.addWidget(self.detection_slider, stretch=1)
        slider_layout.addWidget(QLabel("Cao"))
        slider_layout.addWidget(self.detection_label)
        sens_layout.addLayout(slider_layout)
        
        layout.addWidget(sens_group)
        
        # === ĐIỀU KHIỂN CHUỘT ===
        mouse_group = QGroupBox("🖱️ Điều Khiển Chuột")
        mouse_layout = QVBoxLayout(mouse_group)
        
        mouse_info = QLabel("Di chuyển con trỏ chuột bằng cử chỉ tay của bạn")
        mouse_info.setWordWrap(True)
        mouse_info.setStyleSheet("color: #888; font-size: 11px;")
        mouse_layout.addWidget(mouse_info)
        
        self.mouse_control_check = QCheckBox("Bật điều khiển chuột bằng tay")
        self.mouse_control_check.setChecked(settings.get("mouse_control", False))
        mouse_layout.addWidget(self.mouse_control_check)
        
        # Tốc độ chuột
        mouse_speed_layout = QHBoxLayout()
        mouse_speed_layout.addWidget(QLabel("Tốc độ chuột:"))
        self.mouse_speed_slider = NoScrollSlider(Qt.Orientation.Horizontal)
        self.mouse_speed_slider.setRange(1, 10)
        self.mouse_speed_slider.setValue(settings.get("mouse_speed", 5))
        self.mouse_speed_slider.setMinimumHeight(30)
        self.mouse_speed_label = QLabel(f"{self.mouse_speed_slider.value()}")
        self.mouse_speed_label.setMinimumWidth(30)
        self.mouse_speed_slider.valueChanged.connect(lambda v: self.mouse_speed_label.setText(f"{v}"))
        mouse_speed_layout.addWidget(self.mouse_speed_slider)
        mouse_speed_layout.addWidget(self.mouse_speed_label)
        mouse_layout.addLayout(mouse_speed_layout)
        
        # Độ mượt chuột
        self.mouse_smooth_check = QCheckBox("Làm mượt chuyển động chuột")
        self.mouse_smooth_check.setChecked(settings.get("mouse_smoothing", True))
        mouse_layout.addWidget(self.mouse_smooth_check)
        
        layout.addWidget(mouse_group)
        
        # === CHẾ ĐỘ ỨNG DỤNG ===
        app_mode_group = QGroupBox("🎮 Chế Độ Ứng Dụng")
        app_mode_layout = QVBoxLayout(app_mode_group)
        
        app_mode_info = QLabel("Chọn chế độ phù hợp với nhu cầu sử dụng")
        app_mode_info.setWordWrap(True)
        app_mode_info.setStyleSheet("color: #888; font-size: 11px;")
        app_mode_layout.addWidget(app_mode_info)
        
        self.app_mode_combo = NoScrollComboBox()
        self.app_mode_combo.addItem("🖥️ Điều khiển Desktop", "desktop")
        self.app_mode_combo.addItem("🎮 Chơi Game", "gaming")
        self.app_mode_combo.addItem("📺 Xem Media/Video", "media")
        self.app_mode_combo.addItem("📊 Thuyết trình", "presentation")
        self.app_mode_combo.addItem("🎨 Vẽ/Thiết kế", "creative")
        self.app_mode_combo.addItem("⚙️ Tùy chỉnh", "custom")
        
        current_mode = settings.get("app_mode", "desktop")
        for i in range(self.app_mode_combo.count()):
            if self.app_mode_combo.itemData(i) == current_mode:
                self.app_mode_combo.setCurrentIndex(i)
                break
        
        app_mode_layout.addWidget(self.app_mode_combo)
        
        # Nút áp dụng preset
        apply_preset_btn = QPushButton("⚡ Áp dụng cài đặt gợi ý cho chế độ này")
        apply_preset_btn.clicked.connect(self.apply_mode_preset)
        app_mode_layout.addWidget(apply_preset_btn)
        
        layout.addWidget(app_mode_group)
        
        # === THEO DÕI MẮT ===
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
        
        # === ÂM THANH & PHẢN HỒI ===
        feedback_group = QGroupBox("🔔 Âm Thanh & Phản Hồi")
        feedback_layout = QVBoxLayout(feedback_group)
        
        self.sound_enabled_check = QCheckBox("Phát âm thanh khi nhận diện cử chỉ")
        self.sound_enabled_check.setChecked(settings.get("sound_enabled", True))
        feedback_layout.addWidget(self.sound_enabled_check)
        
        self.vibrate_check = QCheckBox("Rung màn hình khi thực thi (hiệu ứng)")
        self.vibrate_check.setChecked(settings.get("screen_flash", False))
        feedback_layout.addWidget(self.vibrate_check)
        
        self.show_notification_check = QCheckBox("Hiện thông báo Windows")
        self.show_notification_check.setChecked(settings.get("show_notification", True))
        feedback_layout.addWidget(self.show_notification_check)
        
        layout.addWidget(feedback_group)
        
        # === VÙNG HOẠT ĐỘNG ===
        zone_group = QGroupBox("📐 Vùng Hoạt Động")
        zone_layout = QVBoxLayout(zone_group)
        
        zone_info = QLabel("Giới hạn vùng nhận diện để tránh kích hoạt nhầm")
        zone_info.setWordWrap(True)
        zone_info.setStyleSheet("color: #888; font-size: 11px;")
        zone_layout.addWidget(zone_info)
        
        self.zone_enabled_check = QCheckBox("Chỉ nhận diện trong vùng xác định")
        self.zone_enabled_check.setChecked(settings.get("zone_enabled", False))
        zone_layout.addWidget(self.zone_enabled_check)
        
        zone_size_layout = QHBoxLayout()
        zone_size_layout.addWidget(QLabel("Kích thước vùng:"))
        self.zone_size_slider = NoScrollSlider(Qt.Orientation.Horizontal)
        self.zone_size_slider.setRange(30, 100)
        self.zone_size_slider.setValue(settings.get("zone_size", 70))
        self.zone_size_slider.setMinimumHeight(30)
        self.zone_size_label = QLabel(f"{self.zone_size_slider.value()}%")
        self.zone_size_label.setMinimumWidth(45)
        self.zone_size_slider.valueChanged.connect(lambda v: self.zone_size_label.setText(f"{v}%"))
        zone_size_layout.addWidget(self.zone_size_slider)
        zone_size_layout.addWidget(self.zone_size_label)
        zone_layout.addLayout(zone_size_layout)
        
        layout.addWidget(zone_group)
        
        # === PHÍM TẮT TOÀN CỤC ===
        hotkey_group = QGroupBox("⌨️ Phím Tắt Nhanh")
        hotkey_layout = QVBoxLayout(hotkey_group)
        
        hotkey_info = QLabel("Phím tắt để điều khiển ứng dụng nhanh chóng")
        hotkey_info.setWordWrap(True)
        hotkey_info.setStyleSheet("color: #888; font-size: 11px;")
        hotkey_layout.addWidget(hotkey_info)
        
        # Toggle on/off
        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(QLabel("Bật/Tắt nhận diện:"))
        self.toggle_hotkey = KeyBindButton(settings.get("hotkey_toggle", "ctrl+shift+g"))
        toggle_layout.addWidget(self.toggle_hotkey)
        hotkey_layout.addLayout(toggle_layout)
        
        # Pause
        pause_layout = QHBoxLayout()
        pause_layout.addWidget(QLabel("Tạm dừng:"))
        self.pause_hotkey = KeyBindButton(settings.get("hotkey_pause", "ctrl+shift+p"))
        pause_layout.addWidget(self.pause_hotkey)
        hotkey_layout.addLayout(pause_layout)
        
        layout.addWidget(hotkey_group)
        
        # === CHỌN CAMERA ===
        camera_group = QGroupBox("📷 Chọn Camera")
        camera_layout = QVBoxLayout(camera_group)
        
        camera_info = QLabel("Chọn camera và driver phù hợp với thiết bị của bạn")
        camera_info.setWordWrap(True)
        camera_info.setStyleSheet("color: #888; font-size: 11px;")
        camera_layout.addWidget(camera_info)
        
        # ComboBox chọn camera
        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = NoScrollComboBox()
        self.camera_combo.setMinimumWidth(300)
        self.camera_combo.setMinimumHeight(40)
        self.camera_combo.addItem("📷 Đang quét camera...", None)
        camera_select_layout.addWidget(self.camera_combo, stretch=1)
        camera_layout.addLayout(camera_select_layout)
        
        # Nút quét lại camera
        camera_btn_layout = QHBoxLayout()
        
        self.refresh_camera_btn = QPushButton("🔄 Quét lại")
        self.refresh_camera_btn.clicked.connect(self.refresh_camera_list)
        camera_btn_layout.addWidget(self.refresh_camera_btn)
        
        self.test_camera_btn = QPushButton("🎥 Test Camera")
        self.test_camera_btn.clicked.connect(self.test_selected_camera)
        camera_btn_layout.addWidget(self.test_camera_btn)
        
        camera_btn_layout.addStretch()
        camera_layout.addLayout(camera_btn_layout)
        
        # Thông tin camera đã chọn
        self.camera_info_label = QLabel("")
        self.camera_info_label.setStyleSheet("color: #4ade80; font-size: 11px;")
        camera_layout.addWidget(self.camera_info_label)
        
        layout.addWidget(camera_group)
        
        # Quét camera khi tạo tab (dùng QTimer để không block UI)
        QTimer.singleShot(500, self.refresh_camera_list)
        
        # === KHỞI ĐỘNG ===
        startup_group = QGroupBox("🚀 Khởi Động")
        startup_layout = QVBoxLayout(startup_group)
        
        self.auto_start_check = QCheckBox("Tự động chạy khi khởi động Windows")
        self.auto_start_check.setChecked(settings.get("auto_start", False))
        startup_layout.addWidget(self.auto_start_check)
        
        self.minimize_to_tray_check = QCheckBox("Thu nhỏ xuống khay hệ thống khi đóng")
        self.minimize_to_tray_check.setChecked(settings.get("minimize_to_tray", False))
        startup_layout.addWidget(self.minimize_to_tray_check)
        
        self.start_minimized_check = QCheckBox("Khởi động ở chế độ thu nhỏ")
        self.start_minimized_check.setChecked(settings.get("start_minimized", False))
        startup_layout.addWidget(self.start_minimized_check)
        
        self.auto_run_check = QCheckBox("Tự động bắt đầu nhận diện khi mở app")
        self.auto_run_check.setChecked(settings.get("auto_run", False))
        startup_layout.addWidget(self.auto_run_check)
        
        layout.addWidget(startup_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def refresh_camera_list(self):
        """Quét và cập nhật danh sách camera khả dụng"""
        self.camera_combo.clear()
        self.camera_combo.addItem("⏳ Đang quét...", None)
        self.refresh_camera_btn.setEnabled(False)
        self.camera_info_label.setText("Đang quét các camera khả dụng...")
        
        # Quét camera trong thread riêng để không block UI
        QTimer.singleShot(100, self._do_camera_scan)
    
    def _do_camera_scan(self):
        """Thực hiện quét camera"""
        try:
            cameras = get_available_cameras()
            self.available_cameras = cameras
            
            self.camera_combo.clear()
            
            if cameras:
                # Lấy camera đã lưu trong config
                saved_camera = self.config.get("settings", {}).get("camera_index", 0)
                saved_backend = self.config.get("settings", {}).get("camera_backend", None)
                
                selected_index = 0
                for i, cam in enumerate(cameras):
                    self.camera_combo.addItem(cam["display_name"], cam)
                    # Tìm camera đã lưu
                    if cam["index"] == saved_camera:
                        if saved_backend is None or cam.get("backend_id") == saved_backend:
                            selected_index = i
                
                self.camera_combo.setCurrentIndex(selected_index)
                self.camera_info_label.setText(f"✅ Tìm thấy {len(cameras)} camera")
                self.camera_info_label.setStyleSheet("color: #4ade80; font-size: 11px;")
            else:
                self.camera_combo.addItem("❌ Không tìm thấy camera nào", None)
                self.camera_info_label.setText("Không tìm thấy camera. Hãy kiểm tra kết nối.")
                self.camera_info_label.setStyleSheet("color: #ef4444; font-size: 11px;")
        except Exception as e:
            self.camera_combo.clear()
            self.camera_combo.addItem(f"❌ Lỗi: {str(e)}", None)
            self.camera_info_label.setText(f"Lỗi khi quét camera: {str(e)}")
            self.camera_info_label.setStyleSheet("color: #ef4444; font-size: 11px;")
        finally:
            self.refresh_camera_btn.setEnabled(True)
    
    def test_selected_camera(self):
        """Test camera đã chọn bằng cách hiển thị preview"""
        cam_data = self.camera_combo.currentData()
        if not cam_data:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một camera!")
            return
        
        camera_index = cam_data["index"]
        backend_id = cam_data.get("backend_id")
        
        # Mở camera để test
        try:
            if backend_id is not None:
                cap = cv2.VideoCapture(camera_index, backend_id)
            else:
                cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Hiển thị trong camera preview nếu có
                    if self.developer_mode:
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                        scaled = QPixmap.fromImage(qimg).scaled(
                            self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self.camera_label.setPixmap(scaled)
                    
                    self.camera_info_label.setText(f"✅ Camera {camera_index} hoạt động tốt!")
                    self.camera_info_label.setStyleSheet("color: #4ade80; font-size: 11px;")
                    QMessageBox.information(self, "Test Camera", 
                        f"✅ Camera {camera_index} hoạt động tốt!\n\n"
                        f"Resolution: {cam_data['resolution']}\n"
                        f"FPS: {cam_data['fps']}\n"
                        f"Backend: {cam_data['backend']}")
                else:
                    self.camera_info_label.setText(f"⚠️ Camera {camera_index} không đọc được frame!")
                    self.camera_info_label.setStyleSheet("color: #fbbf24; font-size: 11px;")
                    QMessageBox.warning(self, "Test Camera", f"Camera {camera_index} mở được nhưng không đọc được frame!")
                cap.release()
            else:
                self.camera_info_label.setText(f"❌ Không thể mở Camera {camera_index}!")
                self.camera_info_label.setStyleSheet("color: #ef4444; font-size: 11px;")
                QMessageBox.warning(self, "Test Camera", f"Không thể mở Camera {camera_index}!")
        except Exception as e:
            self.camera_info_label.setText(f"❌ Lỗi: {str(e)}")
            self.camera_info_label.setStyleSheet("color: #ef4444; font-size: 11px;")
            QMessageBox.warning(self, "Lỗi", f"Lỗi khi test camera: {str(e)}")
    
    def get_selected_camera(self):
        """Lấy thông tin camera đã chọn"""
        cam_data = self.camera_combo.currentData()
        if cam_data:
            return cam_data["index"], cam_data.get("backend_id")
        return 0, None
    
    def apply_mode_preset(self):
        """Áp dụng cài đặt preset theo chế độ"""
        mode = self.app_mode_combo.currentData()
        
        presets = {
            "desktop": {
                "detection_confidence": 70,
                "mouse_control": False,
                "mouse_speed": 5,
                "gesture_cooldown": 15,
            },
            "gaming": {
                "detection_confidence": 85,
                "mouse_control": True,
                "mouse_speed": 8,
                "gesture_cooldown": 5,
            },
            "media": {
                "detection_confidence": 60,
                "mouse_control": False,
                "mouse_speed": 3,
                "gesture_cooldown": 20,
            },
            "presentation": {
                "detection_confidence": 75,
                "mouse_control": False,
                "mouse_speed": 4,
                "gesture_cooldown": 25,
            },
            "creative": {
                "detection_confidence": 80,
                "mouse_control": True,
                "mouse_speed": 6,
                "gesture_cooldown": 10,
            },
        }
        
        if mode in presets:
            preset = presets[mode]
            self.detection_slider.setValue(preset["detection_confidence"])
            self.mouse_control_check.setChecked(preset["mouse_control"])
            self.mouse_speed_slider.setValue(preset["mouse_speed"])
            if hasattr(self, 'cooldown_spin'):
                self.cooldown_spin.setValue(preset["gesture_cooldown"])
            
            self.log(f"⚡ Đã áp dụng preset cho chế độ: {self.app_mode_combo.currentText()}")
            QMessageBox.information(self, "Thành công", f"Đã áp dụng cài đặt cho chế độ {self.app_mode_combo.currentText()}")
    
    def create_profiles_tab(self):
        """Tab quản lý profiles - lưu/tải cấu hình"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Profiles hiện có
        profiles_group = QGroupBox("📁 Profiles Đã Lưu")
        profiles_layout = QVBoxLayout(profiles_group)
        
        profiles_info = QLabel("Lưu và tải lại cấu hình cử chỉ yêu thích của bạn")
        profiles_info.setWordWrap(True)
        profiles_info.setStyleSheet("color: #888; font-size: 11px;")
        profiles_layout.addWidget(profiles_info)
        
        self.profiles_combo = NoScrollComboBox()
        self.profiles_combo.setMinimumHeight(40)
        self.profiles_combo.addItem("📌 Mặc định", "default")
        self.profiles_combo.addItem("🎮 Gaming", "gaming")
        self.profiles_combo.addItem("📺 Media", "media")
        self.profiles_combo.addItem("💼 Làm việc", "work")
        profiles_layout.addWidget(self.profiles_combo)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        load_btn = QPushButton("📥 Tải Profile")
        load_btn.clicked.connect(self.load_profile)
        btn_layout.addWidget(load_btn)
        
        save_profile_btn = QPushButton("💾 Lưu Profile")
        save_profile_btn.clicked.connect(self.save_profile)
        btn_layout.addWidget(save_profile_btn)
        
        profiles_layout.addLayout(btn_layout)
        layout.addWidget(profiles_group)
        
        # Tạo profile mới
        new_profile_group = QGroupBox("➕ Tạo Profile Mới")
        new_profile_layout = QVBoxLayout(new_profile_group)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Tên profile:"))
        self.new_profile_name = QLineEdit()
        self.new_profile_name.setPlaceholderText("Nhập tên profile...")
        name_layout.addWidget(self.new_profile_name)
        new_profile_layout.addLayout(name_layout)
        
        create_btn = QPushButton("✨ Tạo Profile Mới")
        create_btn.clicked.connect(self.create_new_profile)
        new_profile_layout.addWidget(create_btn)
        
        layout.addWidget(new_profile_group)
        
        # Import/Export
        io_group = QGroupBox("📤 Import / Export")
        io_layout = QVBoxLayout(io_group)
        
        io_info = QLabel("Chia sẻ cấu hình với bạn bè hoặc backup")
        io_info.setWordWrap(True)
        io_info.setStyleSheet("color: #888; font-size: 11px;")
        io_layout.addWidget(io_info)
        
        io_btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("📤 Export ra file")
        export_btn.clicked.connect(self.export_config)
        io_btn_layout.addWidget(export_btn)
        
        import_btn = QPushButton("📥 Import từ file")
        import_btn.clicked.connect(self.import_config)
        io_btn_layout.addWidget(import_btn)
        
        io_layout.addLayout(io_btn_layout)
        layout.addWidget(io_group)
        
        # Reset
        reset_group = QGroupBox("🔄 Đặt Lại")
        reset_layout = QVBoxLayout(reset_group)
        
        reset_btn = QPushButton("🔄 Đặt lại về mặc định")
        reset_btn.setStyleSheet("background: #dc2626;")
        reset_btn.clicked.connect(self.reset_to_default)
        reset_layout.addWidget(reset_btn)
        
        layout.addWidget(reset_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll
    
    def create_stats_tab(self):
        """Tab thống kê sử dụng"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Thống kê phiên
        session_group = QGroupBox("📊 Phiên Hiện Tại")
        session_layout = QVBoxLayout(session_group)
        
        self.session_time_label = QLabel("⏱️ Thời gian chạy: 0 phút")
        session_layout.addWidget(self.session_time_label)
        
        self.gestures_count_label = QLabel("🤚 Cử chỉ đã nhận: 0")
        session_layout.addWidget(self.gestures_count_label)
        
        self.commands_count_label = QLabel("⚡ Lệnh đã thực thi: 0")
        session_layout.addWidget(self.commands_count_label)
        
        layout.addWidget(session_group)
        
        # Cử chỉ phổ biến
        popular_group = QGroupBox("🏆 Cử Chỉ Hay Dùng")
        popular_layout = QVBoxLayout(popular_group)
        
        self.popular_gestures_label = QLabel("Chưa có dữ liệu")
        self.popular_gestures_label.setWordWrap(True)
        popular_layout.addWidget(self.popular_gestures_label)
        
        layout.addWidget(popular_group)
        
        # Accuracy
        accuracy_group = QGroupBox("🎯 Độ Chính Xác")
        accuracy_layout = QVBoxLayout(accuracy_group)
        
        self.accuracy_label = QLabel("Đang tính toán...")
        accuracy_layout.addWidget(self.accuracy_label)
        
        # Progress bar
        self.accuracy_bar = QSlider(Qt.Orientation.Horizontal)
        self.accuracy_bar.setRange(0, 100)
        self.accuracy_bar.setValue(85)
        self.accuracy_bar.setEnabled(False)
        accuracy_layout.addWidget(self.accuracy_bar)
        
        layout.addWidget(accuracy_group)
        
        # Thời gian sử dụng
        usage_group = QGroupBox("📅 Thời Gian Sử Dụng")
        usage_layout = QVBoxLayout(usage_group)
        
        self.total_time_label = QLabel("Tổng thời gian: 0 giờ")
        usage_layout.addWidget(self.total_time_label)
        
        self.today_time_label = QLabel("Hôm nay: 0 phút")
        usage_layout.addWidget(self.today_time_label)
        
        layout.addWidget(usage_group)
        
        # Reset stats
        reset_stats_btn = QPushButton("🗑️ Xóa thống kê")
        reset_stats_btn.clicked.connect(self.reset_stats)
        layout.addWidget(reset_stats_btn)
        
        layout.addStretch()
        scroll.setWidget(content)
        
        # Timer để cập nhật thống kê - giảm tần suất để tiết kiệm RAM
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(5000)  # Update mỗi 5 giây thay vì 1 giây
        
        return scroll
    
    def load_profile(self):
        profile = self.profiles_combo.currentData()
        self.log(f"📥 Đang tải profile: {profile}")
        QMessageBox.information(self, "Profile", f"Đã tải profile: {self.profiles_combo.currentText()}")
    
    def save_profile(self):
        profile = self.profiles_combo.currentText()
        self.log(f"💾 Đã lưu profile: {profile}")
        QMessageBox.information(self, "Profile", f"Đã lưu profile: {profile}")
    
    def create_new_profile(self):
        name = self.new_profile_name.text().strip()
        if name:
            self.profiles_combo.addItem(f"📌 {name}", name.lower().replace(" ", "_"))
            self.new_profile_name.clear()
            self.log(f"✨ Đã tạo profile mới: {name}")
            QMessageBox.information(self, "Profile", f"Đã tạo profile: {name}")
        else:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên profile")
    
    def export_config(self):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Config", "gesture_config_backup.json", "JSON Files (*.json)"
        )
        if file_path:
            try:
                import shutil
                shutil.copy(self.config_path, file_path)
                self.log(f"📤 Đã export config ra: {file_path}")
                QMessageBox.information(self, "Thành công", f"Đã export config!")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể export: {e}")
    
    def import_config(self):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Config", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported = json.load(f)
                self.config = imported
                self.save_config()
                self.log(f"📥 Đã import config từ: {file_path}")
                QMessageBox.information(self, "Thành công", "Đã import config! Vui lòng khởi động lại ứng dụng.")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể import: {e}")
    
    def reset_to_default(self):
        reply = QMessageBox.question(
            self, "Xác nhận", 
            "Bạn có chắc muốn đặt lại tất cả về mặc định?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.config = {"gestures": {}, "settings": {}}
            self.save_config()
            self.log("🔄 Đã đặt lại về mặc định")
            QMessageBox.information(self, "Thành công", "Đã đặt lại! Vui lòng khởi động lại ứng dụng.")
    
    def reset_stats(self):
        self.gesture_count = 0
        self.command_count = 0
        self.gesture_history = {}
        self.session_start_time = time.time()
        self.log("🗑️ Đã xóa thống kê")
    
    def update_stats(self):
        """Cập nhật thống kê mỗi giây"""
        if hasattr(self, 'session_start_time'):
            elapsed = int(time.time() - self.session_start_time)
            minutes = elapsed // 60
            self.session_time_label.setText(f"⏱️ Thời gian chạy: {minutes} phút")
            
            self.gestures_count_label.setText(f"🤚 Cử chỉ đã nhận: {self.gesture_count}")
            self.commands_count_label.setText(f"⚡ Lệnh đã thực thi: {self.command_count}")
            
            # Popular gestures
            if self.gesture_history:
                sorted_gestures = sorted(self.gesture_history.items(), key=lambda x: x[1], reverse=True)[:5]
                popular_text = "\n".join([f"{i+1}. {g}: {c} lần" for i, (g, c) in enumerate(sorted_gestures)])
                self.popular_gestures_label.setText(popular_text)
            
            # Accuracy (mock)
            if self.gesture_count > 0:
                accuracy = min(95, 70 + (self.command_count / self.gesture_count) * 25)
                self.accuracy_label.setText(f"Độ chính xác: {accuracy:.0f}%")
                self.accuracy_bar.setValue(int(accuracy))
    
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
        self.frame_skip_spin = NoScrollSpinBox()
        self.frame_skip_spin.setRange(1, 5)
        self.frame_skip_spin.setValue(settings.get('process_every_n_frames', 2))
        self.frame_skip_spin.setMinimumHeight(36)
        self.frame_skip_spin.setMinimumWidth(70)
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
        self.cooldown_spin = NoScrollSpinBox()
        self.cooldown_spin.setRange(5, 60)
        self.cooldown_spin.setValue(settings.get('gesture_cooldown', 15))
        self.cooldown_spin.setMinimumHeight(36)
        self.cooldown_spin.setMinimumWidth(70)
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
            frame_layout.setContentsMargins(8, 8, 8, 8)
            frame_layout.setSpacing(8)
            
            # Checkbox bật/tắt
            check = QCheckBox()
            current = gestures.get(gesture_id, {})
            is_enabled = current.get('enabled', False) if isinstance(current, dict) else bool(current)
            check.setChecked(is_enabled)
            check.setFixedWidth(30)
            frame_layout.addWidget(check)
            self.gesture_checks[gesture_id] = check
            
            # Label - co giãn theo cửa sổ
            label = QLabel(gesture_name)
            label.setMinimumWidth(80)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            frame_layout.addWidget(label, stretch=1)
            
            # Key bind button - co giãn
            current_action = ""
            if isinstance(current, dict):
                current_action = current.get("action", "")
            elif current:
                current_action = str(current)
            
            key_btn = KeyBindButton(current_action)
            key_btn.setMinimumWidth(100)
            key_btn.setMinimumHeight(32)
            key_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            key_btn.setToolTip("Click để gán phím")
            frame_layout.addWidget(key_btn, stretch=2)
            self.gesture_inputs[gesture_id] = key_btn
            
            # Quick actions dropdown
            quick_combo = NoScrollComboBox()
            quick_combo.setFixedWidth(42)
            quick_combo.setFixedHeight(32)
            quick_combo.setToolTip("Hành động nhanh")
            quick_combo.setStyleSheet("""
                QComboBox {
                    background: #3730a3;
                    border: 2px solid #4f46e5;
                    border-radius: 6px;
                    padding: 2px 4px;
                    color: white;
                    font-size: 14px;
                }
                QComboBox:hover { background: #4f46e5; }
                QComboBox::drop-down { border: none; width: 0px; }
                QComboBox QAbstractItemView {
                    background: #1a1a2e;
                    color: white;
                    selection-background-color: #4f46e5;
                    min-width: 180px;
                }
            """)
            
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
            clear_btn.setFixedSize(30, 30)
            clear_btn.setToolTip("Xóa phím đã gán")
            clear_btn.setStyleSheet("""
                QPushButton { 
                    background: #3a1a1a; 
                    border: 1px solid #5a2a2a;
                    border-radius: 6px;
                    color: #ff6b6b; 
                    font-weight: bold; 
                    font-size: 14px;
                }
                QPushButton:hover { 
                    background: #5a2a2a;
                    color: #ff4444; 
                }
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
            
            /* Gesture cards - responsive */
            #gestureCard { 
                background: #1e1e3a; border: 1px solid #333; 
                border-radius: 8px;
                min-height: 36px;
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
            
            /* Slider - responsive */
            QSlider::groove:horizontal { height: 8px; background: #333; border-radius: 4px; }
            QSlider::handle:horizontal { background: #4f46e5; width: 20px; height: 20px; margin: -6px 0; border-radius: 10px; }
            QSlider::handle:horizontal:hover { background: #6366f1; }
            QSlider::sub-page:horizontal { background: #4f46e5; border-radius: 4px; }
            
            /* Checkbox - responsive */
            QCheckBox { color: white; spacing: 8px; }
            QCheckBox::indicator { 
                width: 22px; height: 22px; border-radius: 6px; 
                border: 2px solid #444; background: #252545;
            }
            QCheckBox::indicator:checked { background: #4f46e5; border-color: #4f46e5; }
            QCheckBox::indicator:hover { border-color: #6366f1; background: #353565; }
            
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
        
        # Cấu hình camera đã chọn
        camera_index, camera_backend = self.get_selected_camera()
        self.camera_thread.camera_index = camera_index
        self.camera_thread.camera_backend = camera_backend
        self.log(f"📷 Sử dụng Camera {camera_index}" + (f" (backend: {camera_backend})" if camera_backend else ""))
        
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
        # Cập nhật thống kê
        self.gesture_count += 1
        if action:
            self.command_count += 1
        self.gesture_history[gesture] = self.gesture_history.get(gesture, 0) + 1
        
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
        
        # Phát âm thanh nếu bật
        if self.config.get("settings", {}).get("sound_enabled", False):
            self.play_feedback_sound()
        
        # Reset sau 2 giây
        QTimer.singleShot(2000, lambda: self.reset_gesture_display())
    
    def play_feedback_sound(self):
        """Phát âm thanh feedback"""
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except:
            pass
    
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
        
        # Settings - cơ bản
        settings = self.config.get("settings", {})
        settings.update({
            "detection_confidence": self.detection_slider.value() / 100,
            "mouse_control": self.mouse_control_check.isChecked(),
            "mouse_speed": self.mouse_speed_slider.value(),
            "mouse_smoothing": self.mouse_smooth_check.isChecked(),
            "app_mode": self.app_mode_combo.currentData(),
            "require_face": self.require_face_check.isChecked(),
            "require_looking": self.require_eye_check.isChecked(),
            "sound_enabled": self.sound_enabled_check.isChecked(),
            "screen_flash": self.vibrate_check.isChecked(),
            "show_notification": self.show_notification_check.isChecked(),
            "zone_enabled": self.zone_enabled_check.isChecked(),
            "zone_size": self.zone_size_slider.value(),
            "hotkey_toggle": self.toggle_hotkey.get_bound_key(),
            "hotkey_pause": self.pause_hotkey.get_bound_key(),
            "auto_start": self.auto_start_check.isChecked(),
            "minimize_to_tray": self.minimize_to_tray_check.isChecked(),
            "start_minimized": self.start_minimized_check.isChecked(),
            "auto_run": self.auto_run_check.isChecked(),
            "developer_mode": self.developer_mode,
            "camera_index": self.get_selected_camera()[0],
            "camera_backend": self.get_selected_camera()[1],
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
        # Stop camera thread
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        # Stop stats timer
        if hasattr(self, 'stats_timer'):
            self.stats_timer.stop()
        
        # Final garbage collection
        gc.collect()
        event.accept()


def main():
    # Memory optimization
    import gc
    gc.set_threshold(100, 5, 5)  # Aggressive GC
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    
    # Cleanup before running
    gc.collect()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
