"""
Optimized Gesture Recognizer - Nhận diện cử chỉ tối ưu hiệu năng
Chỉ nhận diện các cử chỉ được bật trong config
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class GestureResult:
    """Kết quả nhận diện cử chỉ"""
    name: str
    confidence: float
    fingers: List[int]
    

class OptimizedGestureRecognizer:
    """
    Lớp nhận diện cử chỉ tối ưu
    - Chỉ kiểm tra các cử chỉ được bật trong config
    - Cache kết quả để tránh tính toán lặp
    - Early exit khi tìm thấy match
    """
    
    # Map gesture name -> function check
    GESTURE_CHECKS = {
        'fist': 'check_fist',
        'open_palm': 'check_open_palm',
        'pointing': 'check_pointing',
        'peace': 'check_peace',
        'thumbs_up': 'check_thumbs_up',
        'thumb_up': 'check_thumbs_up',  # alias
        'thumbs_down': 'check_thumbs_down',
        'thumb_down': 'check_thumbs_down',  # alias
        'ok': 'check_ok',
        'rock': 'check_rock',
        'three': 'check_three',
        'four': 'check_four',
        'call': 'check_call',
        'swipe_up': 'check_swipe_up',
        'swipe_down': 'check_swipe_down',
        'swipe_left': 'check_swipe_left',
        'swipe_right': 'check_swipe_right',
    }
    
    def __init__(self, enabled_gestures: Set[str] = None, fist_threshold: float = 0.4):
        """
        Khởi tạo recognizer tối ưu
        
        Args:
            enabled_gestures: Set các cử chỉ được bật (None = tất cả)
            fist_threshold: Ngưỡng nhận diện nắm đấm
        """
        self.fist_threshold = fist_threshold
        self.enabled_gestures = enabled_gestures or set(self.GESTURE_CHECKS.keys())
        
        # Build optimized check list - chỉ check những gesture được bật
        self.active_checks = []
        for gesture_name in self.enabled_gestures:
            if gesture_name in self.GESTURE_CHECKS:
                check_method = self.GESTURE_CHECKS[gesture_name]
                if hasattr(self, check_method):
                    self.active_checks.append((gesture_name, getattr(self, check_method)))
        
        # Cache cho landmarks history (dùng cho swipe detection)
        self.landmarks_history: List[np.ndarray] = []
        self.max_history = 10
        
        # Cache kết quả fingers_up
        self._last_landmarks_hash = None
        self._cached_fingers = None
        self._cached_palm_size = None
        
        print(f"[OptimizedRecognizer] Enabled {len(self.active_checks)} gestures: {list(self.enabled_gestures)}")
    
    def update_enabled_gestures(self, enabled_gestures: Set[str]):
        """Cập nhật danh sách cử chỉ được bật"""
        self.enabled_gestures = enabled_gestures
        self.active_checks = []
        for gesture_name in self.enabled_gestures:
            if gesture_name in self.GESTURE_CHECKS:
                check_method = self.GESTURE_CHECKS[gesture_name]
                if hasattr(self, check_method):
                    self.active_checks.append((gesture_name, getattr(self, check_method)))
        print(f"[OptimizedRecognizer] Updated to {len(self.active_checks)} gestures")
    
    def _hash_landmarks(self, landmarks: np.ndarray) -> int:
        """Tạo hash nhanh từ landmarks để cache"""
        # Chỉ hash vài điểm quan trọng thay vì toàn bộ
        key_points = landmarks[[0, 4, 8, 12, 16, 20], :2]  # wrist + finger tips, x,y only
        return hash(key_points.tobytes())
    
    def _get_fingers_up(self, landmarks: np.ndarray) -> List[int]:
        """
        Xác định ngón tay nào đang giơ lên (có cache)
        
        Returns:
            List [thumb, index, middle, ring, pinky] với 1=giơ lên, 0=gập xuống
        """
        # Check cache
        lm_hash = self._hash_landmarks(landmarks)
        if lm_hash == self._last_landmarks_hash and self._cached_fingers is not None:
            return self._cached_fingers
        
        self._last_landmarks_hash = lm_hash
        
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]
        
        # Ngón cái - so sánh x coordinate
        if landmarks[tip_ids[0]][0] < landmarks[tip_ids[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 ngón còn lại - so sánh y coordinate (nhỏ hơn = cao hơn trong ảnh)
        for i in range(1, 5):
            if landmarks[tip_ids[i]][1] < landmarks[tip_ids[i] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        self._cached_fingers = fingers
        return fingers
    
    def _get_palm_size(self, landmarks: np.ndarray) -> float:
        """Tính kích thước bàn tay (có cache)"""
        if self._cached_palm_size is not None and self._last_landmarks_hash == self._hash_landmarks(landmarks):
            return self._cached_palm_size
        
        wrist = landmarks[0]
        palm_base = landmarks[9]
        self._cached_palm_size = np.linalg.norm(palm_base - wrist)
        return self._cached_palm_size
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Tính khoảng cách nhanh (chỉ x,y)"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # ============ GESTURE CHECKS ============
    
    def check_fist(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra nắm đấm"""
        if sum(fingers) != 0:
            return False, 0.0
        
        # Kiểm tra độ chặt
        palm_size = self._get_palm_size(landmarks)
        tip_ids = [8, 12, 16, 20]
        mcp_ids = [5, 9, 13, 17]
        
        for tip_id, mcp_id in zip(tip_ids, mcp_ids):
            dist = self._distance(landmarks[tip_id], landmarks[mcp_id])
            if dist > palm_size * self.fist_threshold:
                return False, 0.0
        
        return True, 0.95
    
    def check_open_palm(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra bàn tay mở"""
        if sum(fingers) == 5:
            return True, 0.95
        return False, 0.0
    
    def check_pointing(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra chỉ tay"""
        if fingers == [0, 1, 0, 0, 0]:
            return True, 0.95
        return False, 0.0
    
    def check_peace(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra dấu V/Peace"""
        if fingers == [0, 1, 1, 0, 0]:
            return True, 0.95
        return False, 0.0
    
    def check_thumbs_up(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra ngón cái lên"""
        if fingers == [1, 0, 0, 0, 0]:
            # Kiểm tra ngón cái hướng lên (y nhỏ)
            if landmarks[4][1] < landmarks[3][1]:
                return True, 0.95
        return False, 0.0
    
    def check_thumbs_down(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra ngón cái xuống"""
        if fingers == [1, 0, 0, 0, 0]:
            # Kiểm tra ngón cái hướng xuống (y lớn)
            if landmarks[4][1] > landmarks[3][1]:
                return True, 0.95
        return False, 0.0
    
    def check_ok(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra dấu OK"""
        # Ngón cái và trỏ chạm nhau
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = self._distance(thumb_tip, index_tip)
        
        palm_size = self._get_palm_size(landmarks)
        if distance < palm_size * 0.2 and sum(fingers[2:]) >= 2:
            return True, 0.90
        return False, 0.0
    
    def check_rock(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra dấu rock"""
        if fingers == [0, 1, 0, 0, 1]:
            return True, 0.95
        return False, 0.0
    
    def check_three(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra số 3"""
        if fingers == [1, 1, 1, 0, 0]:
            return True, 0.95
        return False, 0.0
    
    def check_four(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra số 4"""
        if fingers == [0, 1, 1, 1, 1]:
            return True, 0.95
        return False, 0.0
    
    def check_call(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra cử chỉ gọi điện"""
        if fingers == [1, 0, 0, 0, 1]:
            return True, 0.95
        return False, 0.0
    
    def check_swipe_up(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra vuốt lên"""
        return self._check_swipe(landmarks, 'up')
    
    def check_swipe_down(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra vuốt xuống"""
        return self._check_swipe(landmarks, 'down')
    
    def check_swipe_left(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra vuốt trái"""
        return self._check_swipe(landmarks, 'left')
    
    def check_swipe_right(self, landmarks: np.ndarray, fingers: List[int]) -> Tuple[bool, float]:
        """Kiểm tra vuốt phải"""
        return self._check_swipe(landmarks, 'right')
    
    def _check_swipe(self, landmarks: np.ndarray, direction: str) -> Tuple[bool, float]:
        """Kiểm tra cử chỉ vuốt"""
        if len(self.landmarks_history) < 5:
            return False, 0.0
        
        # So sánh vị trí hiện tại với 5 frame trước
        prev = self.landmarks_history[-5]
        curr = landmarks
        
        # Tracking points
        tracking = [0, 8, 9, 12]
        curr_pos = np.mean([curr[i][:2] for i in tracking], axis=0)
        prev_pos = np.mean([prev[i][:2] for i in tracking], axis=0)
        
        delta = curr_pos - prev_pos
        dist = np.linalg.norm(delta)
        
        threshold = 20
        if dist < threshold:
            return False, 0.0
        
        # Xác định hướng
        if direction == 'up' and delta[1] < -threshold and abs(delta[1]) > abs(delta[0]):
            return True, 0.85
        elif direction == 'down' and delta[1] > threshold and abs(delta[1]) > abs(delta[0]):
            return True, 0.85
        elif direction == 'left' and delta[0] < -threshold and abs(delta[0]) > abs(delta[1]):
            return True, 0.85
        elif direction == 'right' and delta[0] > threshold and abs(delta[0]) > abs(delta[1]):
            return True, 0.85
        
        return False, 0.0
    
    def recognize(self, landmarks: np.ndarray) -> Optional[GestureResult]:
        """
        Nhận diện cử chỉ từ landmarks
        CHỈ kiểm tra các cử chỉ được bật trong config
        
        Args:
            landmarks: Array shape (21, 3) hoặc (21, 2)
            
        Returns:
            GestureResult hoặc None nếu không nhận diện được
        """
        if landmarks is None or len(landmarks) < 21:
            return None
        
        # Ensure numpy array
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)
        
        # Cập nhật history cho swipe detection
        self.landmarks_history.append(landmarks.copy())
        if len(self.landmarks_history) > self.max_history:
            self.landmarks_history.pop(0)
        
        # Tính fingers một lần duy nhất
        fingers = self._get_fingers_up(landmarks)
        
        # Early exit checks - kiểm tra nhanh trước
        total_fingers = sum(fingers)
        
        # Chỉ check các gesture được bật
        for gesture_name, check_func in self.active_checks:
            try:
                is_match, confidence = check_func(landmarks, fingers)
                if is_match:
                    return GestureResult(
                        name=gesture_name,
                        confidence=confidence,
                        fingers=fingers
                    )
            except Exception:
                continue
        
        return None
    
    def recognize_from_list(self, landmarks_list: List[List[float]]) -> Optional[GestureResult]:
        """
        Nhận diện từ landmarks dạng list [[id, x, y, z], ...]
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return None
        
        # Convert to numpy array (chỉ lấy x, y)
        landmarks = np.array([[lm[1], lm[2]] for lm in landmarks_list], dtype=np.float32)
        return self.recognize(landmarks)


class PerformanceOptimizer:
    """
    Lớp tối ưu hiệu năng tổng thể
    """
    
    def __init__(self, config: dict):
        """
        Khởi tạo optimizer
        
        Args:
            config: Config dict chứa gestures và settings
        """
        self.config = config
        self.settings = config.get('settings', {})
        
        # Lấy danh sách gesture được bật
        self.enabled_gestures = self._get_enabled_gestures()
        
        # Các thông số tối ưu
        self.frame_skip = self._calculate_frame_skip()
        self.resolution_scale = self._calculate_resolution_scale()
        self.process_every_n_frames = self.settings.get('process_every_n_frames', 2)
        
        print(f"[PerformanceOptimizer] Initialized:")
        print(f"  - Enabled gestures: {len(self.enabled_gestures)}")
        print(f"  - Frame skip: {self.frame_skip}")
        print(f"  - Resolution scale: {self.resolution_scale}")
    
    def _get_enabled_gestures(self) -> Set[str]:
        """Lấy danh sách cử chỉ được bật từ config"""
        gestures = self.config.get('gestures', {})
        enabled = set()
        
        for gesture_id, gesture_config in gestures.items():
            if isinstance(gesture_config, dict):
                if gesture_config.get('enabled', True) and gesture_config.get('action'):
                    enabled.add(gesture_id)
            elif gesture_config:  # string action
                enabled.add(gesture_id)
        
        return enabled
    
    def _calculate_frame_skip(self) -> int:
        """
        Tính số frame skip dựa trên số gesture được bật
        Ít gesture = có thể skip nhiều hơn
        """
        n_gestures = len(self.enabled_gestures)
        
        if n_gestures <= 2:
            return 3  # Xử lý 1/3 frame
        elif n_gestures <= 5:
            return 2  # Xử lý 1/2 frame
        else:
            return 1  # Xử lý mọi frame
    
    def _calculate_resolution_scale(self) -> float:
        """
        Tính scale độ phân giải dựa trên cấu hình
        """
        # Nếu user set low performance mode
        if self.settings.get('low_performance_mode', False):
            return 0.5
        
        # Dựa trên số gesture
        n_gestures = len(self.enabled_gestures)
        if n_gestures <= 3:
            return 0.75  # Giảm 25% resolution
        else:
            return 1.0
    
    def should_process_frame(self, frame_count: int) -> bool:
        """Kiểm tra có nên xử lý frame này không"""
        return frame_count % self.process_every_n_frames == 0
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame nếu cần để tiết kiệm CPU"""
        if self.resolution_scale >= 1.0:
            return frame
        
        import cv2
        h, w = frame.shape[:2]
        new_w = int(w * self.resolution_scale)
        new_h = int(h * self.resolution_scale)
        return cv2.resize(frame, (new_w, new_h))
    
    def get_optimized_settings(self) -> dict:
        """Trả về settings đã tối ưu"""
        return {
            'enabled_gestures': self.enabled_gestures,
            'frame_skip': self.frame_skip,
            'resolution_scale': self.resolution_scale,
            'max_hands': 1 if len(self.enabled_gestures) <= 5 else self.settings.get('max_hands', 1),
            'detection_confidence': self.settings.get('detection_confidence', 0.7),
            'tracking_confidence': max(0.3, self.settings.get('tracking_confidence', 0.5) - 0.1),  # Giảm một chút
        }


# Singleton instance để tái sử dụng
_recognizer_instance = None

def get_optimized_recognizer(enabled_gestures: Set[str] = None, 
                             fist_threshold: float = 0.4) -> OptimizedGestureRecognizer:
    """Factory function để lấy recognizer instance"""
    global _recognizer_instance
    
    if _recognizer_instance is None:
        _recognizer_instance = OptimizedGestureRecognizer(enabled_gestures, fist_threshold)
    elif enabled_gestures is not None:
        _recognizer_instance.update_enabled_gestures(enabled_gestures)
        _recognizer_instance.fist_threshold = fist_threshold
    
    return _recognizer_instance
