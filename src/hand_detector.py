"""
Module Hand Detector - Phát hiện và theo dõi bàn tay
Sử dụng MediaPipe Hands để trích xuất 21 điểm khớp (keypoints)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class HandDetector:
    """
    Lớp HandDetector sử dụng MediaPipe để phát hiện bàn tay và trích xuất keypoints
    """
    
    def __init__(self, 
                 max_hands: int = 1,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5,
                 require_face: bool = False,
                 fist_threshold: float = 0.4):
        """
        Khởi tạo Hand Detector
        
        Args:
            max_hands: Số lượng tay tối đa cần phát hiện
            detection_confidence: Ngưỡng tin cậy cho việc phát hiện
            tracking_confidence: Ngưỡng tin cậy cho việc theo dõi
            require_face: Yêu cầu phải có khuôn mặt mới nhận diện cử chỉ (mặc định: True)
            fist_threshold: Ngưỡng độ chặt nắm tay (0.2-0.6, cao = chặt hơn)
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.require_face = require_face
        self.fist_threshold = fist_threshold
        
        # Khởi tạo MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Khởi tạo drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Khởi tạo Face Detection và Face Mesh (chỉ khi cần)
        if self.require_face:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 = short range (< 2m), 1 = full range
                min_detection_confidence=0.5
            )
            # Thêm Face Mesh để phát hiện mắt
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # Bật để có iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.mp_face_detection = None
            self.face_detection = None
            self.mp_face_mesh = None
            self.face_mesh = None
        
        # Lưu trữ landmarks trước đó để phát hiện chuyển động
        self.prev_landmarks = None
        self.landmarks_history = []
        self.max_history_length = 15
        
        # Lưu trữ trạng thái phát hiện khuôn mặt
        self.face_detected = False
        self.face_detection_history = []
        self.face_history_length = 5  # Kiểm tra 5 frame gần nhất
        
    def find_hands(self, 
                   image: np.ndarray, 
                   draw: bool = True) -> Tuple[np.ndarray, Optional[List]]:
        """
        Tìm kiếm bàn tay trong hình ảnh
        
        Args:
            image: Hình ảnh đầu vào (BGR format)
            draw: Có vẽ keypoints lên hình ảnh không
            
        Returns:
            Tuple[image, results]: Hình ảnh đã xử lý và kết quả phát hiện
        """
        # Chuyển đổi BGR sang RGB (MediaPipe yêu cầu RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Xử lý hình ảnh để phát hiện bàn tay
        results = self.hands.process(image_rgb)
        
        # Vẽ các điểm khớp và kết nối nếu có yêu cầu
        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return image, results
    
    def get_landmarks(self, 
                      image: np.ndarray, 
                      results) -> List[List[float]]:
        """
        Lấy tọa độ các điểm khớp (landmarks) từ kết quả phát hiện
        
        Args:
            image: Hình ảnh đầu vào
            results: Kết quả từ MediaPipe
            
        Returns:
            List các landmarks với format [id, x, y, z]
            Trong đó:
                - id: ID của điểm khớp (0-20)
                - x, y: Tọa độ pixel trên hình ảnh
                - z: Độ sâu tương đối
        """
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = image.shape
                
                for id, landmark in enumerate(hand_landmarks.landmark):
                    # Chuyển đổi tọa độ normalized (0-1) sang pixel
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cz = landmark.z
                    
                    landmarks_list.append([id, cx, cy, cz])
        
        # Cập nhật lịch sử landmarks để phát hiện chuyển động
        if landmarks_list:
            self.prev_landmarks = landmarks_list.copy()
            self.landmarks_history.append(landmarks_list.copy())
            
            # Giới hạn độ dài lịch sử
            if len(self.landmarks_history) > self.max_history_length:
                self.landmarks_history.pop(0)
        
        return landmarks_list
    
    def get_normalized_landmarks(self, results) -> Optional[np.ndarray]:
        """
        Lấy tọa độ normalized (0-1) của các điểm khớp
        Phù hợp để đưa vào mô hình Machine Learning
        
        Args:
            results: Kết quả từ MediaPipe
            
        Returns:
            Array numpy shape (21, 3) chứa [x, y, z] của 21 điểm khớp
            hoặc None nếu không phát hiện được tay
        """
        if not results.multi_hand_landmarks:
            return None
        
        # Lấy bàn tay đầu tiên
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Chuyển đổi sang array numpy
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks, dtype=np.float32)
    
    def get_hand_info(self, results) -> Optional[dict]:
        """
        Lấy thông tin chi tiết về bàn tay được phát hiện
        
        Args:
            results: Kết quả từ MediaPipe
            
        Returns:
            Dictionary chứa thông tin về bàn tay hoặc None
        """
        if not results.multi_hand_landmarks:
            return None
        
        hand_info = {
            'landmarks': self.get_normalized_landmarks(results),
            'handedness': results.multi_handedness[0].classification[0].label,  # Left/Right
            'confidence': results.multi_handedness[0].classification[0].score
        }
        
        return hand_info
    
    def calculate_distance(self, 
                          point1: Tuple[int, int], 
                          point2: Tuple[int, int]) -> float:
        """
        Tính khoảng cách Euclidean giữa 2 điểm
        
        Args:
            point1: Tọa độ điểm 1 (x, y)
            point2: Tọa độ điểm 2 (x, y)
            
        Returns:
            Khoảng cách giữa 2 điểm
        """
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def fingers_up(self, landmarks_list: List[List[float]]) -> List[int]:
        """
        Xác định ngón tay nào đang giơ lên
        
        Args:
            landmarks_list: List các landmarks
            
        Returns:
            List [thumb, index, middle, ring, pinky] với 1=giơ lên, 0=gập xuống
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return [0, 0, 0, 0, 0]
        
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]  # ID của các đầu ngón tay
        
        # Ngón cái (logic đặc biệt)
        if landmarks_list[tip_ids[0]][1] < landmarks_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 ngón còn lại
        for id in range(1, 5):
            if landmarks_list[tip_ids[id]][2] < landmarks_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def detect_gesture(self, landmarks_list: List[List[float]]) -> str:
        """
        Nhận diện cử chỉ cụ thể dựa trên trạng thái các ngón tay
        
        Args:
            landmarks_list: List các landmarks
            
        Returns:
            Tên cử chỉ được nhận diện
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return "unknown"
        
        fingers = self.fingers_up(landmarks_list)
        total_fingers = sum(fingers)
        
        # Nắm đấm (tất cả ngón gập) - kiểm tra chặt hơn
        if total_fingers == 0:
            # Kiểm tra độ gập: các đầu ngón phải gần lòng bàn tay
            # Lấy tọa độ các điểm
            wrist = landmarks_list[0]  # Cổ tay
            palm_base = landmarks_list[9]  # Gốc ngón giữa (giữa lòng bàn tay)
            
            # Khoảng cách tham chiếu (cổ tay đến gốc ngón giữa)
            palm_size = self.calculate_distance(
                (wrist[1], wrist[2]), 
                (palm_base[1], palm_base[2])
            )
            
            # Kiểm tra từng đầu ngón tay (trừ ngón cái)
            tip_ids = [8, 12, 16, 20]  # Đầu ngón trỏ, giữa, áp út, út
            mcp_ids = [5, 9, 13, 17]   # Gốc các ngón
            
            is_tight_fist = True
            for tip_id, mcp_id in zip(tip_ids, mcp_ids):
                tip = landmarks_list[tip_id]
                mcp = landmarks_list[mcp_id]
                
                # Khoảng cách từ đầu ngón đến gốc ngón
                tip_to_mcp = self.calculate_distance(
                    (tip[1], tip[2]),
                    (mcp[1], mcp[2])
                )
                
                # Nếu đầu ngón quá xa gốc ngón (> threshold% kích thước bàn tay) -> nắm hờ
                if tip_to_mcp > palm_size * self.fist_threshold:
                    is_tight_fist = False
                    break
            
            if is_tight_fist:
                return "fist"
            else:
                return "loose_fist"  # Nắm hờ - không thực thi action
        
        # Bàn tay mở (tất cả ngón giơ)
        if total_fingers == 5:
            return "open_palm"
        
        # Ngón trỏ (chỉ ngón trỏ giơ)
        if fingers == [0, 1, 0, 0, 0]:
            return "pointing"
        
        # Peace/Victory (ngón trỏ và ngón giữa)
        if fingers == [0, 1, 1, 0, 0]:
            return "peace"
        
        # OK (ngón cái và ngón trỏ chạm nhau)
        if self.is_ok_gesture(landmarks_list):
            return "ok"
        
        # Thumbs up
        if fingers == [1, 0, 0, 0, 0]:
            return "thumbs_up"
        
        # Rock (ngón út và ngón trỏ giơ)
        if fingers == [0, 1, 0, 0, 1]:
            return "rock"
        
        # Số 3 (ngón cái, trỏ, giữa)
        if fingers == [1, 1, 1, 0, 0]:
            return "three"
        
        # Số 4 (trỏ, giữa, áp út, út)
        if fingers == [0, 1, 1, 1, 1]:
            return "four"
        
        # Gọi điện thoại (ngón cái và út giơ)
        if fingers == [1, 0, 0, 0, 1]:
            return "call"
        
        return f"fingers_{total_fingers}"
    
    def is_ok_gesture(self, landmarks_list: List[List[float]]) -> bool:
        """
        Kiểm tra cử chỉ OK (ngón cái và ngón trỏ chạm nhau tạo thành vòng tròn)
        
        Args:
            landmarks_list: List các landmarks
            
        Returns:
            True nếu là cử chỉ OK
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return False
        
        # Tọa độ đầu ngón cái (4) và đầu ngón trỏ (8)
        thumb_tip = (landmarks_list[4][1], landmarks_list[4][2])
        index_tip = (landmarks_list[8][1], landmarks_list[8][2])
        
        # Tính khoảng cách
        distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Nếu khoảng cách nhỏ hơn ngưỡng, xem như đang chạm nhau
        # Kiểm tra thêm 3 ngón còn lại có giơ lên không
        fingers = self.fingers_up(landmarks_list)
        if distance < 30 and sum(fingers[2:]) >= 2:  # 3 ngón cuối giơ lên
            return True
        
        return False
    
    def is_pinching(self, landmarks_list: List[List[float]]) -> Tuple[bool, float]:
        """
        Kiểm tra cử chỉ nhéo/kẹp (ngón cái và ngón trỏ gần nhau)
        
        Args:
            landmarks_list: List các landmarks
            
        Returns:
            Tuple[is_pinching, distance]: Có đang nhéo không và khoảng cách
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return False, 0.0
        
        thumb_tip = (landmarks_list[4][1], landmarks_list[4][2])
        index_tip = (landmarks_list[8][1], landmarks_list[8][2])
        
        distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Ngưỡng cho việc nhéo
        is_pinching = distance < 40
        
        return is_pinching, distance
    
    def get_hand_orientation(self, landmarks_list: List[List[float]]) -> str:
        """
        Xác định hướng của bàn tay (lên, xuống, trái, phải)
        
        Args:
            landmarks_list: List các landmarks
            
        Returns:
            Hướng của bàn tay: "up", "down", "left", "right"
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return "unknown"
        
        # So sánh vị trí cổ tay (0) và ngón giữa (9)
        wrist_y = landmarks_list[0][2]
        middle_base_y = landmarks_list[9][2]
        
        wrist_x = landmarks_list[0][1]
        middle_base_x = landmarks_list[9][1]
        
        # Xác định hướng dọc
        vertical_diff = abs(wrist_y - middle_base_y)
        horizontal_diff = abs(wrist_x - middle_base_x)
        
        if vertical_diff > horizontal_diff:
            return "up" if wrist_y > middle_base_y else "down"
        else:
            return "right" if wrist_x < middle_base_x else "left"
    
    def is_swiping(self, landmarks_list: Optional[List[List[float]]] = None, 
                   prev_landmarks: Optional[List[List[float]]] = None,
                   threshold: float = 20) -> Tuple[bool, str]:
        """
        Phát hiện cử chỉ vuốt (swipe) bằng cách so sánh vị trí hiện tại với trước đó
        
        Args:
            landmarks_list: List các landmarks hiện tại (nếu None sẽ dùng landmarks gần nhất)
            prev_landmarks: List các landmarks frame trước (nếu None sẽ dùng self.prev_landmarks)
            threshold: Ngưỡng khoảng cách để xác định vuốt
            
        Returns:
            Tuple[is_swiping, direction]: Có đang vuốt không và hướng vuốt
        """
        # Sử dụng landmarks được lưu nếu không được truyền vào
        if landmarks_list is None:
            landmarks_list = self.landmarks_history[-1] if self.landmarks_history else None
        
        if prev_landmarks is None:
            # So sánh với frame càng xa càng dễ phát hiện chuyển động
            if len(self.landmarks_history) >= 5:
                prev_landmarks = self.landmarks_history[-5]
            elif len(self.landmarks_history) >= 3:
                prev_landmarks = self.landmarks_history[-3]
            else:
                prev_landmarks = self.prev_landmarks
        
        if not landmarks_list or not prev_landmarks:
            return False, "none"
        
        if len(landmarks_list) < 21 or len(prev_landmarks) < 21:
            return False, "none"
        
        # Sử dụng nhiều điểm để theo dõi chuyển động (chính xác hơn)
        # Lấy trung bình của ngón trỏ (8), giữa (12) và cổ tay (0)
        tracking_points = [0, 8, 9, 12]
        
        curr_x = sum(landmarks_list[i][1] for i in tracking_points) / len(tracking_points)
        curr_y = sum(landmarks_list[i][2] for i in tracking_points) / len(tracking_points)
        
        prev_x = sum(prev_landmarks[i][1] for i in tracking_points) / len(tracking_points)
        prev_y = sum(prev_landmarks[i][2] for i in tracking_points) / len(tracking_points)
        
        delta_x = curr_x - prev_x
        delta_y = curr_y - prev_y
        
        # Tính khoảng cách di chuyển
        distance = np.sqrt(delta_x**2 + delta_y**2)
        
        if distance < threshold:
            return False, "none"
        
        # Giảm yêu cầu hướng rõ ràng để nhạy hơn
        dominant_threshold = 1.15  # Hướng chính chỉ cần lớn hơn 15%
        
        # Xác định hướng vuốt
        # Lưu ý: Trong OpenCV, Y tăng từ trên xuống dưới
        # delta_y < 0 = tay di chuyển từ DƯỚI lên TRÊN (vuốt lên như điện thoại)
        # delta_y > 0 = tay di chuyển từ TRÊN xuống DƯỚI (vuốt xuống như điện thoại)
        
        abs_dx = abs(delta_x)
        abs_dy = abs(delta_y)
        
        # Kiểm tra hướng nào chiếm ưu thế
        if abs_dx > abs_dy * dominant_threshold:
            # Chuyển động ngang chiếm ưu thế
            direction = "right" if delta_x > 0 else "left"
        elif abs_dy > abs_dx * dominant_threshold:
            # Chuyển động dọc chiếm ưu thế
            # QUAN TRỌNG: delta_y < 0 = tay đi LÊN (vuốt lên)
            direction = "up" if delta_y < 0 else "down"
        else:
            # Gần bằng nhau - chọn hướng có delta lớn hơn
            if abs_dx > abs_dy:
                direction = "right" if delta_x > 0 else "left"
            else:
                direction = "up" if delta_y < 0 else "down"
        
        return True, direction
    
    def detect_swipe_gesture(self, landmarks_list: List[List[float]]) -> str:
        """
        Phát hiện cử chỉ vuốt và trả về tên cử chỉ
        
        Args:
            landmarks_list: List các landmarks hiện tại
            
        Returns:
            Tên cử chỉ vuốt: "swipe_up", "swipe_down", "swipe_left", "swipe_right", hoặc "none"
        """
        is_swipe, direction = self.is_swiping(landmarks_list)
        
        if is_swipe and direction != "none":
            return f"swipe_{direction}"
        
        return "none"
    
    def calculate_hand_size(self, landmarks_list: List[List[float]]) -> float:
        """
        Tính kích thước bàn tay (để phát hiện zoom in/out)
        
        Args:
            landmarks_list: List các landmarks
            
        Returns:
            Kích thước bàn tay (khoảng cách từ cổ tay đến ngón giữa)
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return 0.0
        
        wrist = (landmarks_list[0][1], landmarks_list[0][2])
        middle_tip = (landmarks_list[12][1], landmarks_list[12][2])
        
        return self.calculate_distance(wrist, middle_tip)
    
    def detect_zoom_gesture(self, landmarks_list: Optional[List[List[float]]] = None,
                           prev_landmarks: Optional[List[List[float]]] = None) -> Tuple[bool, str]:
        """
        Phát hiện cử chỉ zoom (spread = zoom in, pinch = zoom out)
        
        Args:
            landmarks_list: List các landmarks hiện tại (nếu None sẽ dùng landmarks gần nhất)
            prev_landmarks: List các landmarks frame trước (nếu None sẽ dùng self.prev_landmarks)
            
        Returns:
            Tuple[is_zooming, zoom_type]: Có đang zoom không và loại zoom
        """
        # Sử dụng landmarks được lưu nếu không được truyền vào
        if landmarks_list is None:
            landmarks_list = self.landmarks_history[-1] if self.landmarks_history else None
        
        if prev_landmarks is None:
            prev_landmarks = self.prev_landmarks
        
        if not landmarks_list or not prev_landmarks:
            return False, "none"
        
        curr_size = self.calculate_hand_size(landmarks_list)
        prev_size = self.calculate_hand_size(prev_landmarks)
        
        size_diff = curr_size - prev_size
        
        # Ngưỡng thay đổi kích thước
        if abs(size_diff) < 5:
            return False, "none"
        
        zoom_type = "zoom_in" if size_diff > 0 else "zoom_out"
        return True, zoom_type
    
    def is_waving(self, landmarks_history: Optional[List[List[List[float]]]] = None, 
                  frames: int = 10) -> bool:
        """
        Phát hiện cử chỉ vẫy tay (chuyển động qua lại)
        
        Args:
            landmarks_history: Lịch sử landmarks của nhiều frame (nếu None sẽ dùng self.landmarks_history)
            frames: Số frame tối thiểu để kiểm tra
            
        Returns:
            True nếu đang vẫy tay
        """
        # Sử dụng lịch sử được lưu nếu không được truyền vào
        if landmarks_history is None:
            landmarks_history = self.landmarks_history
        
        if len(landmarks_history) < frames:
            return False
        
        # Lấy vị trí x của ngón giữa trong các frame
        x_positions = []
        for landmarks in landmarks_history[-frames:]:
            if landmarks and len(landmarks) >= 21:
                x_positions.append(landmarks[12][1])  # Ngón giữa
        
        if len(x_positions) < frames:
            return False
        
        # Đếm số lần thay đổi hướng (oscillation)
        direction_changes = 0
        for i in range(1, len(x_positions) - 1):
            if (x_positions[i] > x_positions[i-1] and x_positions[i] > x_positions[i+1]) or \
               (x_positions[i] < x_positions[i-1] and x_positions[i] < x_positions[i+1]):
                direction_changes += 1
        
        # Nếu có ít nhất 2-3 lần thay đổi hướng, xem như đang vẫy tay
        return direction_changes >= 2
    
    def get_finger_angles(self, landmarks_list: List[List[float]]) -> List[float]:
        """
        Tính góc uốn của từng ngón tay
        
        Args:
            landmarks_list: List các landmarks
            
        Returns:
            List góc của 5 ngón tay (degree)
        """
        if not landmarks_list or len(landmarks_list) < 21:
            return [0.0] * 5
        
        angles = []
        finger_joints = [
            [1, 2, 3, 4],      # Ngón cái
            [5, 6, 7, 8],      # Ngón trỏ
            [9, 10, 11, 12],   # Ngón giữa
            [13, 14, 15, 16],  # Ngón áp út
            [17, 18, 19, 20]   # Ngón út
        ]
        
        for joints in finger_joints:
            # Tính góc tại khớp giữa (joint[1])
            p1 = np.array([landmarks_list[joints[0]][1], landmarks_list[joints[0]][2]])
            p2 = np.array([landmarks_list[joints[1]][1], landmarks_list[joints[1]][2]])
            p3 = np.array([landmarks_list[joints[3]][1], landmarks_list[joints[3]][2]])
            
            # Vector từ p2 đến p1 và p2 đến p3
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Tính góc
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(np.degrees(angle))
        
        return angles
    
    def detect_face(self, image: np.ndarray, draw: bool = False) -> Tuple[bool, Optional[dict]]:
        """
        Phát hiện khuôn mặt trong hình ảnh
        
        Args:
            image: Hình ảnh đầu vào (BGR format)
            draw: Có vẽ khung khuôn mặt lên hình ảnh không
            
        Returns:
            Tuple[face_detected, face_info]: Có phát hiện khuôn mặt không và thông tin
        """
        # Nếu không yêu cầu face detection, luôn trả về True
        if not self.require_face or self.face_detection is None:
            self.face_detected = True
            return True, None
        
        # Chuyển đổi BGR sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Phát hiện khuôn mặt
        results = self.face_detection.process(image_rgb)
        
        face_detected = False
        face_info = None
        
        if results.detections:
            face_detected = True
            detection = results.detections[0]  # Lấy khuôn mặt đầu tiên
            
            # Lấy bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = image.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            
            face_info = {
                'bbox': bbox,
                'confidence': detection.score[0]
            }
            
            # Vẽ khung nếu yêu cầu
            if draw:
                x, y, width, height = bbox
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image, f"Face: {face_info['confidence']:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Cập nhật lịch sử
        self.face_detection_history.append(face_detected)
        if len(self.face_detection_history) > self.face_history_length:
            self.face_detection_history.pop(0)
        
        # Xác định trạng thái: cần ít nhất 3/5 frame có mặt
        self.face_detected = sum(self.face_detection_history) >= 3
        
        return face_detected, face_info
    
    def is_face_present(self) -> bool:
        """
        Kiểm tra xem có khuôn mặt trong frame gần đây không
        
        Returns:
            True nếu phát hiện khuôn mặt ổn định
        """
        return self.face_detected
    
    def is_looking_at_camera(self, image: np.ndarray, draw: bool = True) -> tuple:
        """
        Kiểm tra xem người dùng có đang nhìn vào camera không
        Sử dụng Face Mesh + iris landmarks để phát hiện hướng nhìn chính xác
        
        Thuật toán tối ưu:
        - Tính vị trí iris trong mắt theo chiều ngang (X)
        - Ngưỡng 0.30-0.70 = nhìn thẳng (dễ nhận hơn)
        - Hiển thị debug info bằng tiếng Việt
        
        Args:
            image: Hình ảnh đầu vào (BGR format)
            draw: Có vẽ thông tin lên ảnh không
            
        Returns:
            tuple: (is_looking, debug_info) - đang nhìn không và thông tin debug
        """
        # Nếu không yêu cầu face -> luôn True
        if not self.require_face or self.face_mesh is None:
            return True, None
        
        # Chuyển đổi BGR sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Phát hiện face mesh
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            if draw:
                cv2.putText(image, "KHONG TIM THAY MAT", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return False, None
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Iris indices (từ Face Mesh với refine_landmarks=True)
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        LEFT_EYE = [33, 133]   # outer, inner
        RIGHT_EYE = [362, 263]  # outer, inner
        
        # Chuyển đổi sang tọa độ pixel
        mesh_points = np.array(
            [[p.x * w, p.y * h] for p in face_landmarks.landmark]
        )
        
        # Lấy điểm mắt và iris
        left_eye_pts = mesh_points[LEFT_EYE]
        right_eye_pts = mesh_points[RIGHT_EYE]
        left_iris = mesh_points[LEFT_IRIS].mean(axis=0)
        right_iris = mesh_points[RIGHT_IRIS].mean(axis=0)
        
        # Tính vị trí iris theo chiều ngang (X)
        def iris_pos_x(eye_points, iris_center):
            x1 = eye_points[0][0]
            x2 = eye_points[1][0]
            cx = iris_center[0]
            eye_width = abs(x2 - x1) if abs(x2 - x1) > 0 else 1
            # Chuẩn hóa: 0 = nhìn trái, 0.5 = nhìn thẳng, 1 = nhìn phải
            pos = (cx - min(x1, x2)) / eye_width
            return pos
        
        left_x = iris_pos_x(left_eye_pts, left_iris)
        right_x = iris_pos_x(right_eye_pts, right_iris)
        avg_x = (left_x + right_x) / 2
        
        # Ngưỡng mở rộng: 0.30 - 0.70 (dễ nhận hơn)
        is_looking = 0.30 < avg_x < 0.70
        
        # Vẽ thông tin lên ảnh nếu yêu cầu
        if draw:
            # Vẽ iris (chấm vàng)
            cv2.circle(image, tuple(left_iris.astype(int)), 4, (0, 255, 255), -1)
            cv2.circle(image, tuple(right_iris.astype(int)), 4, (0, 255, 255), -1)
            
            # Vẽ đường viền mắt
            cv2.line(image, tuple(left_eye_pts[0].astype(int)), 
                    tuple(left_eye_pts[1].astype(int)), (255, 255, 0), 1)
            cv2.line(image, tuple(right_eye_pts[0].astype(int)), 
                    tuple(right_eye_pts[1].astype(int)), (255, 255, 0), 1)
            
            # Text trạng thái bằng tiếng Việt
            if is_looking:
                text = "DANG NHIN CAMERA"
                color = (0, 255, 0)
            else:
                if avg_x < 0.30:
                    text = "NHIN SANG PHAI"
                elif avg_x > 0.70:
                    text = "NHIN SANG TRAI"
                else:
                    text = "KHONG NHIN CAMERA"
                color = (0, 0, 255)
            
            cv2.putText(image, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Debug: hiển thị giá trị iris position
            cv2.putText(image, f"Iris X: {avg_x:.2f} (0.30-0.70 = OK)", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        debug_info = {
            'left_x': left_x,
            'right_x': right_x,
            'avg_x': avg_x,
            'is_looking': is_looking
        }
        
        return is_looking, debug_info
    
    def close(self):
        """
        Đóng và giải phóng tài nguyên
        """
        self.hands.close()
        if self.face_detection is not None:
            self.face_detection.close()
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()


if __name__ == "__main__":
    # Test module
    print("Testing Hand Detector...")
    print("Phím tắt:")
    print("  'q' - Thoát")
    print("  '1' - Test cử chỉ tĩnh")
    print("  '2' - Test vuốt (swipe)")
    print("  '3' - Test vẫy tay")
    print("  '4' - Test zoom")
    
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    test_mode = 1  # 1=static, 2=swipe, 3=wave, 4=zoom
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Flip để dễ điều khiển
        img = cv2.flip(img, 1)
        
        # Phát hiện bàn tay
        img, results = detector.find_hands(img)
        
        # Lấy landmarks
        landmarks_list = detector.get_landmarks(img, results)
        
        if landmarks_list:
            if test_mode == 1:
                # Test cử chỉ tĩnh
                gesture = detector.detect_gesture(landmarks_list)
                fingers = detector.fingers_up(landmarks_list)
                orientation = detector.get_hand_orientation(landmarks_list)
                
                cv2.putText(img, f"Gesture: {gesture}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Fingers: {sum(fingers)}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Orient: {orientation}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            elif test_mode == 2:
                # Test vuốt
                is_swipe, direction = detector.is_swiping(landmarks_list)
                swipe_gesture = detector.detect_swipe_gesture(landmarks_list)
                
                color = (0, 255, 0) if is_swipe else (0, 0, 255)
                cv2.putText(img, f"Swipe: {is_swipe}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"Direction: {direction}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"Gesture: {swipe_gesture}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
            elif test_mode == 3:
                # Test vẫy tay
                is_wave = detector.is_waving()
                
                color = (0, 255, 0) if is_wave else (0, 0, 255)
                cv2.putText(img, f"Waving: {is_wave}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"History: {len(detector.landmarks_history)}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
            elif test_mode == 4:
                # Test zoom
                is_zoom, zoom_type = detector.detect_zoom_gesture(landmarks_list)
                pinch, distance = detector.is_pinching(landmarks_list)
                
                color = (0, 255, 0) if is_zoom else (0, 0, 255)
                cv2.putText(img, f"Zoom: {is_zoom} - {zoom_type}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"Pinch: {pinch} ({distance:.1f})", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Hiển thị mode
        cv2.putText(img, f"Mode: {test_mode} (Press 1-4)", (10, img.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Hand Detector Test", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            test_mode = 1
            print("Mode: Cử chỉ tĩnh")
        elif key == ord('2'):
            test_mode = 2
            print("Mode: Vuốt (Swipe)")
        elif key == ord('3'):
            test_mode = 3
            print("Mode: Vẫy tay")
        elif key == ord('4'):
            test_mode = 4
            print("Mode: Zoom")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
