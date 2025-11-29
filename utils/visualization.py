"""
Module Visualization - Hiển thị kết quả và debug
Các hàm hỗ trợ vẽ và hiển thị thông tin
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class Colors:
    """Định nghĩa màu sắc"""
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    ORANGE = (0, 165, 255)
    PURPLE = (128, 0, 128)


def draw_text(image: np.ndarray,
             text: str,
             position: Tuple[int, int],
             font_scale: float = 1.0,
             color: Tuple[int, int, int] = Colors.WHITE,
             thickness: int = 2,
             with_background: bool = True) -> np.ndarray:
    """
    Vẽ text lên ảnh
    
    Args:
        image: Ảnh đầu vào
        text: Text cần vẽ
        position: Vị trí (x, y)
        font_scale: Kích thước font
        color: Màu text
        thickness: Độ dày
        with_background: Vẽ background cho text
        
    Returns:
        Ảnh đã vẽ text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if with_background:
        # Tính kích thước text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Vẽ background
        cv2.rectangle(
            image,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + baseline + 5),
            Colors.BLACK,
            -1
        )
    
    # Vẽ text
    cv2.putText(
        image, text, position, font,
        font_scale, color, thickness, cv2.LINE_AA
    )
    
    return image


def draw_info_panel(image: np.ndarray,
                   info_dict: dict,
                   position: Tuple[int, int] = (10, 30),
                   line_height: int = 30) -> np.ndarray:
    """
    Vẽ panel thông tin
    
    Args:
        image: Ảnh đầu vào
        info_dict: Dictionary chứa thông tin {key: value}
        position: Vị trí bắt đầu
        line_height: Khoảng cách giữa các dòng
        
    Returns:
        Ảnh đã vẽ panel
    """
    y = position[1]
    
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        draw_text(image, text, (position[0], y), 
                 font_scale=0.6, color=Colors.GREEN, thickness=2)
        y += line_height
    
    return image


def draw_fps(image: np.ndarray, 
            fps: float,
            position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Vẽ FPS lên ảnh
    
    Args:
        image: Ảnh đầu vào
        fps: Giá trị FPS
        position: Vị trí
        
    Returns:
        Ảnh đã vẽ FPS
    """
    text = f"FPS: {int(fps)}"
    draw_text(image, text, position, 
             font_scale=0.7, color=Colors.GREEN, thickness=2)
    return image


def draw_gesture_info(image: np.ndarray,
                     gesture_name: str,
                     confidence: float,
                     position: Tuple[int, int] = (10, 70)) -> np.ndarray:
    """
    Vẽ thông tin cử chỉ được nhận diện
    
    Args:
        image: Ảnh đầu vào
        gesture_name: Tên cử chỉ
        confidence: Độ tin cậy
        position: Vị trí
        
    Returns:
        Ảnh đã vẽ thông tin
    """
    # Vẽ tên cử chỉ
    text = f"Gesture: {gesture_name}"
    draw_text(image, text, position, 
             font_scale=0.8, color=Colors.YELLOW, thickness=2)
    
    # Vẽ confidence
    conf_text = f"Confidence: {confidence:.2f}"
    draw_text(image, conf_text, (position[0], position[1] + 35),
             font_scale=0.6, color=Colors.CYAN, thickness=2)
    
    return image


def draw_bounding_box(image: np.ndarray,
                     bbox: Tuple[int, int, int, int],
                     color: Tuple[int, int, int] = Colors.GREEN,
                     thickness: int = 2,
                     label: Optional[str] = None) -> np.ndarray:
    """
    Vẽ bounding box
    
    Args:
        image: Ảnh đầu vào
        bbox: Bounding box (x, y, w, h)
        color: Màu
        thickness: Độ dày
        label: Nhãn (optional)
        
    Returns:
        Ảnh đã vẽ bbox
    """
    x, y, w, h = bbox
    
    # Vẽ rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Vẽ label nếu có
    if label:
        draw_text(image, label, (x, y - 10),
                 font_scale=0.5, color=color, thickness=1)
    
    return image


def draw_landmarks_connections(image: np.ndarray,
                              landmarks: List[List[float]],
                              connections: List[Tuple[int, int]],
                              landmark_color: Tuple[int, int, int] = Colors.RED,
                              connection_color: Tuple[int, int, int] = Colors.GREEN) -> np.ndarray:
    """
    Vẽ landmarks và kết nối giữa chúng
    
    Args:
        image: Ảnh đầu vào
        landmarks: List landmarks [[id, x, y, z], ...]
        connections: List các kết nối [(id1, id2), ...]
        landmark_color: Màu landmarks
        connection_color: Màu kết nối
        
    Returns:
        Ảnh đã vẽ
    """
    if not landmarks:
        return image
    
    # Tạo dict để tra cứu nhanh
    landmark_dict = {lm[0]: (lm[1], lm[2]) for lm in landmarks}
    
    # Vẽ các kết nối
    for connection in connections:
        id1, id2 = connection
        if id1 in landmark_dict and id2 in landmark_dict:
            pt1 = landmark_dict[id1]
            pt2 = landmark_dict[id2]
            cv2.line(image, pt1, pt2, connection_color, 2)
    
    # Vẽ các landmarks
    for landmark in landmarks:
        id, x, y, z = landmark
        cv2.circle(image, (x, y), 5, landmark_color, -1)
    
    return image


def draw_control_zone(image: np.ndarray,
                     zone_rect: Tuple[int, int, int, int],
                     active: bool = False) -> np.ndarray:
    """
    Vẽ vùng điều khiển
    
    Args:
        image: Ảnh đầu vào
        zone_rect: Rectangle (x, y, w, h)
        active: Vùng có đang active không
        
    Returns:
        Ảnh đã vẽ
    """
    x, y, w, h = zone_rect
    color = Colors.GREEN if active else Colors.YELLOW
    
    # Vẽ khung vùng điều khiển
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Vẽ label
    label = "Active Zone" if active else "Control Zone"
    draw_text(image, label, (x + 10, y + 30),
             font_scale=0.7, color=color, thickness=2)
    
    return image


def create_blank_image(width: int = 640,
                      height: int = 480,
                      color: Tuple[int, int, int] = Colors.BLACK) -> np.ndarray:
    """
    Tạo ảnh trắng
    
    Args:
        width: Chiều rộng
        height: Chiều cao
        color: Màu nền
        
    Returns:
        Ảnh blank
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def draw_progress_bar(image: np.ndarray,
                     progress: float,
                     position: Tuple[int, int],
                     size: Tuple[int, int] = (200, 20),
                     color: Tuple[int, int, int] = Colors.GREEN) -> np.ndarray:
    """
    Vẽ progress bar
    
    Args:
        image: Ảnh đầu vào
        progress: Giá trị progress (0-1)
        position: Vị trí (x, y)
        size: Kích thước (w, h)
        color: Màu
        
    Returns:
        Ảnh đã vẽ
    """
    x, y = position
    w, h = size
    
    # Vẽ background
    cv2.rectangle(image, (x, y), (x + w, y + h), Colors.WHITE, 2)
    
    # Vẽ progress
    progress_w = int(w * max(0, min(1, progress)))
    cv2.rectangle(image, (x, y), (x + progress_w, y + h), color, -1)
    
    # Vẽ phần trăm
    percentage = int(progress * 100)
    text_pos = (x + w // 2 - 20, y + h // 2 + 5)
    draw_text(image, f"{percentage}%", text_pos,
             font_scale=0.4, color=Colors.BLACK, thickness=1,
             with_background=False)
    
    return image


class VisualizationManager:
    """
    Lớp quản lý visualization cho ứng dụng
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Khởi tạo Visualization Manager
        
        Args:
            frame_width: Chiều rộng frame
            frame_height: Chiều cao frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = 0
        self.frame_count = 0
        
    def draw_main_ui(self,
                    image: np.ndarray,
                    gesture_name: str = "None",
                    confidence: float = 0.0,
                    fps: float = 0.0,
                    additional_info: dict = None) -> np.ndarray:
        """
        Vẽ UI chính
        
        Args:
            image: Ảnh đầu vào
            gesture_name: Tên cử chỉ
            confidence: Độ tin cậy
            fps: FPS
            additional_info: Thông tin thêm
            
        Returns:
            Ảnh đã vẽ UI
        """
        # Vẽ FPS
        draw_fps(image, fps, position=(10, 30))
        
        # Vẽ gesture info
        draw_gesture_info(image, gesture_name, confidence, position=(10, 70))
        
        # Vẽ thông tin thêm nếu có
        if additional_info:
            y = 140
            for key, value in additional_info.items():
                text = f"{key}: {value}"
                draw_text(image, text, (10, y),
                         font_scale=0.5, color=Colors.CYAN, thickness=1)
                y += 25
        
        # Vẽ hướng dẫn
        help_text = "Press 'q' to quit | Press 'h' for help"
        draw_text(image, help_text, (10, self.frame_height - 20),
                 font_scale=0.5, color=Colors.WHITE, thickness=1)
        
        return image


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization functions...")
    
    # Tạo ảnh test
    image = create_blank_image(640, 480, Colors.BLACK)
    
    # Test các functions
    draw_text(image, "Test Text", (50, 50), color=Colors.GREEN)
    draw_fps(image, 30.5, (10, 30))
    draw_gesture_info(image, "Open", 0.95, (10, 100))
    draw_progress_bar(image, 0.75, (200, 200))
    
    # Hiển thị
    cv2.imshow("Visualization Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Visualization tests completed!")
