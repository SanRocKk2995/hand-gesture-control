"""
Module Preprocessing - Tiền xử lý hình ảnh
Các hàm hỗ trợ xử lý ảnh trước khi đưa vào phát hiện bàn tay
"""

import cv2
import numpy as np


def resize_image(image: np.ndarray, 
                 width: int = 640, 
                 height: int = 480) -> np.ndarray:
    """
    Resize ảnh về kích thước chuẩn
    
    Args:
        image: Ảnh đầu vào
        width: Chiều rộng mong muốn
        height: Chiều cao mong muốn
        
    Returns:
        Ảnh đã resize
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def flip_image(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    """
    Lật ảnh (mirror effect cho webcam)
    
    Args:
        image: Ảnh đầu vào
        horizontal: Lật theo chiều ngang
        
    Returns:
        Ảnh đã lật
    """
    if horizontal:
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)


def adjust_brightness_contrast(image: np.ndarray,
                               brightness: int = 0,
                               contrast: int = 0) -> np.ndarray:
    """
    Điều chỉnh độ sáng và độ tương phản
    
    Args:
        image: Ảnh đầu vào
        brightness: Giá trị brightness (-100 đến 100)
        contrast: Giá trị contrast (-100 đến 100)
        
    Returns:
        Ảnh đã điều chỉnh
    """
    # Điều chỉnh brightness
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    # Điều chỉnh contrast
    if contrast != 0:
        alpha_c = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma_c = 127 * (1 - alpha_c)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image


def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Giảm nhiễu sử dụng Gaussian Blur
    
    Args:
        image: Ảnh đầu vào
        kernel_size: Kích thước kernel (số lẻ)
        
    Returns:
        Ảnh đã giảm nhiễu
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh sang grayscale
    
    Args:
        image: Ảnh đầu vào (BGR)
        
    Returns:
        Ảnh grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Tăng cường contrast sử dụng histogram equalization
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã tăng cường contrast
    """
    # Chuyển sang YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Equalize histogram trên kênh Y
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    
    # Chuyển lại BGR
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def crop_region(image: np.ndarray, 
                x: int, y: int, 
                width: int, height: int) -> np.ndarray:
    """
    Cắt vùng quan tâm (ROI) từ ảnh
    
    Args:
        image: Ảnh đầu vào
        x, y: Tọa độ góc trên trái
        width, height: Kích thước vùng cắt
        
    Returns:
        Ảnh đã cắt
    """
    return image[y:y+height, x:x+width]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa giá trị pixel về [0, 1]
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã chuẩn hóa
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh từ [0, 1] về [0, 255]
    
    Args:
        image: Ảnh đã chuẩn hóa
        
    Returns:
        Ảnh uint8
    """
    return (image * 255).astype(np.uint8)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Áp dụng adaptive threshold
    
    Args:
        image: Ảnh grayscale
        
    Returns:
        Ảnh binary
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )


class ImagePreprocessor:
    """
    Lớp tiền xử lý ảnh với pipeline tùy chỉnh
    """
    
    def __init__(self, 
                 target_size: tuple = (640, 480),
                 flip_horizontal: bool = True,
                 enhance: bool = False):
        """
        Khởi tạo preprocessor
        
        Args:
            target_size: Kích thước output (width, height)
            flip_horizontal: Lật ảnh theo chiều ngang
            enhance: Có tăng cường contrast không
        """
        self.target_size = target_size
        self.flip_horizontal = flip_horizontal
        self.enhance = enhance
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Xử lý ảnh qua pipeline
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh đã xử lý
        """
        # Flip
        if self.flip_horizontal:
            image = flip_image(image, horizontal=True)
        
        # Resize
        if self.target_size is not None:
            image = resize_image(image, 
                               width=self.target_size[0],
                               height=self.target_size[1])
        
        # Enhance contrast
        if self.enhance:
            image = enhance_contrast(image)
        
        return image


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing functions...")
    
    # Tạo ảnh test
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test các functions
    resized = resize_image(test_image, 320, 240)
    print(f"Resized shape: {resized.shape}")
    
    flipped = flip_image(test_image)
    print(f"Flipped shape: {flipped.shape}")
    
    gray = convert_to_grayscale(test_image)
    print(f"Grayscale shape: {gray.shape}")
    
    # Test preprocessor
    preprocessor = ImagePreprocessor()
    processed = preprocessor.process(test_image)
    print(f"Processed shape: {processed.shape}")
    
    print("\nPreprocessing tests completed!")
