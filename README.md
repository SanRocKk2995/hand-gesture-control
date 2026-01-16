# 🖐️ Hand Gesture Control

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.x-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Điều khiển máy tính bằng cử chỉ tay sử dụng Computer Vision và Machine Learning**

</div>

---

## 📖 Giới thiệu

**Hand Gesture Control** là ứng dụng cho phép bạn điều khiển máy tính hoàn toàn bằng cử chỉ tay thông qua webcam. Sử dụng công nghệ MediaPipe của Google để nhận diện bàn tay và các ngón tay theo thời gian thực.

### ✨ Tính năng chính

- 🖱️ **Virtual Mouse** - Di chuyển chuột bằng ngón tay
- 👆 **Click & Drag** - Click, double-click và kéo thả
- ⌨️ **Keyboard Shortcuts** - Gán phím tắt cho từng cử chỉ
- 🎮 **Gaming Support** - Hỗ trợ điều khiển game
- ⚡ **Real-time** - Xử lý theo thời gian thực với FPS cao
- 🎨 **User-friendly GUI** - Giao diện đồ họa thân thiện

---

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- Webcam
- Windows 10/11 (khuyến nghị)

### Cài đặt dependencies

```bash
pip install opencv-python mediapipe pyautogui numpy PyQt6
```

### Clone repository

```bash
git clone https://github.com/your-username/hand-gesture-control.git
cd hand-gesture-control
```

---

## 📁 Cấu trúc dự án

```
hand-gesture-control/
├── src/
│   ├── app_optimized.py      # 🎯 Ứng dụng chính với GUI
│   ├── mouse_control.py      # 🖱️ Virtual Mouse độc lập
│   ├── hand_detector.py      # 👋 Module phát hiện bàn tay
│   ├── optimized_recognizer.py # 🧠 Nhận diện cử chỉ
│   ├── command_mapper.py     # ⌨️ Ánh xạ cử chỉ → lệnh
│   ├── collect_data.py       # 📊 Thu thập dữ liệu training
│   └── train_model.py        # 🤖 Huấn luyện model
├── utils/
│   ├── preprocessing.py      # Tiền xử lý dữ liệu
│   └── visualization.py      # Hiển thị kết quả
├── models/                   # Thư mục chứa trained models
├── data/                     # Dữ liệu training
├── docs/                     # Tài liệu
├── gesture_config.json       # ⚙️ Cấu hình cử chỉ
└── README.md
```

---

## 🎮 Hướng dẫn sử dụng

### 1️⃣ Virtual Mouse (Khuyên dùng cho người mới)

Chạy file điều khiển chuột đơn giản:

```bash
python src/mouse_control.py
```

#### Các cử chỉ:

| Cử chỉ | Hành động |
|--------|-----------|
| ☝️ **1 ngón (trỏ)** | Di chuyển chuột |
| ✌️ **2 ngón + chụm** | Click trái |
| 👌 **OK sign** | Double click |
| 🖐️ **Xòe 5 ngón** | Click phải |
| ✊ **Nắm tay** | Kéo thả (Drag) |

#### Phím tắt:
- `Q` - Thoát
- `+` / `-` - Tăng/giảm độ mượt
- `R` - Reset vị trí chuột

---

### 2️⃣ Ứng dụng đầy đủ với GUI

Chạy ứng dụng chính với giao diện đồ họa:

```bash
python src/app_optimized.py
```

#### Tính năng GUI:
- 📹 Xem camera trực tiếp
- ⚙️ Cấu hình cử chỉ tùy chỉnh
- 🎚️ Điều chỉnh độ nhạy
- 📊 Hiển thị FPS và trạng thái
- 🔧 Chế độ nhà phát triển

---

## 🤚 Các cử chỉ được hỗ trợ

| Cử chỉ | Mô tả | Mặc định |
|--------|-------|----------|
| ✊ Fist | Nắm tay | Space |
| 🖐️ Open Palm | Xòe bàn tay | - |
| ☝️ Pointing | Chỉ 1 ngón | - |
| ✌️ Peace | Chữ V | - |
| 👍 Thumbs Up | Like | - |
| 👎 Thumbs Down | Dislike | - |
| 👌 OK | Ngón cái + trỏ | - |
| 🤘 Rock | Rock sign | - |
| 3️⃣ Three | 3 ngón | - |
| 4️⃣ Four | 4 ngón | - |
| 📞 Call | Điện thoại | - |
| ⬆️ Swipe Up | Vuốt lên | - |
| ⬇️ Swipe Down | Vuốt xuống | - |
| ⬅️ Swipe Left | Vuốt trái | - |
| ➡️ Swipe Right | Vuốt phải | - |

---

## ⚙️ Cấu hình

Chỉnh sửa file `gesture_config.json` để tùy chỉnh:

```json
{
  "gestures": {
    "fist": {
      "action": "space",
      "enabled": true
    },
    "peace": {
      "action": "ctrl+c",
      "enabled": true
    }
  }
}
```

---

## 🛠️ Tối ưu hiệu năng

Ứng dụng được tối ưu với các thuật toán:

- **Adaptive Resolution** - Tự động điều chỉnh độ phân giải theo RAM
- **Adaptive Frame Skip** - Bỏ qua frame thông minh khi không cần
- **Temporal Caching** - Cache kết quả nhận diện giữa các frame
- **Smart GC** - Garbage Collection thông minh
- **Object Pooling** - Tái sử dụng bộ nhớ

---

## 🔧 Build EXE

Tạo file thực thi để chạy độc lập:

```bash
# Sử dụng PyInstaller
pyinstaller HandGestureControl.spec

# Hoặc với debug
pyinstaller HandGestureControl_Debug.spec
```

File EXE sẽ được tạo trong thư mục `dist/`.

---

## 📝 Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| CPU | Dual-core 2GHz | Quad-core 3GHz+ |
| RAM | 4GB | 8GB+ |
| Webcam | 480p | 720p+ |
| GPU | Không bắt buộc | Có thì tốt hơn |

---

## 🐛 Xử lý lỗi thường gặp

### Camera không hoạt động
```bash
# Kiểm tra camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### MediaPipe lỗi
```bash
pip install --upgrade mediapipe
```

### PyAutoGUI không hoạt động
```bash
# Chạy với quyền Admin trên Windows
```

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Tác giả

**Hand Gesture Control Team**

---

## 🙏 Cảm ơn

- [MediaPipe](https://mediapipe.dev/) - Google's ML solutions
- [OpenCV](https://opencv.org/) - Computer Vision library
- [PyAutoGUI](https://pyautogui.readthedocs.io/) - GUI automation

---
# 📊 BÁO CÁO CÔNG NGHỆ SỬ DỤNG
## Dự án: Hand Gesture Control - Điều khiển máy tính bằng cử chỉ tay

---

## 📋 Mục lục
1. [Tổng quan công nghệ](#1-tổng-quan-công-nghệ)
2. [Ngôn ngữ lập trình](#2-ngôn-ngữ-lập-trình)
3. [Thư viện Computer Vision](#3-thư-viện-computer-vision)
4. [Machine Learning Framework](#4-machine-learning-framework)
5. [GUI Framework](#5-gui-framework)
6. [Các thư viện hỗ trợ](#6-các-thư-viện-hỗ-trợ)
7. [Kiến trúc hệ thống](#7-kiến-trúc-hệ-thống)
8. [So sánh công nghệ](#8-so-sánh-công-nghệ)

---

## 1. Tổng quan công nghệ

| Thành phần | Công nghệ | Phiên bản | Mục đích |
|------------|-----------|-----------|----------|
| Ngôn ngữ | Python | 3.8+ | Ngôn ngữ chính |
| Computer Vision | OpenCV | 4.x | Xử lý hình ảnh |
| Hand Tracking | MediaPipe | Latest | Nhận diện bàn tay |
| GUI | PyQt6 | 6.x | Giao diện người dùng |
| Automation | PyAutoGUI | Latest | Điều khiển chuột/bàn phím |
| Tính toán | NumPy | Latest | Xử lý mảng số |

---

## 2. Ngôn ngữ lập trình

### 🐍 Python 3.8+

| Đặc điểm | Mô tả |
|----------|-------|
| **Loại** | Ngôn ngữ thông dịch, đa mục đích |
| **Paradigm** | OOP, Functional, Procedural |
| **Typing** | Dynamic typing |
| **Ưu điểm** | Dễ học, hệ sinh thái ML/AI phong phú |
| **Nhược điểm** | Chậm hơn compiled languages |

#### Lý do chọn Python:
- ✅ Hệ sinh thái Machine Learning và Computer Vision mạnh mẽ
- ✅ Cú pháp đơn giản, dễ đọc và bảo trì
- ✅ Cộng đồng lớn, tài liệu phong phú
- ✅ Tích hợp tốt với MediaPipe và OpenCV
- ✅ Cross-platform (Windows, macOS, Linux)

---

## 3. Thư viện Computer Vision

### 📷 OpenCV (Open Source Computer Vision Library)

| Thông tin | Chi tiết |
|-----------|----------|
| **Website** | https://opencv.org |
| **License** | Apache 2.0 |
| **Ngôn ngữ gốc** | C++ |
| **Python binding** | opencv-python |

#### Chức năng sử dụng trong dự án:

```python
import cv2

# 1. Đọc camera
cap = cv2.VideoCapture(0)

# 2. Xử lý frame
img = cv2.flip(img, 1)  # Lật ảnh
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển màu

# 3. Vẽ UI
cv2.circle(img, (x, y), radius, color, thickness)
cv2.rectangle(img, pt1, pt2, color, thickness)
cv2.putText(img, text, pos, font, scale, color, thickness)

# 4. Hiển thị
cv2.imshow("Window", img)
```

#### Ưu điểm OpenCV:
| Ưu điểm | Mô tả |
|---------|-------|
| Tốc độ | Tối ưu C++, rất nhanh |
| Đa nền tảng | Windows, Linux, macOS, Android, iOS |
| Cộng đồng | Lớn nhất trong lĩnh vực CV |
| Tài liệu | Phong phú, nhiều tutorial |

---

## 4. Machine Learning Framework

### 🤖 MediaPipe (Google)

| Thông tin | Chi tiết |
|-----------|----------|
| **Nhà phát triển** | Google |
| **Website** | https://mediapipe.dev |
| **License** | Apache 2.0 |
| **Đặc điểm** | On-device ML, real-time |

#### MediaPipe Hands - Chi tiết kỹ thuật:

| Thông số | Giá trị |
|----------|---------|
| **Số điểm landmark** | 21 điểm trên mỗi bàn tay |
| **Tốc độ** | 30+ FPS trên CPU |
| **Độ chính xác** | ~95.7% |
| **Model size** | ~3MB |

#### 21 Hand Landmarks:

```
WRIST = 0
THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4
INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8
MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12
RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16
PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20
```

#### Pipeline xử lý:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│   Camera    │ -> │ Palm Detect  │ -> │ Hand Track  │ -> │ Landmark │
│   Input     │    │   Model      │    │   Model     │    │  Output  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
```

#### Cách sử dụng trong dự án:

```python
import mediapipe as mp

# Khởi tạo
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,      # Video mode
    max_num_hands=1,              # Tối đa 1 tay
    min_detection_confidence=0.7, # Ngưỡng phát hiện
    min_tracking_confidence=0.5   # Ngưỡng theo dõi
)

# Xử lý
results = hands.process(rgb_image)
if results.multi_hand_landmarks:
    for hand in results.multi_hand_landmarks:
        for id, landmark in enumerate(hand.landmark):
            x, y = landmark.x, landmark.y  # Tọa độ normalized [0,1]
```

#### So sánh với các giải pháp khác:

| Framework | Tốc độ | Độ chính xác | Dễ sử dụng | GPU cần? |
|-----------|--------|--------------|------------|----------|
| **MediaPipe** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Không |
| OpenPose | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Có |
| Detectron2 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Có |
| YOLO | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Khuyến nghị |

---

## 5. GUI Framework

### 🖼️ PyQt6

| Thông tin | Chi tiết |
|-----------|----------|
| **Dựa trên** | Qt 6 (C++) |
| **License** | GPL v3 / Commercial |
| **Binding** | Python |
| **Widgets** | 100+ built-in widgets |

#### Các widget sử dụng:

```python
from PyQt6.QtWidgets import (
    QApplication,      # Ứng dụng chính
    QMainWindow,       # Cửa sổ chính
    QWidget,           # Widget cơ bản
    QVBoxLayout,       # Layout dọc
    QHBoxLayout,       # Layout ngang
    QPushButton,       # Nút bấm
    QLabel,            # Nhãn text/ảnh
    QSlider,           # Thanh trượt
    QCheckBox,         # Checkbox
    QComboBox,         # Dropdown
    QTabWidget,        # Tab
    QSystemTrayIcon,   # Icon system tray
)
```

#### Ưu điểm PyQt6:

| Ưu điểm | Mô tả |
|---------|-------|
| Native look | Giao diện đẹp, giống app native |
| Cross-platform | Windows, macOS, Linux |
| Signals/Slots | Event handling mạnh mẽ |
| Rich widgets | Nhiều widget có sẵn |
| Documentation | Tài liệu đầy đủ |

#### So sánh GUI frameworks:

| Framework | Native Look | Performance | Learning Curve | Features |
|-----------|-------------|-------------|----------------|----------|
| **PyQt6** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Tkinter | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| wxPython | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Kivy | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 6. Các thư viện hỗ trợ

### 🖱️ PyAutoGUI

| Chức năng | Code |
|-----------|------|
| Di chuyển chuột | `pyautogui.moveTo(x, y)` |
| Click | `pyautogui.click()` |
| Double click | `pyautogui.doubleClick()` |
| Right click | `pyautogui.rightClick()` |
| Kéo thả | `pyautogui.mouseDown()` / `mouseUp()` |
| Gõ phím | `pyautogui.press('space')` |
| Tổ hợp phím | `pyautogui.hotkey('ctrl', 'c')` |

### 🔢 NumPy

| Chức năng | Sử dụng |
|-----------|---------|
| Tính khoảng cách | `np.hypot(dx, dy)` |
| Nội suy | `np.interp(x, [a,b], [c,d])` |
| Giới hạn giá trị | `np.clip(x, min, max)` |
| Tạo mảng | `np.array([...])` |

---

## 7. Kiến trúc hệ thống

### 📐 Sơ đồ tổng quan:

```
┌────────────────────────────────────────────────────────────────┐
│                        APPLICATION                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐  │
│   │   Camera    │    │   MediaPipe   │    │    Gesture     │  │
│   │   Input     │ -> │   Hands       │ -> │   Recognition  │  │
│   │  (OpenCV)   │    │  (21 points)  │    │                │  │
│   └─────────────┘    └──────────────┘    └────────┬────────┘  │
│                                                    │          │
│   ┌─────────────┐    ┌──────────────┐    ┌────────▼────────┐  │
│   │   Display   │    │   PyQt6      │    │   Command       │  │
│   │   Output    │ <- │   GUI        │ <- │   Mapper        │  │
│   │  (OpenCV)   │    │              │    │  (PyAutoGUI)    │  │
│   └─────────────┘    └──────────────┘    └─────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 📦 Module Dependencies:

```
app_optimized.py
    ├── hand_detector.py
    │       └── mediapipe
    │       └── opencv
    │       └── numpy
    ├── optimized_recognizer.py
    │       └── numpy
    ├── command_mapper.py
    │       └── pyautogui
    └── PyQt6 (GUI)

mouse_control.py (Standalone)
    ├── mediapipe
    ├── opencv
    ├── pyautogui
    └── numpy
```

---

## 8. So sánh công nghệ

### 📊 Bảng so sánh tổng hợp:

| Tiêu chí | Lựa chọn hiện tại | Thay thế | Lý do chọn |
|----------|-------------------|----------|------------|
| **Hand Tracking** | MediaPipe | OpenPose, YOLO | Nhanh, không cần GPU, dễ dùng |
| **Video Capture** | OpenCV | Pygame, Pillow | Chuẩn công nghiệp, tốc độ cao |
| **GUI** | PyQt6 | Tkinter, wxPython | Đẹp, nhiều tính năng |
| **Mouse Control** | PyAutoGUI | pynput, ctypes | Cross-platform, API đơn giản |
| **Math** | NumPy | Pure Python | Tốc độ, vectorization |

### ⚡ Performance Metrics:

| Metric | Giá trị | Đơn vị |
|--------|---------|--------|
| FPS (640x480) | 25-30 | frames/s |
| Latency | 30-50 | ms |
| CPU Usage | 15-25 | % |
| RAM Usage | 150-250 | MB |
| GPU Usage | 0-5 | % (không bắt buộc) |

### 🔮 Công nghệ tiềm năng cho tương lai:

| Công nghệ | Mục đích | Khả năng tích hợp |
|-----------|----------|-------------------|
| TensorFlow Lite | Model optimization | Cao |
| ONNX Runtime | Cross-platform inference | Cao |
| WebRTC | Web integration | Trung bình |
| CUDA | GPU acceleration | Trung bình |

---

## 📚 Tài liệu tham khảo

1. **MediaPipe Documentation** - https://google.github.io/mediapipe/
2. **OpenCV Documentation** - https://docs.opencv.org/
3. **PyQt6 Documentation** - https://www.riverbankcomputing.com/static/Docs/PyQt6/
4. **PyAutoGUI Documentation** - https://pyautogui.readthedocs.io/
5. **NumPy Documentation** - https://numpy.org/doc/

---

## 📝 Kết luận

Dự án Hand Gesture Control sử dụng stack công nghệ hiện đại và phù hợp:

| Điểm mạnh | Điểm cần cải thiện |
|-----------|-------------------|
| ✅ Chạy trên CPU, không cần GPU | ⚠️ Có thể tối ưu thêm với GPU |
| ✅ Real-time performance | ⚠️ Phụ thuộc vào ánh sáng |
| ✅ Cross-platform | ⚠️ Chưa có mobile version |
| ✅ Dễ mở rộng | ⚠️ Model chưa customizable |
| ✅ Open-source | |

---

<div align="center">

**Báo cáo được tạo: Tháng 01/2026**

</div>


<div align="center">

⭐ **Nếu dự án hữu ích, hãy cho một star!** ⭐

</div>
