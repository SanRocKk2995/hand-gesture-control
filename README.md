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

<div align="center">

⭐ **Nếu dự án hữu ích, hãy cho một star!** ⭐

</div>
