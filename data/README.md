# Gesture Recognition Dataset

Thư mục này chứa dữ liệu cử chỉ tay để train model.

## Cấu trúc dữ liệu

Mỗi file JSON chứa:
- `metadata`: Thông tin về dataset
  - `num_samples`: Số lượng samples
  - `num_gestures`: Số lượng cử chỉ
  - `gestures`: Mapping ID -> tên cử chỉ
  - `timestamp`: Thời gian thu thập
  
- `data`: Mảng các samples
  - `features`: Vector đặc trưng (63 dimensions từ 21 landmarks x 3)
  - `label`: ID của cử chỉ
  - `gesture_name`: Tên cử chỉ

## Thu thập dữ liệu

```bash
python src/collect_data.py --output data/gestures --samples 100
```

## Train model

```bash
python src/train_model.py --data data/gestures --output models/gesture_model.pkl
```
