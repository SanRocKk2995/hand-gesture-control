# Trained Models

Thư mục này chứa các model đã được train.

## Sử dụng model

```bash
python src/main.py --use-ml --model models/gesture_model.pkl
```

## Các model có sẵn

- `gesture_model.pkl`: Random Forest classifier
- `gesture_model_svm.pkl`: SVM classifier (nếu có)

## Đánh giá model

Model được đánh giá dựa trên:
- **Training Accuracy**: Độ chính xác trên tập training
- **Validation Accuracy**: Độ chính xác trên tập validation
- **Classification Report**: Precision, Recall, F1-score cho từng cử chỉ
