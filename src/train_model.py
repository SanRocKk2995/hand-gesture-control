"""
Script train model từ dữ liệu đã thu thập
"""

import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from gesture_classifier import GestureClassifier


def load_data_from_json(json_path: str):
    """
    Load dữ liệu từ file JSON
    
    Args:
        json_path: Đường dẫn file JSON
        
    Returns:
        Tuple (X, y) - features và labels
    """
    print(f"Loading data from {json_path}...")
    
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    
    # Lấy metadata
    metadata = data_dict['metadata']
    print(f"Metadata:")
    print(f"  Number of samples: {metadata['num_samples']}")
    print(f"  Number of gestures: {metadata['num_gestures']}")
    print(f"  Timestamp: {metadata['timestamp']}")
    
    # Lấy data
    data = data_dict['data']
    
    X = []
    y = []
    
    for item in data:
        X.append(item['features'])
        y.append(item['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    return X, y, metadata


def load_multiple_datasets(data_dir: str):
    """
    Load nhiều file dữ liệu từ thư mục
    
    Args:
        data_dir: Thư mục chứa các file JSON
        
    Returns:
        Tuple (X, y) - combined features và labels
    """
    print(f"Loading all datasets from {data_dir}...")
    
    X_all = []
    y_all = []
    
    # Tìm tất cả file JSON
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    print(f"Found {len(json_files)} dataset files")
    
    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        X, y, _ = load_data_from_json(json_path)
        X_all.append(X)
        y_all.append(y)
    
    # Combine tất cả
    X_combined = np.vstack(X_all)
    y_combined = np.hstack(y_all)
    
    print(f"\nCombined data shape: X={X_combined.shape}, y={y_combined.shape}")
    
    return X_combined, y_combined


def train_model(data_path: str, 
                model_type: str = "random_forest",
                test_size: float = 0.2,
                output_path: str = "../models/gesture_model.pkl"):
    """
    Train model từ dữ liệu
    
    Args:
        data_path: Đường dẫn đến file/folder dữ liệu
        model_type: Loại model ("random_forest", "svm")
        test_size: Tỷ lệ test set
        output_path: Đường dẫn lưu model
    """
    print("=" * 60)
    print("TRAINING GESTURE RECOGNITION MODEL")
    print("=" * 60)
    
    # Load data
    if os.path.isfile(data_path):
        X, y, metadata = load_data_from_json(data_path)
    elif os.path.isdir(data_path):
        X, y = load_multiple_datasets(data_path)
        metadata = None
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    # Thống kê
    print("\nData statistics:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        gesture_name = GestureClassifier.GESTURES.get(label, "Unknown")
        print(f"  {gesture_name} (ID {label}): {count} samples")
    
    # Split data
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Khởi tạo và train model
    print(f"\nInitializing {model_type} model...")
    classifier = GestureClassifier(model_type=model_type)
    
    print("\nTraining model...")
    metrics = classifier.train(X_train, y_train, X_test, y_test)
    
    # Hiển thị kết quả
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
    
    # Lưu model
    print(f"\nSaving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classifier.save_model(output_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return classifier, metrics


def main():
    """
    Hàm main
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train gesture recognition model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data file or directory")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=["random_forest", "svm"],
                       help="Model type")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size ratio")
    parser.add_argument("--output", type=str, default="../models/gesture_model.pkl",
                       help="Output model path")
    
    args = parser.parse_args()
    
    try:
        # Train model
        train_model(
            data_path=args.data,
            model_type=args.model_type,
            test_size=args.test_size,
            output_path=args.output
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
