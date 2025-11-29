"""
Script thu thập dữ liệu để train model
Sử dụng để tạo dataset các cử chỉ tay
"""

import cv2
import numpy as np
import os
import json
import time
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier


class DataCollector:
    """
    Lớp thu thập dữ liệu cử chỉ tay
    """
    
    def __init__(self, 
                 output_dir: str = "../data/gestures",
                 camera_id: int = 0):
        """
        Khởi tạo Data Collector
        
        Args:
            output_dir: Thư mục lưu dữ liệu
            camera_id: ID camera
        """
        self.output_dir = output_dir
        self.camera_id = camera_id
        
        # Tạo thư mục nếu chưa có
        os.makedirs(output_dir, exist_ok=True)
        
        # Khởi tạo detector
        self.detector = HandDetector()
        
        # Khởi tạo camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Danh sách các cử chỉ cần thu thập
        self.gestures = GestureClassifier.GESTURES
        self.current_gesture_id = 0
        
        # Data storage
        self.collected_data = []
        self.samples_per_gesture = 100
        self.current_samples = 0
        
        print("Data Collector initialized")
        print(f"Output directory: {output_dir}")
        print(f"Gestures to collect: {list(self.gestures.values())}")
    
    def collect_gesture_samples(self, gesture_id: int, num_samples: int = 100):
        """
        Thu thập samples cho một cử chỉ
        
        Args:
            gesture_id: ID của cử chỉ
            num_samples: Số samples cần thu thập
        """
        gesture_name = self.gestures[gesture_id]
        print(f"\n{'='*60}")
        print(f"Thu thập dữ liệu cho cử chỉ: {gesture_name}")
        print(f"Cần thu thập: {num_samples} samples")
        print(f"{'='*60}")
        print("\nHãy chuẩn bị cử chỉ và nhấn SPACE để bắt đầu...")
        
        collected = 0
        collecting = False
        
        while collected < num_samples:
            success, frame = self.cap.read()
            if not success:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Phát hiện tay
            frame, results = self.detector.find_hands(frame, draw=True)
            
            # Vẽ thông tin
            info_text = f"Gesture: {gesture_name} ({gesture_id})"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            progress_text = f"Progress: {collected}/{num_samples}"
            cv2.putText(frame, progress_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if collecting:
                status = "COLLECTING..."
                color = (0, 0, 255)
            else:
                status = "Press SPACE to start"
                color = (255, 255, 0)
            
            cv2.putText(frame, status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Thu thập data nếu đang collecting
            if collecting:
                landmarks = self.detector.get_normalized_landmarks(results)
                
                if landmarks is not None:
                    # Preprocess landmarks
                    classifier = GestureClassifier()
                    features = classifier.preprocess_landmarks(landmarks)
                    
                    if features is not None:
                        # Lưu data
                        self.collected_data.append({
                            'features': features.tolist(),
                            'label': gesture_id,
                            'gesture_name': gesture_name
                        })
                        
                        collected += 1
                        
                        # Vẽ visual feedback
                        cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2),
                                 20, (0, 255, 0), -1)
            
            cv2.imshow("Data Collection", frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                collecting = not collecting
                if collecting:
                    print("Bắt đầu thu thập...")
                else:
                    print("Tạm dừng thu thập")
            elif key == ord('q'):
                print("\nDừng thu thập sớm")
                break
            elif key == ord('s'):
                # Skip gesture
                print(f"\nBỏ qua cử chỉ {gesture_name}")
                return False
        
        print(f"✓ Đã thu thập {collected} samples cho {gesture_name}")
        return True
    
    def run_collection(self):
        """
        Chạy quá trình thu thập cho tất cả cử chỉ
        """
        print("\n" + "="*60)
        print("BẮT ĐẦU THU THẬP DỮ LIỆU")
        print("="*60)
        print("\nHướng dẫn:")
        print("- Nhấn SPACE để bắt đầu/dừng thu thập")
        print("- Nhấn 's' để bỏ qua cử chỉ hiện tại")
        print("- Nhấn 'q' để thoát")
        print("="*60)
        
        for gesture_id, gesture_name in self.gestures.items():
            success = self.collect_gesture_samples(gesture_id, self.samples_per_gesture)
            
            if not success:
                continue
            
            # Nghỉ giữa các cử chỉ
            if gesture_id < len(self.gestures) - 1:
                print("\nNghỉ 3 giây trước khi chuyển sang cử chỉ tiếp theo...")
                time.sleep(3)
        
        # Lưu dữ liệu
        self.save_data()
    
    def save_data(self):
        """
        Lưu dữ liệu đã thu thập
        """
        if not self.collected_data:
            print("Không có dữ liệu để lưu!")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"gesture_data_{timestamp}.json")
        
        # Tạo metadata
        data_dict = {
            'metadata': {
                'num_samples': len(self.collected_data),
                'num_gestures': len(self.gestures),
                'gestures': self.gestures,
                'timestamp': timestamp
            },
            'data': self.collected_data
        }
        
        # Lưu file
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Đã lưu {len(self.collected_data)} samples vào {filename}")
        print(f"{'='*60}")
        
        # Thống kê
        print("\nThống kê:")
        gesture_counts = {}
        for item in self.collected_data:
            label = item['label']
            gesture_name = item['gesture_name']
            gesture_counts[gesture_name] = gesture_counts.get(gesture_name, 0) + 1
        
        for gesture_name, count in gesture_counts.items():
            print(f"  {gesture_name}: {count} samples")
    
    def cleanup(self):
        """
        Dọn dẹp tài nguyên
        """
        self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()


def main():
    """
    Hàm main
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect gesture data")
    parser.add_argument("--output", type=str, default="../data/gestures",
                       help="Output directory")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID")
    parser.add_argument("--samples", type=int, default=100,
                       help="Samples per gesture")
    
    args = parser.parse_args()
    
    # Tạo collector
    collector = DataCollector(
        output_dir=args.output,
        camera_id=args.camera
    )
    collector.samples_per_gesture = args.samples
    
    try:
        # Chạy thu thập
        collector.run_collection()
    except KeyboardInterrupt:
        print("\n\nThu thập bị ngắt bởi người dùng")
    except Exception as e:
        print(f"\n\nLỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
