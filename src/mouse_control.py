"""
🖱️ Virtual Mouse - Điều khiển chuột bằng cử chỉ tay
Phiên bản: 2.0 - Dễ sử dụng nhất

CỬ CHỈ:
  ☝️  1 ngón (trỏ)         → Di chuyển chuột
  ✌️  2 ngón (trỏ+giữa)    → Di chuyển + Chụm = Click trái
  🖐️ 5 ngón (xòe bàn tay)  → Click phải
  ✊ Nắm tay               → Kéo thả (Drag)
  👌 OK (ngón cái+trỏ chạm) → Double click
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ==================== CẤU HÌNH ====================
class Config:
    # Camera
    CAMERA_ID = 0
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    
    # Vùng nhận diện (0 = toàn bộ camera)
    FRAME_MARGIN = 60
    
    # Độ mượt di chuyển (1=nhanh nhưng giật, 7=mượt nhưng chậm)
    SMOOTHING = 5
    
    # Ngưỡng click
    PINCH_THRESHOLD = 35      # Khoảng cách để click (pixels)
    CLICK_COOLDOWN = 0.4      # Thời gian chờ giữa các click (giây)
    DOUBLE_CLICK_SPEED = 0.25 # Tốc độ double click
    
    # MediaPipe
    DETECTION_CONF = 0.75
    TRACKING_CONF = 0.65


# ==================== KHỞI TẠO ====================
print("\n" + "="*60)
print("  🖱️  VIRTUAL MOUSE - ĐIỀU KHIỂN BẰNG CỬ CHỈ TAY")
print("="*60)

# PyAutoGUI setup
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()
print(f"  📺 Màn hình: {SCREEN_W} x {SCREEN_H}")

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=Config.DETECTION_CONF,
    min_tracking_confidence=Config.TRACKING_CONF
)

# Camera
cap = cv2.VideoCapture(Config.CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("  ❌ Không thể mở camera!")
    exit()
print(f"  📷 Camera: OK")

# ==================== TRẠNG THÁI ====================
class State:
    prev_x = SCREEN_W // 2
    prev_y = SCREEN_H // 2
    last_click = 0
    last_right_click = 0
    last_double_click = 0
    is_dragging = False
    drag_start_time = 0
    fps_time = time.time()
    fps = 0
    mode = "🔍 Đang tìm tay..."

# ==================== HÀM HỖ TRỢ ====================
def get_finger_states(lm):
    """Trả về trạng thái 5 ngón tay [thumb, index, middle, ring, pinky]"""
    if not lm or len(lm) < 21:
        return [0, 0, 0, 0, 0]
    
    fingers = []
    
    # Ngón cái: so sánh X
    fingers.append(1 if lm[4][0] < lm[3][0] - 15 else 0)
    
    # 4 ngón còn lại: đầu ngón cao hơn khớp
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(1 if lm[tip][1] < lm[pip][1] - 15 else 0)
    
    return fingers

def distance(p1, p2):
    """Khoảng cách 2 điểm"""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def smooth_move(new_x, new_y):
    """Di chuyển mượt với EMA"""
    alpha = 2.0 / (Config.SMOOTHING + 1)
    State.prev_x += alpha * (new_x - State.prev_x)
    State.prev_y += alpha * (new_y - State.prev_y)
    return int(State.prev_x), int(State.prev_y)

def map_coords(x, y, w, h):
    """Chuyển tọa độ camera → màn hình"""
    margin = Config.FRAME_MARGIN
    mx = np.interp(x, (margin, w - margin), (0, SCREEN_W))
    my = np.interp(y, (margin, h - margin), (0, SCREEN_H))
    return np.clip(mx, 0, SCREEN_W-1), np.clip(my, 0, SCREEN_H-1)

def draw_ui(img, h, w):
    """Vẽ giao diện"""
    # Header
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Mode với màu sắc
    mode_colors = {
        "MOVE": (0, 255, 100),
        "CLICK": (0, 200, 255),
        "RIGHT": (255, 100, 100),
        "DRAG": (255, 0, 255),
        "DOUBLE": (255, 255, 0),
    }
    color = (200, 200, 200)
    for key, c in mode_colors.items():
        if key in State.mode:
            color = c
            break
    
    cv2.putText(img, State.mode, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(img, f"FPS: {State.fps}", (w - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Hướng dẫn ở dưới
    guide = "1 ngon:Move | 2 ngon Chum:Click | 5 ngon:R-Click | Nam tay:Drag | Q:Thoat"
    cv2.putText(img, guide, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    # Vùng hoạt động
    m = Config.FRAME_MARGIN
    cv2.rectangle(img, (m, m + 80), (w - m, h - m), (100, 100, 255), 2)

def draw_cursor(img, x, y, mode="move"):
    """Vẽ cursor trên camera"""
    if mode == "move":
        cv2.circle(img, (x, y), 18, (0, 255, 100), 3)
        cv2.circle(img, (x, y), 6, (0, 255, 100), -1)
    elif mode == "click":
        cv2.circle(img, (x, y), 25, (0, 200, 255), -1)
        cv2.circle(img, (x, y), 25, (255, 255, 255), 3)
    elif mode == "drag":
        cv2.circle(img, (x, y), 20, (255, 0, 255), -1)
        cv2.circle(img, (x, y), 20, (255, 255, 255), 3)

# ==================== PRINT HƯỚNG DẪN ====================
print("-" * 60)
print("  📖 HƯỚNG DẪN SỬ DỤNG:")
print("  ─────────────────────")
print("  ☝️  1 ngón (trỏ)        → Di chuyển chuột")
print("  ✌️  2 ngón rồi chụm     → Click trái") 
print("  🖐️ Xòe 5 ngón          → Click phải")
print("  ✊ Nắm tay giữ         → Kéo thả (Drag)")
print("  👌 Ngón cái+trỏ chạm   → Double click")
print("-" * 60)
print("  ⌨️  PHÍM TẮT:")
print("  ─────────────")
print("  [Q] Thoát  |  [+/-] Tăng/giảm độ mượt  |  [R] Reset vị trí")
print("=" * 60 + "\n")

# ==================== MAIN LOOP ====================
try:
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        now = time.time()
        State.fps = int(1 / (now - State.fps_time + 0.001))
        State.fps_time = now
        
        State.mode = "🔍 Đang tìm tay..."
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            
            # Vẽ skeleton tay
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2))
            
            # Lấy landmarks
            lm = [[int(p.x * w), int(p.y * h)] for p in hand.landmark]
            fingers = get_finger_states(lm)
            total = sum(fingers)
            
            # Vị trí ngón trỏ
            idx_tip = lm[8]
            thumb_tip = lm[4]
            
            # ═══════════════════════════════════════════════
            # ☝️ CHẾ ĐỘ 1: DI CHUYỂN (chỉ ngón trỏ)
            # ═══════════════════════════════════════════════
            if fingers == [0, 1, 0, 0, 0]:
                State.mode = "☝️ MOVE - Di chuyển"
                
                # Kết thúc drag nếu đang kéo
                if State.is_dragging:
                    pyautogui.mouseUp()
                    State.is_dragging = False
                
                mx, my = map_coords(idx_tip[0], idx_tip[1], w, h)
                sx, sy = smooth_move(mx, my)
                pyautogui.moveTo(sx, sy)
                draw_cursor(img, idx_tip[0], idx_tip[1], "move")
            
            # ═══════════════════════════════════════════════
            # ✌️ CHẾ ĐỘ 2: CLICK TRÁI (2 ngón, chụm để click)
            # ═══════════════════════════════════════════════
            elif fingers[1] == 1 and fingers[2] == 1 and total == 2:
                mid_tip = lm[12]
                dist = distance(idx_tip, mid_tip)
                
                # Vẽ đường nối 2 ngón
                cv2.line(img, tuple(idx_tip), tuple(mid_tip), (0, 200, 255), 3)
                
                if dist < Config.PINCH_THRESHOLD:
                    State.mode = "👆 CLICK!"
                    draw_cursor(img, (idx_tip[0]+mid_tip[0])//2, (idx_tip[1]+mid_tip[1])//2, "click")
                    
                    if now - State.last_click > Config.CLICK_COOLDOWN:
                        pyautogui.click()
                        State.last_click = now
                else:
                    State.mode = f"✌️ Chụm để Click ({int(dist)}px)"
                    draw_cursor(img, idx_tip[0], idx_tip[1], "move")
                    
                    # Vẫn di chuyển chuột
                    mx, my = map_coords(idx_tip[0], idx_tip[1], w, h)
                    sx, sy = smooth_move(mx, my)
                    pyautogui.moveTo(sx, sy)
            
            # ═══════════════════════════════════════════════
            # 👌 CHẾ ĐỘ 3: DOUBLE CLICK (ngón cái chạm ngón trỏ)
            # ═══════════════════════════════════════════════
            elif fingers[0] == 1 and fingers[1] == 1 and distance(thumb_tip, idx_tip) < 30:
                State.mode = "👌 DOUBLE CLICK!"
                cv2.circle(img, tuple(idx_tip), 30, (0, 255, 255), -1)
                
                if now - State.last_double_click > Config.CLICK_COOLDOWN:
                    pyautogui.doubleClick()
                    State.last_double_click = now
            
            # ═══════════════════════════════════════════════
            # 🖐️ CHẾ ĐỘ 4: CLICK PHẢI (xòe 5 ngón)
            # ═══════════════════════════════════════════════
            elif total >= 4:
                State.mode = "🖐️ RIGHT CLICK"
                cv2.circle(img, tuple(lm[9]), 40, (255, 100, 100), 3)
                
                if now - State.last_right_click > Config.CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    State.last_right_click = now
            
            # ═══════════════════════════════════════════════
            # ✊ CHẾ ĐỘ 5: KÉO THẢ (nắm tay)
            # ═══════════════════════════════════════════════
            elif total == 0:
                palm = lm[9]  # Giữa lòng bàn tay
                
                if not State.is_dragging:
                    State.is_dragging = True
                    State.drag_start_time = now
                    pyautogui.mouseDown()
                
                State.mode = "✊ DRAG - Đang kéo"
                mx, my = map_coords(palm[0], palm[1], w, h)
                sx, sy = smooth_move(mx, my)
                pyautogui.moveTo(sx, sy)
                draw_cursor(img, palm[0], palm[1], "drag")
            
            else:
                # Trạng thái chuyển tiếp
                if State.is_dragging:
                    pyautogui.mouseUp()
                    State.is_dragging = False
                State.mode = f"🖐️ {total} ngón..."
        
        else:
            # Không thấy tay → thả drag
            if State.is_dragging:
                pyautogui.mouseUp()
                State.is_dragging = False
        
        # Vẽ UI
        draw_ui(img, h, w)
        
        cv2.imshow("Virtual Mouse", img)
        
        # Phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            Config.SMOOTHING = min(10, Config.SMOOTHING + 1)
            print(f"  Smoothing: {Config.SMOOTHING}")
        elif key == ord('-'):
            Config.SMOOTHING = max(1, Config.SMOOTHING - 1)
            print(f"  Smoothing: {Config.SMOOTHING}")
        elif key == ord('r'):
            State.prev_x, State.prev_y = SCREEN_W // 2, SCREEN_H // 2
            pyautogui.moveTo(SCREEN_W // 2, SCREEN_H // 2)
            print("  Reset vị trí chuột!")

except KeyboardInterrupt:
    pass

finally:
    if State.is_dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n  ✅ Đã thoát Virtual Mouse!\n")
