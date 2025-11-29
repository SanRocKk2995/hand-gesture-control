"""
Module Command Mapper - Ánh xạ cử chỉ tay sang lệnh điều khiển máy tính
Sử dụng PyAutoGUI để thực hiện các hành động
"""

import pyautogui
import time
import numpy as np
from typing import Tuple, Optional
from collections import deque


class CommandMapper:
    """
    Lớp ánh xạ và thực thi các lệnh điều khiển từ cử chỉ tay
    """
    
    def __init__(self, 
                 gesture_config: dict = None,
                 screen_width: Optional[int] = None,
                 screen_height: Optional[int] = None,
                 smoothing: int = 5):
        """
        Khởi tạo Command Mapper
        
        Args:
            gesture_config: Dict mapping gesture -> action từ config file
            screen_width: Chiều rộng màn hình (None = auto detect)
            screen_height: Chiều cao màn hình (None = auto detect)
            smoothing: Số frame để làm mượt chuyển động chuột
        """
        # Lưu gesture config
        self.gesture_config = gesture_config or {}
        
        # Lấy kích thước màn hình
        if screen_width is None or screen_height is None:
            self.screen_width, self.screen_height = pyautogui.size()
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height
        
        # Cài đặt PyAutoGUI
        pyautogui.FAILSAFE = True  # Di chuột đến góc trên trái để dừng
        pyautogui.PAUSE = 0.01     # Delay giữa các lệnh
        
        # Trạng thái
        self.last_gesture = None
        self.last_action_time = 0
        self.action_cooldown = 0.5  # Giây giữa các action
        
        # Smoothing cho di chuyển chuột
        self.smoothing = smoothing
        self.prev_positions = deque(maxlen=smoothing)
        
        # Vùng deadzone (tránh rung)
        self.deadzone = 10  # pixels
        self.last_mouse_pos = None
        
        # Mapping action name sang method
        self.action_methods = {
            # Chuột
            "left_click": self.left_click,
            "right_click": self.right_click,
            "double_click": self.double_click,
            "middle_click": self.middle_click,
            "move_mouse": self.move_mouse,
            
            # Cuộn
            "scroll_up": self.scroll_up,
            "scroll_down": self.scroll_down,
            "scroll_left": self.scroll_left,
            "scroll_right": self.scroll_right,
            
            # Phím đặc biệt
            "press_space": self.press_space,
            "press_enter": self.press_enter,
            "press_esc": self.press_esc,
            "press_tab": self.press_tab,
            "press_backspace": self.press_backspace,
            
            # Phím mũi tên
            "press_up": self.press_up,
            "press_down": self.press_down,
            "press_left": self.press_left,
            "press_right": self.press_right,
            
            # Tổ hợp phím Ctrl
            "ctrl_c": self.ctrl_c,
            "ctrl_v": self.ctrl_v,
            "ctrl_x": self.ctrl_x,
            "ctrl_z": self.ctrl_z,
            "ctrl_y": self.ctrl_y,
            "ctrl_a": self.ctrl_a,
            "ctrl_s": self.ctrl_s,
            "ctrl_f": self.ctrl_f,
            "ctrl_w": self.ctrl_w,
            "ctrl_t": self.ctrl_t,
            
            # Tổ hợp phím Alt
            "alt_tab": self.alt_tab,
            "alt_f4": self.alt_f4,
            
            # Media
            "volume_up": self.volume_up,
            "volume_down": self.volume_down,
            "volume_mute": self.volume_mute,
            "play_pause": self.play_pause,
            "next_track": self.next_track,
            "prev_track": self.prev_track,
            
            # Tiện ích
            "take_screenshot": self.take_screenshot,
            "open_browser": self.open_browser,
            "minimize_window": self.minimize_window,
            "maximize_window": self.maximize_window,
            "do_nothing": self.do_nothing
        }
        
        # Mapping cử chỉ cũ (để tương thích ngược)
        self.gesture_actions = {
            "Fist": self.left_click,
            "Open": self.move_mouse,
            "Point": self.move_mouse,
            "Peace": self.right_click,
            "ThumbsUp": self.scroll_up,
            "ThumbsDown": self.scroll_down,
            "OK": self.double_click,
            "Three": self.take_screenshot
        }
        
        print(f"Command Mapper initialized")
        print(f"Screen size: {self.screen_width} x {self.screen_height}")
    
    def convert_to_screen_coords(self, 
                                 hand_x: float, 
                                 hand_y: float,
                                 frame_width: int,
                                 frame_height: int) -> Tuple[int, int]:
        """
        Chuyển đổi tọa độ bàn tay sang tọa độ màn hình
        
        Args:
            hand_x, hand_y: Tọa độ bàn tay trong frame
            frame_width, frame_height: Kích thước frame camera
            
        Returns:
            Tuple (screen_x, screen_y)
        """
        # Mapping với lề để tránh vùng biên
        margin_x = int(frame_width * 0.1)
        margin_y = int(frame_height * 0.1)
        
        # Normalize về [0, 1]
        norm_x = (hand_x - margin_x) / (frame_width - 2 * margin_x)
        norm_y = (hand_y - margin_y) / (frame_height - 2 * margin_y)
        
        # Clamp về [0, 1]
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Scale lên kích thước màn hình
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        
        return screen_x, screen_y
    
    def smooth_position(self, x: int, y: int) -> Tuple[int, int]:
        """
        Làm mượt vị trí chuột bằng moving average
        
        Args:
            x, y: Tọa độ hiện tại
            
        Returns:
            Tuple (smoothed_x, smoothed_y)
        """
        self.prev_positions.append((x, y))
        
        if len(self.prev_positions) < self.smoothing:
            return x, y
        
        # Tính trung bình
        avg_x = int(np.mean([pos[0] for pos in self.prev_positions]))
        avg_y = int(np.mean([pos[1] for pos in self.prev_positions]))
        
        return avg_x, avg_y
    
    def move_mouse(self, 
                   hand_x: int, 
                   hand_y: int,
                   frame_width: int,
                   frame_height: int,
                   **kwargs):
        """
        Di chuyển chuột dựa trên vị trí bàn tay
        
        Args:
            hand_x, hand_y: Tọa độ bàn tay trong frame
            frame_width, frame_height: Kích thước frame
        """
        # Chuyển đổi sang tọa độ màn hình
        screen_x, screen_y = self.convert_to_screen_coords(
            hand_x, hand_y, frame_width, frame_height
        )
        
        # Làm mượt
        smooth_x, smooth_y = self.smooth_position(screen_x, screen_y)
        
        # Kiểm tra deadzone
        if self.last_mouse_pos is not None:
            dx = abs(smooth_x - self.last_mouse_pos[0])
            dy = abs(smooth_y - self.last_mouse_pos[1])
            
            if dx < self.deadzone and dy < self.deadzone:
                return  # Không di chuyển nếu trong vùng deadzone
        
        # Di chuyển chuột
        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
        self.last_mouse_pos = (smooth_x, smooth_y)
    
    def left_click(self, **kwargs):
        """
        Click chuột trái
        """
        if self._can_perform_action():
            pyautogui.click()
            print("Action: Left Click")
            self._update_action_time()
    
    def right_click(self, **kwargs):
        """
        Click chuột phải
        """
        if self._can_perform_action():
            pyautogui.rightClick()
            print("Action: Right Click")
            self._update_action_time()
    
    def double_click(self, **kwargs):
        """
        Double click chuột trái
        """
        if self._can_perform_action():
            pyautogui.doubleClick()
            print("Action: Double Click")
            self._update_action_time()
    
    def scroll_up(self, **kwargs):
        """
        Cuộn lên
        """
        if self._can_perform_action(cooldown=1.0):
            pyautogui.scroll(80)
            print("Action: Scroll Up")
            self._update_action_time()
    
    def scroll_down(self, **kwargs):
        """
        Cuộn xuống
        """
        if self._can_perform_action(cooldown=1.0):
            pyautogui.scroll(-80)
            print("Action: Scroll Down")
            self._update_action_time()
    
    def take_screenshot(self, **kwargs):
        """Chụp màn hình"""
        if self._can_perform_action():
            screenshot = pyautogui.screenshot()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            print(f"Screenshot saved: {filename}")
            self._update_action_time()
    
    # ====== THÊM CÁC ACTION MỚI ======
    
    def middle_click(self, **kwargs):
        """Nhấp chuột giữa"""
        if self._can_perform_action():
            pyautogui.middleClick()
            print("Action: Middle Click")
            self._update_action_time()
    
    def scroll_left(self, **kwargs):
        """Cuộn trái"""
        if self._can_perform_action(cooldown=1.0):
            pyautogui.hscroll(-3)
            print("Action: Scroll Left")
            self._update_action_time()
    
    def scroll_right(self, **kwargs):
        """Cuộn phải"""
        if self._can_perform_action(cooldown=1.0):
            pyautogui.hscroll(3)
            print("Action: Scroll Right")
            self._update_action_time()
    
    # Phím đặc biệt
    def press_space(self, **kwargs):
        """Nhấn Space"""
        if self._can_perform_action():
            pyautogui.press('space')
            print("Action: Press Space")
            self._update_action_time()
    
    def press_enter(self, **kwargs):
        """Nhấn Enter"""
        if self._can_perform_action():
            pyautogui.press('enter')
            print("Action: Press Enter")
            self._update_action_time()
    
    def press_esc(self, **kwargs):
        """Nhấn ESC"""
        if self._can_perform_action():
            pyautogui.press('esc')
            print("Action: Press ESC")
            self._update_action_time()
    
    def press_tab(self, **kwargs):
        """Nhấn Tab"""
        if self._can_perform_action():
            pyautogui.press('tab')
            print("Action: Press Tab")
            self._update_action_time()
    
    def press_backspace(self, **kwargs):
        """Nhấn Backspace"""
        if self._can_perform_action():
            pyautogui.press('backspace')
            print("Action: Press Backspace")
            self._update_action_time()
    
    # Phím mũi tên
    def press_up(self, **kwargs):
        """Nhấn mũi tên Lên"""
        if self._can_perform_action(cooldown=0.3):
            pyautogui.press('up')
            print("Action: Press Up Arrow")
            self._update_action_time()
    
    def press_down(self, **kwargs):
        """Nhấn mũi tên Xuống"""
        if self._can_perform_action(cooldown=0.3):
            pyautogui.press('down')
            print("Action: Press Down Arrow")
            self._update_action_time()
    
    def press_left(self, **kwargs):
        """Nhấn mũi tên Trái"""
        if self._can_perform_action(cooldown=0.3):
            pyautogui.press('left')
            print("Action: Press Left Arrow")
            self._update_action_time()
    
    def press_right(self, **kwargs):
        """Nhấn mũi tên Phải"""
        if self._can_perform_action(cooldown=0.3):
            pyautogui.press('right')
            print("Action: Press Right Arrow")
            self._update_action_time()
    
    # Tổ hợp phím Ctrl
    def ctrl_c(self, **kwargs):
        """Ctrl+C (Sao chép)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'c')
            print("Action: Ctrl+C (Copy)")
            self._update_action_time()
    
    def ctrl_v(self, **kwargs):
        """Ctrl+V (Dán)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'v')
            print("Action: Ctrl+V (Paste)")
            self._update_action_time()
    
    def ctrl_x(self, **kwargs):
        """Ctrl+X (Cắt)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'x')
            print("Action: Ctrl+X (Cut)")
            self._update_action_time()
    
    def ctrl_z(self, **kwargs):
        """Ctrl+Z (Hoàn tác)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'z')
            print("Action: Ctrl+Z (Undo)")
            self._update_action_time()
    
    def ctrl_y(self, **kwargs):
        """Ctrl+Y (Làm lại)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'y')
            print("Action: Ctrl+Y (Redo)")
            self._update_action_time()
    
    def ctrl_a(self, **kwargs):
        """Ctrl+A (Chọn tất cả)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'a')
            print("Action: Ctrl+A (Select All)")
            self._update_action_time()
    
    def ctrl_s(self, **kwargs):
        """Ctrl+S (Lưu)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 's')
            print("Action: Ctrl+S (Save)")
            self._update_action_time()
    
    def ctrl_f(self, **kwargs):
        """Ctrl+F (Tìm kiếm)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'f')
            print("Action: Ctrl+F (Find)")
            self._update_action_time()
    
    def ctrl_w(self, **kwargs):
        """Ctrl+W (Đóng tab)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 'w')
            print("Action: Ctrl+W (Close Tab)")
            self._update_action_time()
    
    def ctrl_t(self, **kwargs):
        """Ctrl+T (Tab mới)"""
        if self._can_perform_action():
            pyautogui.hotkey('ctrl', 't')
            print("Action: Ctrl+T (New Tab)")
            self._update_action_time()
    
    # Tổ hợp phím Alt
    def alt_tab(self, **kwargs):
        """Alt+Tab (Chuyển cửa sổ)"""
        if self._can_perform_action():
            pyautogui.hotkey('alt', 'tab')
            print("Action: Alt+Tab (Switch Window)")
            self._update_action_time()
    
    def alt_f4(self, **kwargs):
        """Alt+F4 (Đóng ứng dụng)"""
        if self._can_perform_action():
            pyautogui.hotkey('alt', 'F4')
            print("Action: Alt+F4 (Close App)")
            self._update_action_time()
    
    # Media controls
    def volume_up(self, **kwargs):
        """Tăng âm lượng"""
        if self._can_perform_action(cooldown=0.2):
            pyautogui.press('volumeup')
            print("Action: Volume Up")
            self._update_action_time()
    
    def volume_down(self, **kwargs):
        """Giảm âm lượng"""
        if self._can_perform_action(cooldown=0.2):
            pyautogui.press('volumedown')
            print("Action: Volume Down")
            self._update_action_time()
    
    def volume_mute(self, **kwargs):
        """Tắt/Bật tiếng"""
        if self._can_perform_action():
            pyautogui.press('volumemute')
            print("Action: Volume Mute/Unmute")
            self._update_action_time()
    
    def play_pause(self, **kwargs):
        """Phát/Dừng media"""
        if self._can_perform_action():
            pyautogui.press('playpause')
            print("Action: Play/Pause")
            self._update_action_time()
    
    def next_track(self, **kwargs):
        """Bài tiếp theo"""
        if self._can_perform_action():
            pyautogui.press('nexttrack')
            print("Action: Next Track")
            self._update_action_time()
    
    def prev_track(self, **kwargs):
        """Bài trước"""
        if self._can_perform_action():
            pyautogui.press('prevtrack')
            print("Action: Previous Track")
            self._update_action_time()
    
    # Tiện ích
    def open_browser(self, **kwargs):
        """Mở trình duyệt"""
        if self._can_perform_action():
            pyautogui.hotkey('win', 'r')
            time.sleep(0.3)
            pyautogui.write('firefox')
            pyautogui.press('enter')
            print("Action: Open Browser")
            self._update_action_time()
    
    def minimize_window(self, **kwargs):
        """Thu nhỏ cửa sổ"""
        if self._can_perform_action():
            pyautogui.hotkey('win', 'down')
            print("Action: Minimize Window")
            self._update_action_time()
    
    def maximize_window(self, **kwargs):
        """Phóng to cửa sổ"""
        if self._can_perform_action():
            pyautogui.hotkey('win', 'up')
            print("Action: Maximize Window")
            self._update_action_time()
    
    def custom_key(self, key_name: str, **kwargs):
        """Nhấn phím tùy chỉnh"""
        if self._can_perform_action():
            pyautogui.press(key_name)
            print(f"Action: Press {key_name}")
            self._update_action_time()
    
    def do_nothing(self, **kwargs):
        """Không làm gì"""
        pass
    
    def execute_action(self,
                      action_name: str,
                      hand_x: Optional[int] = None,
                      hand_y: Optional[int] = None,
                      frame_width: Optional[int] = None,
                      frame_height: Optional[int] = None):
        """
        Thực thi action theo tên
        
        Args:
            action_name: Tên action (vd: "left_click", "ctrl_c")
            hand_x, hand_y: Tọa độ bàn tay (cho move_mouse)
            frame_width, frame_height: Kích thước frame
        """
        if action_name not in self.action_methods:
            print(f"Unknown action: {action_name}")
            return
        
        action = self.action_methods[action_name]
        
        # Chuẩn bị kwargs
        kwargs = {}
        if hand_x is not None:
            kwargs['hand_x'] = hand_x
            kwargs['hand_y'] = hand_y
            kwargs['frame_width'] = frame_width
            kwargs['frame_height'] = frame_height
        
        # Thực thi action
        try:
            action(**kwargs)
        except Exception as e:
            print(f"Error executing {action_name}: {e}")
    
    def execute_gesture(self, 
                       gesture_name: str,
                       hand_x: Optional[int] = None,
                       hand_y: Optional[int] = None,
                       frame_width: Optional[int] = None,
                       frame_height: Optional[int] = None):
        """
        Thực thi action tương ứng với cử chỉ từ config
        
        Args:
            gesture_name: Tên cử chỉ (vd: "fist", "open_palm")
            hand_x, hand_y: Tọa độ bàn tay (cho move_mouse)
            frame_width, frame_height: Kích thước frame
            
        Returns:
            Action đã thực thi hoặc None
        """
        # Lấy action từ gesture_config
        gesture_data = self.gesture_config.get(gesture_name, {})
        
        if isinstance(gesture_data, dict):
            action_str = gesture_data.get("action", "")
            enabled = gesture_data.get("enabled", True)
            if not enabled or not action_str:
                return None
        elif isinstance(gesture_data, str):
            action_str = gesture_data
        else:
            return None
        
        if not action_str:
            return None
        
        # Thực thi action
        success = self.execute_key_action(action_str)
        
        if success:
            return action_str
        return None
    
    def execute_key_action(self, action_str: str) -> bool:
        """
        Thực thi phím/tổ hợp phím tùy ý
        
        Args:
            action_str: Chuỗi phím (vd: "space", "ctrl+c", "alt+tab")
            
        Returns:
            True nếu thực thi thành công
        """
        if not action_str or not self._can_perform_action():
            return False
        
        try:
            # Xử lý các action đặc biệt (chuột, media...)
            special_actions = {
                "click": lambda: pyautogui.click(),
                "left_click": lambda: pyautogui.click(),
                "right_click": lambda: pyautogui.rightClick(),
                "double_click": lambda: pyautogui.doubleClick(),
                "middle_click": lambda: pyautogui.middleClick(),
                "scroll_up": lambda: pyautogui.scroll(80),
                "scroll_down": lambda: pyautogui.scroll(-80),
                "scroll_left": lambda: pyautogui.hscroll(-3),
                "scroll_right": lambda: pyautogui.hscroll(3),
                # Di chuyển chuột
                "mouse_up": lambda: pyautogui.move(0, -50),
                "mouse_down": lambda: pyautogui.move(0, 50),
                "mouse_left": lambda: pyautogui.move(-50, 0),
                "mouse_right": lambda: pyautogui.move(50, 0),
                "mouse_center": lambda: pyautogui.moveTo(self.screen_width // 2, self.screen_height // 2),
                "mouse_drag": lambda: pyautogui.mouseDown() if not pyautogui.mouseDown else pyautogui.mouseUp(),
                # Media
                "volume_up": lambda: pyautogui.press('volumeup'),
                "volume_down": lambda: pyautogui.press('volumedown'),
                "volume_mute": lambda: pyautogui.press('volumemute'),
                "play_pause": lambda: pyautogui.press('playpause'),
                "next_track": lambda: pyautogui.press('nexttrack'),
                "prev_track": lambda: pyautogui.press('prevtrack'),
                "stop": lambda: pyautogui.press('stop'),
                "brightness_up": lambda: pyautogui.press('brightnessup'),
                "brightness_down": lambda: pyautogui.press('brightnessdown'),
                "print_screen": lambda: pyautogui.press('printscreen'),
            }
            
            if action_str in special_actions:
                special_actions[action_str]()
                print(f"Action: {action_str}")
                self._update_action_time()
                return True
            
            # Xử lý tổ hợp phím (ctrl+c, alt+tab, win+d...)
            if '+' in action_str:
                keys = action_str.lower().split('+')
                # Chuyển đổi tên phím
                key_mapping = {
                    'ctrl': 'ctrl',
                    'alt': 'alt', 
                    'shift': 'shift',
                    'win': 'win',
                    'meta': 'win',
                    'enter': 'enter',
                    'return': 'enter',
                    'esc': 'escape',
                    'escape': 'escape',
                    'del': 'delete',
                    'ins': 'insert',
                }
                mapped_keys = [key_mapping.get(k, k) for k in keys]
                pyautogui.hotkey(*mapped_keys)
                print(f"Action: {action_str}")
                self._update_action_time()
                return True
            
            # Xử lý phím đơn
            key_mapping = {
                'space': 'space',
                'enter': 'enter',
                'return': 'enter',
                'esc': 'escape',
                'escape': 'escape',
                'tab': 'tab',
                'backspace': 'backspace',
                'delete': 'delete',
                'del': 'delete',
                'insert': 'insert',
                'ins': 'insert',
                'home': 'home',
                'end': 'end',
                'pageup': 'pageup',
                'pagedown': 'pagedown',
                'up': 'up',
                'down': 'down',
                'left': 'left',
                'right': 'right',
                'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
                'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
                'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',
                'capslock': 'capslock',
                'numlock': 'numlock',
                'scrolllock': 'scrolllock',
                'pause': 'pause',
            }
            
            key = key_mapping.get(action_str.lower(), action_str.lower())
            pyautogui.press(key)
            print(f"Action: Press {key}")
            self._update_action_time()
            return True
            
        except Exception as e:
            print(f"Error executing action '{action_str}': {e}")
            return False
    
    def _can_perform_action(self, cooldown: Optional[float] = None) -> bool:
        """
        Kiểm tra xem có thể thực hiện action không (dựa trên cooldown)
        
        Args:
            cooldown: Thời gian cooldown (None = dùng default)
            
        Returns:
            True nếu có thể thực hiện
        """
        if cooldown is None:
            cooldown = self.action_cooldown
        
        current_time = time.time()
        if current_time - self.last_action_time >= cooldown:
            return True
        return False
    
    def _update_action_time(self):
        """
        Cập nhật thời gian thực hiện action cuối cùng
        """
        self.last_action_time = time.time()
    
    def reset(self):
        """
        Reset trạng thái
        """
        self.last_gesture = None
        self.last_action_time = 0
        self.prev_positions.clear()
        self.last_mouse_pos = None
        print("Command Mapper reset")


if __name__ == "__main__":
    # Test module
    print("Testing Command Mapper...")
    
    mapper = CommandMapper()
    
    # Test các action
    print("\nTesting actions (sẽ thực thi sau 3 giây)...")
    time.sleep(3)
    
    print("Testing mouse move...")
    mapper.move_mouse(320, 240, 640, 480)
    
    time.sleep(1)
    print("Testing left click...")
    mapper.left_click()
    
    time.sleep(1)
    print("Testing scroll...")
    mapper.scroll_up()
    
    print("\nTest completed!")
