import Jetson.GPIO as GPIO
import time
from RPLCD.i2c import CharLCD
import tkinter as tk
from tkinter import Button, Label, Frame, Entry, Toplevel, messagebox, Scrollbar, LabelFrame
import cv2
from PIL import Image, ImageTk
from detect import trt_v5
import csv
import datetime
import os
import numpy as np

class FaceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        # self.window.geometry("1200x720") # Bỏ kích thước cố định để cửa sổ tự co giãn
        self.window.configure(bg="#f0f2f5") 

        # Folder setup
        self.images_dir = 'attendance_images'
        os.makedirs(self.images_dir, exist_ok=True)

        # CSV setup
        self.csv_file = 'attendance.csv'
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'time', 'image_path'])

        # Cấu hình tham số cho model (TensorRT)
        yolo_trt = './models/yolov5m-face_fp16.engine'
        resnet_trt = './models/backbone_fp16.engine'
        conf = 0.5
        iou_thres = 0.5
        img_size = (640, 640)
        classes_txt = './/yolov5-face//classes.txt'

        # Khởi tạo detector
        self.detector = trt_v5(
            vid_path=0, 
            yolo_trt=yolo_trt, 
            resnet_trt=resnet_trt, 
            img_size=img_size, 
            classes=classes_txt, 
            webcam=True,
            conf_thres=conf,
            iou_thres=iou_thres
        )

        # Mở camera
        self.vid = cv2.VideoCapture(0)
        
        if not self.vid.isOpened():
            print("Không thể mở camera")
            return

        # Lấy kích thước video
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # --- BỐ CỤC GIAO DIỆN ---
        
        # Container chính
        self.main_container = Frame(window, bg="#f0f2f5")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 1. Cột Trái: Video Feed + Nút Điểm danh
        self.left_column = Frame(self.main_container, bg="#f0f2f5")
        self.left_column.pack(side=tk.LEFT, anchor=tk.N) # anchor N để không bị giãn dọc

        # Khung chứa video (để tạo viền)
        self.video_frame = Frame(self.left_column, bg="black", bd=2, relief=tk.SUNKEN)
        self.video_frame.pack(side=tk.TOP)
        
        # Canvas hiển thị video
        self.canvas = tk.Canvas(self.video_frame, width=self.width, height=self.height, bg="black", highlightthickness=0)
        self.canvas.pack() 

        # Hiển thị Ngày & Giờ
        self.lbl_datetime = Label(self.left_column, text="", font=("Segoe UI", 14), bg="#f0f2f5", fg="#333")
        self.lbl_datetime.pack(side=tk.TOP, pady=(10, 0))

        # --- Lựa chọn chế độ ---
        self.mode_frame = Frame(self.left_column, bg="#f0f2f5")
        self.mode_frame.pack(side=tk.TOP, pady=(10, 5))
        
        self.mode_var = tk.StringVar(value="manual")
        
        tk.Radiobutton(self.mode_frame, text="Thủ công", variable=self.mode_var, value="manual", 
                       bg="#f0f2f5", font=("Segoe UI", 11), command=self.toggle_mode).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(self.mode_frame, text="Tự động", variable=self.mode_var, value="auto", 
                       bg="#f0f2f5", font=("Segoe UI", 11), command=self.toggle_mode).pack(side=tk.LEFT, padx=10)
        # -----------------------

        # Nút Điểm danh ngay (To hơn, nằm dưới camera)
        self.btn_attendance = Button(self.left_column, text="DIEM DANH NGAY", font=("Segoe UI", 16, "bold"), bg="#4CAF50", fg="white", activebackground="#388E3C", activeforeground="white", relief=tk.FLAT, cursor="hand2", command=self.take_attendance)
        self.btn_attendance.pack(side=tk.TOP, fill=tk.X, pady=(15, 0), ipady=10)

        # 2. Cột Phải: Sidebar điều khiển
        self.sidebar = Frame(self.main_container, bg="#ffffff", bd=1, relief=tk.RIDGE)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), anchor="n")
        
        # Tiêu đề Sidebar
        Label(self.sidebar, text="He Thong Diem Danh", font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#333").pack(pady=(15, 10), padx=10)

        # --- Nhóm 1: Đăng ký khuôn mặt ---
        self.group_register = LabelFrame(self.sidebar, text="Them khuon mat moi", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#555", bd=1, relief=tk.GROOVE)
        self.group_register.pack(fill=tk.X, padx=10, pady=5)

        self.entry_name = Entry(self.group_register, font=("Segoe UI", 11), bd=1, relief=tk.SOLID, width=30)
        self.entry_name.pack(pady=(10, 5), padx=10, fill=tk.X)
        
        self.btn_add_face = Button(self.group_register, text="Chup & Them", font=("Segoe UI", 10, "bold"), bg="#2196F3", fg="white", activebackground="#1976D2", activeforeground="white", relief=tk.FLAT, cursor="hand2", command=self.trigger_add_face)
        self.btn_add_face.pack(pady=(5, 5), padx=10, fill=tk.X)

        self.btn_manage = Button(self.group_register, text="Quan ly DS Khuon mat", font=("Segoe UI", 9), bg="#FF9800", fg="white", activebackground="#F57C00", activeforeground="white", relief=tk.FLAT, cursor="hand2", command=self.open_manager_window)
        self.btn_manage.pack(pady=(5, 10), padx=10, fill=tk.X)

        # --- Nhóm 2: Tác vụ khác ---
        self.group_attendance = LabelFrame(self.sidebar, text="Tac vu khac", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#555", bd=1, relief=tk.GROOVE)
        self.group_attendance.pack(fill=tk.X, padx=10, pady=10)

        self.btn_attendance_list = Button(self.group_attendance, text="Xem lich su diem danh", font=("Segoe UI", 9), bg="#009688", fg="white", activebackground="#00796B", activeforeground="white", relief=tk.FLAT, cursor="hand2", command=self.open_attendance_window)
        self.btn_attendance_list.pack(pady=(10, 10), padx=10, fill=tk.X)

        # --- Nhóm 3: Lịch sử gần đây ---
        Label(self.sidebar, text="Ghi nhan gan day", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#333").pack(pady=(10, 5), anchor="w", padx=10)
        
        self.history_container = Frame(self.sidebar, bg="#f9f9f9", bd=1, relief=tk.SUNKEN)
        self.history_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.recent_widgets = [] 
        self.check_attendance = False
        self.register_new_face = False
        self.register_name = ""
        
        # Biến lưu thời gian điểm danh gần nhất cho chế độ tự động {name: datetime}
        self.last_attendance_time = {}

        # Load history on startup
        self.load_recent_history()

        # Nút thoát
        self.btn_quit = Button(self.sidebar, text="Thoát chương trình", font=("Segoe UI", 10, "bold"), bg="#d32f2f", fg="white", activebackground="#b71c1c", activeforeground="white", relief=tk.FLAT, cursor="hand2", command=self.quit)
        self.btn_quit.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=15)

        # Bắt đầu vòng lặp update frame
        self.delay = 15 
        # --- Chèn vào cuối hàm __init__ ---
        try:
            self.lcd = CharLCD(
                i2c_expander="PCF8574",
                address=0x27,
                port=7,
                cols=16,
                rows=2,
                charmap="A00",
                auto_linebreaks=False,
            )
            self.lcd_last_update = time.time()
            self.lcd_is_off = False
            self.show_lcd("He Thong", "San sang!") # Hàm này sẽ viết ở Bước 3
        except Exception as e:
            print(f"Lỗi LCD: {e}")
            self.lcd = None
        # self.update()
        self.BUTTON_PIN = 7
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.BUTTON_PIN, GPIO.IN)
        self.last_button_state = GPIO.LOW
        # 2. Sau đó mới gọi hàm update
        self.update()
        self.window.mainloop()

    def take_attendance(self):
        self.check_attendance = True

    def toggle_mode(self):
        if self.mode_var.get() == "auto":
            self.btn_attendance.config(state="disabled", bg="#a5d6a7")
        else:
            self.btn_attendance.config(state="normal", bg="#4CAF50")
    def show_lcd(self, line1, line2):
        if self.lcd:
            try:
                self.lcd.backlight_enabled = True
                self.lcd.display_enabled = True
                self.lcd_is_off = False
                self.lcd_last_update = time.time() # Lưu mốc thời gian cập nhật

                self.lcd.cursor_pos = (0, 0)
                self.lcd.write_string(line1[:16].ljust(16))
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:16].ljust(16))
            except: pass

    def turn_off_lcd(self):
        if self.lcd and not self.lcd_is_off:
            try:
                self.lcd.backlight_enabled = False
                self.lcd.display_enabled = False
                self.lcd_is_off = True
            except: pass
    def record_attendance(self, name, face_img):
        if name == "unknown":
            return

        # Lấy thời gian hiện tại
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S %d/%m")
        
        # Save image
        filename = f"{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(self.images_dir, filename)
        cv2.imwrite(save_path, face_img)

        # Ghi vào CSV
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, time_str, save_path])
        self.show_lcd(name, time_str)


        # Cập nhật sidebar
        self.add_history_item(name, face_img, time_str)

    def open_attendance_window(self):
        att_win = Toplevel(self.window)
        att_win.title("Lịch sử điểm danh")
        att_win.geometry("600x600")

        # Scrollable setup
        canvas = tk.Canvas(att_win)
        scrollbar = Scrollbar(att_win, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Load CSV
        if not os.path.exists(self.csv_file):
            Label(scrollable_frame, text="Chưa có dữ liệu").pack()
            return

        with open(self.csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) <= 1:
             Label(scrollable_frame, text="Chưa có dữ liệu").pack()
             return

        # Display newest first
        for i, row in enumerate(reversed(rows[1:])):
            original_index = len(rows) - 1 - i 
            
            if len(row) < 3: continue
            name, time_str, img_path = row[0], row[1], row[2]

            row_frame = Frame(scrollable_frame, pady=5, padx=5, bd=1, relief=tk.RIDGE)
            row_frame.pack(fill=tk.X, padx=5, pady=2)

            # Image
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img = img.resize((50, 50))
                    img_tk = ImageTk.PhotoImage(img)
                    lbl_img = Label(row_frame, image=img_tk)
                    lbl_img.image = img_tk
                    lbl_img.pack(side=tk.LEFT, padx=5)
                else:
                    Label(row_frame, text="No Img").pack(side=tk.LEFT, padx=5)
            except:
                Label(row_frame, text="Error").pack(side=tk.LEFT, padx=5)

            # Text
            Label(row_frame, text=f"{name}\n{time_str}", font=("Arial", 10), width=30, anchor="w").pack(side=tk.LEFT, padx=5)

            # Delete Button
            Button(row_frame, text="Xóa", bg="#F44336", fg="white",
                   command=lambda idx=original_index: self.delete_attendance_record(idx, att_win)).pack(side=tk.RIGHT, padx=5)

    def delete_attendance_record(self, index, window):
        if not messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa bản ghi này?"):
            return

        try:
            with open(self.csv_file, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if 0 <= index < len(rows):
                row_to_del = rows[index]
                # Delete image file
                if len(row_to_del) >= 3:
                    img_path = row_to_del[2]
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except:
                            pass
                
                del rows[index]

                with open(self.csv_file, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                
                window.destroy()
                self.open_attendance_window()
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def open_manager_window(self):
        manager_win = Toplevel(self.window)
        manager_win.title("Quản lý danh sách khuôn mặt")
        manager_win.geometry("500x600")

        # Scrollable frame setup
        canvas = tk.Canvas(manager_win)
        scrollbar = Scrollbar(manager_win, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Load faces
        db_img_dir = 'database_image'
        if not os.path.exists(db_img_dir):
            os.makedirs(db_img_dir)

        files = [f for f in os.listdir(db_img_dir) if f.endswith('.jpg')]
        
        for f in files:
            name = os.path.splitext(f)[0]
            img_path = os.path.join(db_img_dir, f)
            
            row_frame = Frame(scrollable_frame, pady=5, padx=5, bd=1, relief=tk.RIDGE)
            row_frame.pack(fill=tk.X, padx=5, pady=2)

            # Image
            try:
                img = Image.open(img_path)
                img = img.resize((50, 50))
                img_tk = ImageTk.PhotoImage(img)
                lbl_img = Label(row_frame, image=img_tk)
                lbl_img.image = img_tk
                lbl_img.pack(side=tk.LEFT, padx=5)
            except:
                Label(row_frame, text="No Img").pack(side=tk.LEFT, padx=5)

            # Name Entry
            ent_name = Entry(row_frame, width=20)
            ent_name.insert(0, name)
            ent_name.pack(side=tk.LEFT, padx=5)

            # Buttons
            btn_save = Button(row_frame, text="Lưu", bg="#4CAF50", fg="white",
                              command=lambda n=name, e=ent_name: self.rename_face(n, e.get(), manager_win))
            btn_save.pack(side=tk.LEFT, padx=2)

            btn_del = Button(row_frame, text="Xóa", bg="#F44336", fg="white",
                             command=lambda n=name: self.delete_face(n, manager_win))
            btn_del.pack(side=tk.LEFT, padx=2)

    def rename_face(self, old_name, new_name, window):
        if not new_name or new_name == old_name:
            return

        try:
            # Rename image
            old_img = os.path.join('database_image', f"{old_name}.jpg")
            new_img = os.path.join('database_image', f"{new_name}.jpg")
            if os.path.exists(old_img):
                os.rename(old_img, new_img)

            # Rename tensor
            old_npy = os.path.join('database_tensor', f"{old_name}.npy")
            new_npy = os.path.join('database_tensor', f"{new_name}.npy")
            if os.path.exists(old_npy):
                os.rename(old_npy, new_npy)
            
            messagebox.showinfo("Thành công", f"Đã đổi tên {old_name} thành {new_name}")
            self.detector.reload_database()
            window.destroy()
            self.open_manager_window() # Refresh
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def delete_face(self, name, window):
        if not messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa {name}?"):
            return

        try:
            # Delete image
            img_path = os.path.join('database_image', f"{name}.jpg")
            if os.path.exists(img_path):
                os.remove(img_path)

            # Delete tensor
            npy_path = os.path.join('database_tensor', f"{name}.npy")
            if os.path.exists(npy_path):
                os.remove(npy_path)

            self.detector.reload_database()
            window.destroy()
            self.open_manager_window() # Refresh
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def trigger_add_face(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Thông báo", "Vui lòng nhập tên!")
            return
        self.register_name = name
        self.register_new_face = True

    def load_recent_history(self):
        if not os.path.exists(self.csv_file):
            return
            
        try:
            with open(self.csv_file, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            if len(rows) <= 1:
                return
                
            # Skip header, take last 5
            data_rows = rows[1:]
            recent_rows = data_rows[-5:]
            
            for row in recent_rows:
                # Ensure row has enough columns (name, time, image_path)
                if len(row) >= 3:
                    name, time_str, img_path = row[0], row[1], row[2]
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.add_history_item(name, img, time_str)
        except Exception as e:
            print(f"Error loading history: {e}")

    def update(self):
        # Kiểm tra nút bấm vật lý (GPIO)
        current_state = GPIO.input(self.BUTTON_PIN)
        if current_state == GPIO.HIGH and self.last_button_state == GPIO.LOW:
            print("Nút vật lý đã bấm!")
            self.take_attendance() # Gọi hàm điểm danh tương tự như bấm nút trên màn hình
            # Có thể thêm time.sleep(0.2) ở đây để chống dội (debounce) nếu cần
        self.last_button_state = current_state
        # Cập nhật ngày giờ
        now = datetime.datetime.now()
        if self.lcd and not self.lcd_is_off:
            if (time.time() - self.lcd_last_update) > 10.0:
                self.turn_off_lcd()
        self.lbl_datetime.config(text=now.strftime("%H:%M:%S  -  %d/%m/%Y"))

        ret, frame = self.vid.read()
        if ret:
            # Nhận diện khuôn mặt
            # detect_face trả về danh sách kết quả: [(x1, y1, x2, y2, name, conf), ...]
            results = self.detector.detect_face(frame)
            
            recognized_list = []
            
            # Vẽ khung và tạo danh sách recognized_list
            for x1, y1, x2, y2, name, conf in results:
                # Vẽ bounding box (Xanh lá nếu đã biết, Đỏ nếu unknown)
                color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Cắt ảnh khuôn mặt để dùng cho điểm danh
                h_img, w_img = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                
                face_img = frame[y1:y2, x1:x2].copy()
                if face_img.size > 0:
                    recognized_list.append((name, face_img))
            
            # Xử lý thêm khuôn mặt mới
            if self.register_new_face:
                self.register_new_face = False
                if recognized_list:
                    # Lấy khuôn mặt (đã được lọc là to nhất ở detect.py)
                    _, face_img = recognized_list[0]
                    name = self.register_name
                    
                    # 1. Save Image
                    os.makedirs('database_image', exist_ok=True)
                    img_path = os.path.join('database_image', f"{name}.jpg")
                    cv2.imwrite(img_path, face_img)
                    
                    # 2. Extract Feature
                    feat = self.detector.extract_feat(face_img)
                    
                    # 3. Save Feature
                    os.makedirs('database_tensor', exist_ok=True)
                    np.save(os.path.join('database_tensor', f"{name}.npy"), feat)
                    
                    # 4. Update Runtime Database
                    self.detector.db_names.append(name)
                    if len(self.detector.db_feats) == 0:
                         self.detector.db_feats = feat[np.newaxis, :]
                    else:
                         self.detector.db_feats = np.vstack((self.detector.db_feats, feat))
                    
                    print(f"Đã thêm khuôn mặt: {name}")
                    self.entry_name.delete(0, tk.END)
                    messagebox.showinfo("Thành công", f"Đã thêm khuôn mặt: {name}")
                else:
                    print("Không tìm thấy khuôn mặt!")
                    messagebox.showwarning("Thất bại", "Không tìm thấy khuôn mặt nào!")

            # Xử lý ghi nhận và hiển thị sidebar
            
            # 1. Chế độ Tự động
            if self.mode_var.get() == "auto":
                for name, face_img in recognized_list:
                    if name == "unknown":
                        continue

                    now = datetime.datetime.now()
                    # Kiểm tra cooldown (ví dụ: 30 giây mới được điểm danh lại 1 lần)
                    if name not in self.last_attendance_time or (now - self.last_attendance_time[name]).total_seconds() > 30:
                        self.record_attendance(name, face_img)
                        self.last_attendance_time[name] = now

            # 2. Chế độ Thủ công
            if self.check_attendance:
                self.check_attendance = False # Reset flag
                if recognized_list:
                    # Lấy người cuối cùng trong danh sách (hoặc xử lý tất cả nếu cần)
                    name, face_img = recognized_list[-1]
                    if name != "unknown":
                        self.record_attendance(name, face_img)
                    else:
                        messagebox.showwarning("Cảnh báo", "Không thể điểm danh: Khuôn mặt chưa được đăng ký (unknown)")

            if frame is not None:
                # Chuyển đổi màu từ BGR (OpenCV) sang RGB (Tkinter/PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Chuyển thành ảnh PIL và sau đó là ImageTk để hiển thị trên Tkinter
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def add_history_item(self, name, face_img, time_str):
        # Resize ảnh mặt nhỏ lại cho danh sách
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (50, 50))
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(face_img))
        
        # Create item frame (container)
        item_frame = Frame(self.history_container, bg="white", bd=0, pady=2)
        
        # Inner frame for styling (border, padding)
        inner = Frame(item_frame, bg="white", bd=1, relief=tk.SOLID)
        inner.pack(fill=tk.X, padx=2, pady=2)

        # Image label
        lbl_img = Label(inner, image=img_tk, bg="white")
        lbl_img.image = img_tk # Keep reference
        lbl_img.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Info label
        lbl_text = Label(inner, text=f"{name}\n{time_str}", font=("Segoe UI", 9), bg="white", justify=tk.LEFT, fg="#333")
        lbl_text.pack(side=tk.LEFT, padx=5)
        
        # Insert at top
        if self.recent_widgets:
            item_frame.pack(side=tk.TOP, fill=tk.X, pady=2, before=self.recent_widgets[0])
        else:
            item_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            
        self.recent_widgets.insert(0, item_frame)
        
        # Limit to 5 items
        if len(self.recent_widgets) > 5:
            oldest = self.recent_widgets.pop()
            oldest.destroy()

    def quit(self):
        if self.vid.isOpened():
            self.vid.release()
        if self.lcd:
            self.lcd.clear()
            self.lcd.backlight_enabled = False
            self.lcd.close()
        GPIO.cleanup() # Thêm dòng này
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root, "Face Recognition App")