"""
漫画气泡检测 UI
使用 YOLO11n 模型检测漫画中的对话气泡
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading

try:
    from PIL import Image, ImageTk
except ImportError:
    print("请安装 Pillow: pip install Pillow")
    exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("请安装 ultralytics: pip install ultralytics>=8.0.0")
    exit(1)


class BubbleDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("漫画气泡检测器")
        self.root.geometry("1200x800")

        # 模型路径
        self.model_path = Path(__file__).parent / "best.pt"
        self.model = None
        self.current_image_path = None
        self.result_image = None

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 顶部按钮栏
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.select_btn = ttk.Button(btn_frame, text="选择图片", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.detect_btn = ttk.Button(btn_frame, text="检测气泡", command=self.detect_bubbles, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(btn_frame, text="保存结果", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 状态标签
        self.status_label = ttk.Label(btn_frame, text="正在加载模型...")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # 图片显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧：原图
        left_frame = ttk.LabelFrame(image_frame, text="原图", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.original_canvas = tk.Canvas(left_frame, bg="#f0f0f0")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # 右侧：检测结果
        right_frame = ttk.LabelFrame(image_frame, text="检测结果", padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.result_canvas = tk.Canvas(right_frame, bg="#f0f0f0")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        # 底部信息栏
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_label = ttk.Label(info_frame, text="")
        self.info_label.pack(side=tk.LEFT)

    def load_model(self):
        """后台加载模型"""
        def _load():
            try:
                if not self.model_path.exists():
                    self.root.after(0, lambda: messagebox.showerror("错误", f"模型文件不存在: {self.model_path}"))
                    self.root.after(0, lambda: self.status_label.config(text="模型加载失败"))
                    return

                self.model = YOLO(str(self.model_path))
                self.root.after(0, lambda: self.status_label.config(text="模型已加载"))
                self.root.after(0, lambda: self.select_btn.config(state=tk.NORMAL))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"加载模型失败: {e}"))
                self.root.after(0, lambda: self.status_label.config(text="模型加载失败"))

        threading.Thread(target=_load, daemon=True).start()

    def select_image(self):
        """选择图片文件"""
        filetypes = [
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("所有文件", "*.*")
        ]
        path = filedialog.askopenfilename(title="选择漫画图片", filetypes=filetypes)
        if path:
            self.current_image_path = path
            self.display_image(path, self.original_canvas)
            self.detect_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
            self.result_canvas.delete("all")
            self.info_label.config(text=f"已选择: {Path(path).name}")

    def display_image(self, image_path_or_pil, canvas):
        """在画布上显示图片"""
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil)
        else:
            img = image_path_or_pil

        # 获取画布尺寸
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 550
            canvas_height = 700

        # 缩放图片以适应画布
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)

        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_resized)

        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # 保持引用

    def detect_bubbles(self):
        """执行气泡检测"""
        if not self.current_image_path or not self.model:
            return

        self.detect_btn.config(state=tk.DISABLED)
        self.status_label.config(text="正在检测...")

        def _detect():
            try:
                results = self.model(self.current_image_path, imgsz=1600)[0]

                # 获取带标注的结果图
                result_img = results.plot()

                # BGR 转 RGB
                import cv2
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                self.result_image = Image.fromarray(result_rgb)

                # 统计检测结果
                num_bubbles = len(results.boxes) if results.boxes is not None else 0

                # 更新 UI
                self.root.after(0, lambda: self.display_image(self.result_image, self.result_canvas))
                self.root.after(0, lambda: self.status_label.config(text="检测完成"))
                self.root.after(0, lambda: self.info_label.config(text=f"检测到 {num_bubbles} 个气泡"))
                self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"检测失败: {e}"))
                self.root.after(0, lambda: self.status_label.config(text="检测失败"))
                self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))

        threading.Thread(target=_detect, daemon=True).start()

    def save_result(self):
        """保存检测结果"""
        if not self.result_image:
            return

        filetypes = [
            ("PNG 文件", "*.png"),
            ("JPEG 文件", "*.jpg"),
            ("所有文件", "*.*")
        ]
        path = filedialog.asksaveasfilename(
            title="保存结果",
            filetypes=filetypes,
            defaultextension=".png"
        )
        if path:
            self.result_image.save(path)
            messagebox.showinfo("成功", f"结果已保存到:\n{path}")


def main():
    root = tk.Tk()
    app = BubbleDetectorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
