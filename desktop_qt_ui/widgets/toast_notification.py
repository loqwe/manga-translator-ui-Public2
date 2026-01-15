"""
Toast通知组件
用于显示操作状态的非阻塞通知
"""
import os
import subprocess
import platform
from PyQt6.QtWidgets import QLabel, QGraphicsOpacityEffect
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont, QCursor


class ToastNotification(QLabel):
    """Toast通知组件"""
    
    clicked = pyqtSignal(str)  # 点击事件，传递附加数据（如文件路径）
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        # 不使用WA_TranslucentBackground，保持背景可见
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # 灰色样式
        self.setStyleSheet("""
            QLabel {
                background-color: #505050;
                color: #FFFFFF;
                border: 1px solid #787878;
                border-radius: 10px;
                padding: 8px 20px;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        self.setAutoFillBackground(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMaximumWidth(800)
        
        # 设置字体
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        
        # 透明度效果
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        # 动画
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # 自动关闭定时器
        self.auto_close_timer = QTimer()
        self.auto_close_timer.timeout.connect(self.fade_out)
        
        # 附加数据（如文件路径）
        self._extra_data = None
        self._clickable = False
    
    def show_toast(self, message, duration=3000, success=True, clickable_path=None):
        """
        显示Toast通知
        
        Args:
            message: 显示的消息
            duration: 显示持续时间（毫秒）
            success: 是否为成功消息（影响颜色）
            clickable_path: 可点击的路径（如果提供，Toast可点击打开文件夹）
        """
        self.setText(message)
        self.adjustSize()
        
        # 灰色样式，去掉min-width让它自适应文字
        if success:
            self.setStyleSheet("""
                QLabel {
                    background-color: #505050;
                    color: #FFFFFF;
                    border: 1px solid #787878;
                    border-radius: 10px;
                    padding: 8px 20px;
                    font-size: 13px;
                    line-height: 1.4;
                }
            """)
        else:
            # 错误时使用深红灰色
            self.setStyleSheet("""
                QLabel {
                    background-color: #644646;
                    color: #FFFFFF;
                    border: 1px solid #8B5A5A;
                    border-radius: 10px;
                    padding: 8px 20px;
                    font-size: 13px;
                    line-height: 1.4;
                }
            """)
        self.setAutoFillBackground(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMaximumWidth(800)
        
        # 设置可点击
        self._clickable = clickable_path is not None
        self._extra_data = clickable_path
        if self._clickable:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            # 可点击时添加完整提示
            self.setText(message + "\n(点击打开所在文件夹)")
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        
        # 调整大小后重新计算尺寸
        self.adjustSize()
        
        # 定位到窗口中心偏下（80%位置）
        if self.parent():
            # 使用parent widget的坐标系统
            parent_widget = self.parent()
            # 获取parent widget的矩形区域
            parent_rect = parent_widget.rect()
            # 计算parent的全局左上角位置
            parent_top_left = parent_widget.mapToGlobal(parent_rect.topLeft())
            # 计算Toast的全局位置（水平居中，垂直80%）
            toast_x = parent_top_left.x() + (parent_rect.width() - self.width()) // 2
            toast_y = parent_top_left.y() + int(parent_rect.height() * 0.8)
            self.move(toast_x, toast_y)
        else:
            # 如果没有parent，使用屏幕中心偏下
            from PyQt6.QtWidgets import QApplication
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = int(screen.height() * 0.8)
            self.move(x, y)
        
        # 淡入动画
        self.fade_in()
        
        # 设置自动关闭 - 确保在主线程
        if duration > 0:
            self.auto_close_timer.stop()  # 先停止旧的timer
            self.auto_close_timer.start(duration)
    
    def fade_in(self):
        """淡入动画"""
        self.opacity_effect.setOpacity(0)
        self.show()
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.start()
    
    def fade_out(self):
        """淡出动画"""
        self.auto_close_timer.stop()
        self.fade_animation.stop()  # 停止可能正在进行的动画
        self.fade_animation.setStartValue(self.opacity_effect.opacity())
        self.fade_animation.setEndValue(0)
        # 断开之前的连接，避免重复连接
        try:
            self.fade_animation.finished.disconnect()
        except:
            pass
        self.fade_animation.finished.connect(self.close)  # 用close替代hide，确保销毁
        self.fade_animation.start()
    
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if self._clickable and self._extra_data:
            self.open_file_location(self._extra_data)
            self.clicked.emit(self._extra_data)
        self.fade_out()
    
    @staticmethod
    def open_file_location(file_path):
        """打开文件所在文件夹并选中文件"""
        if not os.path.exists(file_path):
            return
        
        system = platform.system()
        
        try:
            if system == "Windows":
                # Windows: 使用 explorer /select,<路径>
                subprocess.run(['explorer', '/select,', os.path.normpath(file_path)])
            elif system == "Darwin":  # macOS
                # macOS: 使用 open -R <路径>
                subprocess.run(['open', '-R', file_path])
            else:  # Linux
                # Linux: 打开文件所在目录
                folder_path = os.path.dirname(file_path)
                subprocess.run(['xdg-open', folder_path])
        except Exception as e:
            print(f"无法打开文件位置: {e}")


class ToastManager:
    """Toast管理器，管理多个Toast的显示"""
    
    def __init__(self, parent):
        self.parent = parent
        self.active_toasts = []
    
    def show_toast(self, message, duration=3000, success=True, clickable_path=None):
        """显示一个Toast通知"""
        # 清理已关闭的Toast
        self.active_toasts = [toast for toast in self.active_toasts if toast.isVisible()]
        
        # 创建新的Toast
        toast = ToastNotification(self.parent)
        
        # 如果有多个Toast，堆叠显示
        if self.active_toasts:
            last_toast = self.active_toasts[-1]
            y_offset = last_toast.y() - toast.height() - 10
            parent_rect = self.parent.geometry()
            x = parent_rect.right() - toast.width() - 20
            toast.move(x, y_offset)
        
        toast.show_toast(message, duration, success, clickable_path)
        self.active_toasts.append(toast)
        
        return toast
    
    def show_success(self, message, duration=3000, clickable_path=None):
        """显示成功Toast"""
        return self.show_toast(message, duration, True, clickable_path)
    
    def show_error(self, message, duration=3000):
        """显示错误Toast"""
        return self.show_toast(message, duration, False, None)
    
    def show_info(self, message, duration=3000):
        """显示信息Toast"""
        return self.show_toast(message, duration, True, None)
    
    def close_all(self):
        """关闭所有活跃的Toast"""
        for toast in self.active_toasts:
            if toast.isVisible():
                toast.fade_out()
        self.active_toasts.clear()

