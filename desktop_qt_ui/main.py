import sys
import os
import logging
import warnings

# 抑制第三方库的警告（必须在导入其他库之前设置）
warnings.filterwarnings('ignore', message='.*Triton.*')
warnings.filterwarnings('ignore', message='.*triton.*')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ctranslate2')
warnings.filterwarnings('ignore', module='xformers')

# 在 PyTorch 初始化前设置显存优化，允许使用共享显存
# expandable_segments 可以减少显存碎片，避免 OOM 错误
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# 设置 Hugging Face 镜像站（国内用户加速下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'

# 禁用 SSL 验证（解决 hf-mirror.com 证书问题）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

# 修复便携版Python的路径问题：将脚本所在目录添加到sys.path开头
# 便携版Python使用._pth文件会禁用自动添加脚本目录的默认行为
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 修复PyInstaller打包后onnxruntime的DLL加载问题
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # 运行在PyInstaller打包环境中
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        # 只设置DLL搜索路径，不预加载
        # 让Python的导入机制自然处理DLL加载
        os.add_dll_directory(sys._MEIPASS)
        onnx_capi_dir = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
        if os.path.exists(onnx_capi_dir):
            os.add_dll_directory(onnx_capi_dir)

from PyQt6.QtWidgets import QApplication
from main_window import MainWindow
from services import init_services

# 全局异常处理器，捕获未处理的异常并记录到日志
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """全局异常处理器，防止程序静默崩溃"""
    import traceback
    
    # 忽略 KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # 格式化异常信息
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    # 记录到日志（会写入 logs/app.log 和 logs/error.log）
    logging.critical(f"未捕获的异常导致程序崩溃:\n{error_msg}")
    
    # 同时输出到控制台（确保能看到）
    print(f"\n{'='*60}", file=sys.stderr)
    print("❌ 程序发生未捕获的异常:", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(error_msg, file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

# 设置全局异常处理器
sys.excepthook = global_exception_handler

def main():
    """
    应用主入口
    """
    # --- 日志配置（异步优化）---
    import queue
    import threading
    import atexit
    
    # 创建异步日志处理器
    class AsyncStreamHandler(logging.Handler):
        """异步日志处理器，避免阻塞主线程"""
        def __init__(self, stream=sys.stdout):
            super().__init__()
            self.stream = stream
            # 限制队列大小为1000，避免日志过多导致内存占用
            self.log_queue = queue.Queue(maxsize=1000)
            self.running = True
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        
        def _worker(self):
            while self.running:
                try:
                    # ✅ 减少超时时间，更快处理日志
                    record = self.log_queue.get(timeout=0.01)
                    if record is None:
                        break
                    msg = self.format(record)
                    self.stream.write(msg + '\n')
                    # ✅ 每条日志立即刷新
                    self.stream.flush()
                except queue.Empty:
                    # ✅ 即使队列为空也刷新一次，确保之前的输出显示
                    try:
                        self.stream.flush()
                    except:
                        pass
                    continue
                except Exception:
                    pass
        
        def emit(self, record):
            try:
                self.log_queue.put_nowait(record)
            except queue.Full:
                pass  # 队列满时丢弃日志，避免阻塞
        
        def close(self):
            self.running = False
            self.log_queue.put(None)
            self.thread.join(timeout=1)
            super().close()
    
    # 配置异步日志（控制台）
    async_handler = AsyncStreamHandler(sys.stdout)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    async_handler.setFormatter(log_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 根日志器设为 DEBUG 以允许所有日志通过
    root_logger.addHandler(async_handler)
    
    # 确保程序退出时正确关闭日志处理器
    atexit.register(async_handler.close)
    
    # --- 日志文件配置 ---
    from datetime import datetime
    
    # 创建强制刷新的文件处理器类（确保日志立即写入磁盘，防止丢失）
    class FlushingFileHandler(logging.FileHandler):
        """每次写入后立即刷新到磁盘的文件处理器"""
        def emit(self, record):
            super().emit(record)
            self.flush()  # 强制刷新缓冲区
    
    # 日志目录放在 result/ 下
    if getattr(sys, 'frozen', False):
        log_dir = os.path.join(os.path.dirname(sys.executable), '_internal', 'result')
    else:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file_path = os.path.join(log_dir, f'log_{timestamp}.txt')
    
    # 使用强制刷新的文件处理器
    file_handler = FlushingFileHandler(log_file_path, encoding='utf-8', delay=False)
    file_handler.setLevel(logging.DEBUG)  # 始终为 DEBUG 级别
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # 确保程序退出时关闭文件处理器
    atexit.register(file_handler.close)
    
    logging.info(f"UI日志文件: {log_file_path}")
    
    # --- 确保过滤列表文件存在 ---
    try:
        from manga_translator.utils.text_filter import ensure_filter_list_exists
        ensure_filter_list_exists()
    except Exception as e:
        logging.warning(f"创建过滤列表文件失败: {e}")
    
    # --- 崩溃捕获 (faulthandler) ---
    # 启用 faulthandler 以捕获 C++ 级别的崩溃 (Segmentation Fault 等)
    # 将崩溃信息直接写入同一个日志文件
    import faulthandler
    # 使用 file_handler 的流对象
    faulthandler.enable(file=file_handler.stream, all_threads=True)
    logging.info("已启用崩溃捕获 (faulthandler)，崩溃信息将记录在此文件中")

    # --- 环境设置 ---
    # Windows特殊处理：必须在创建QApplication之前设置AppUserModelID
    if sys.platform == 'win32':
        try:
            import ctypes
            # 设置AppUserModelID，让Windows识别这是独立应用
            myappid = 'manga.translator.ui.1.0'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
    
    # 1. 创建 QApplication 实例
    app = QApplication(sys.argv)
    app.setApplicationName("Manga Translator")
    app.setOrganizationName("Manga Translator")
    
    # 设置 Qt 异常处理钩子（捕获信号槽中的异常）
    def qt_message_handler(mode, context, message):
        """Qt 消息处理器，捕获 Qt 内部错误"""
        from PyQt6.QtCore import QtMsgType
        if mode == QtMsgType.QtFatalMsg:
            logging.critical(f"Qt Fatal: {message} (file: {context.file}, line: {context.line})")
        elif mode == QtMsgType.QtCriticalMsg:
            logging.error(f"Qt Critical: {message}")
        elif mode == QtMsgType.QtWarningMsg:
            # 过滤一些常见的无害警告
            if "QWindowsWindow::setGeometry" not in message:
                logging.warning(f"Qt Warning: {message}")
        # Debug 和 Info 级别不记录，避免日志过多
    
    from PyQt6.QtCore import qInstallMessageHandler
    qInstallMessageHandler(qt_message_handler)
    
    # 设置应用程序图标（用于任务栏/Dock）
    from PyQt6.QtGui import QIcon
    
    # 确定图标路径
    if getattr(sys, 'frozen', False):
        # 打包环境：图标在 _internal 目录下
        exe_dir = os.path.dirname(sys.executable)
        icon_path = os.path.join(exe_dir, '_internal', 'doc', 'images', 'icon.ico')
    else:
        # 开发环境
        base_icon_dir = os.path.join(os.path.dirname(__file__), '..', 'doc', 'images')
        # macOS 使用 .icns 文件
        if sys.platform == 'darwin':
            icon_path = os.path.join(base_icon_dir, 'icon.icns')
        else:
            icon_path = os.path.join(base_icon_dir, 'icon.ico')
    
    icon_path = os.path.abspath(icon_path)
    app_icon = None
    
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)

    # 2. 初始化所有服务
    # 设置正确的根目录：打包后指向_internal，开发时指向项目根目录
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller打包环境：所有资源在_internal目录
        root_dir = sys._MEIPASS
    else:
        # 开发环境：资源在项目根目录
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if not init_services(root_dir):
        logging.fatal("Fatal: Service initialization failed.")
        sys.exit(1)

    # 3. 创建并显示主窗口
    main_window = MainWindow()
    
    # 确保主窗口也设置了图标
    if app_icon and not app_icon.isNull():
        main_window.setWindowIcon(app_icon)
    
    main_window.show()
    
    # Windows特殊处理：强制窗口显示在最前面
    if sys.platform == 'win32':
        # 设置窗口标志，使其显示在最前面
        from PyQt6.QtCore import Qt
        main_window.setWindowFlags(main_window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        main_window.show()  # 重新显示以应用标志
        # 立即取消置顶，避免一直在最前面
        main_window.setWindowFlags(main_window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        main_window.show()  # 再次显示以应用标志
        
        # 设置图标并刷新
        if app_icon and not app_icon.isNull():
            main_window.setWindowIcon(app_icon)
            app.processEvents()  # 强制处理事件，刷新任务栏图标
    
    main_window.raise_()  # 将窗口提升到最前面
    main_window.activateWindow()  # 激活窗口
    app.processEvents()  # 处理所有待处理事件

    # 4. 启动事件循环
    ret = app.exec()
    logging.info("Exiting application...")
    
    # 确保所有日志都写入文件
    try:
        # 刷新所有日志处理器
        for handler in logging.root.handlers:
            handler.flush()
        
        # 关闭异步日志处理器
        if 'async_handler' in locals():
            async_handler.close()
        
        # 关闭文件日志处理器
        if 'file_handler' in locals():
            file_handler.flush()
            file_handler.close()
    except Exception as e:
        print(f"关闭日志处理器时出错: {e}", file=sys.stderr)
    
    # 使用 os._exit 强制退出，防止守护线程阻塞
    os._exit(ret)

if __name__ == '__main__':
    # 在创建QApplication之前设置DPI策略，这是解决DPI问题的另一种稳妥方式
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    main()
