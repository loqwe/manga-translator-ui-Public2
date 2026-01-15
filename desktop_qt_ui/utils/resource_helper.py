"""
资源路径辅助函数
用于处理开发环境和PyInstaller打包环境的资源路径
"""
import os
import sys


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller

    Args:
        relative_path: 相对于项目根目录的路径

    Returns:
        绝对路径
    """
    try:
        # PyInstaller打包环境
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境：从desktop_qt_ui向上一级到项目根目录
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.join(base_path, relative_path)
