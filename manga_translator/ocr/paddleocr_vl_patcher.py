"""
PaddleOCR-VL 模型文件自动修补工具

在加载模型前自动应用必要的修改，避免手动修改模型文件
"""

import os
import sys


def patch_paddleocr_vl_files(model_path: str):
    """
    自动修补 PaddleOCR-VL 模型文件
    
    Args:
        model_path: 模型目录路径
    """
    # 1. 创建 __init__.py 文件（如果不存在）
    init_file = os.path.join(model_path, '__init__.py')
    if not os.path.exists(init_file):
        init_content = '''# 将 ernie4_5 映射到当前模块，避免 transformers 查找不存在的模块
import sys
from pathlib import Path

# 获取当前模块路径
current_module_path = Path(__file__).parent

# 创建模块别名
if 'transformers.models.ernie4_5' not in sys.modules:
    # 将当前模块注册为 ernie4_5
    sys.modules['transformers.models.ernie4_5'] = sys.modules[__name__]
    sys.modules['transformers.models.ernie4_5.configuration_ernie4_5'] = sys.modules.get(f'{__name__}.configuration_paddleocr_vl')
    sys.modules['transformers.models.ernie4_5.modeling_ernie4_5'] = sys.modules.get(f'{__name__}.modeling_paddleocr_vl')

# 同时映射 ernie4_5_moe
if 'transformers.models.ernie4_5_moe' not in sys.modules:
    sys.modules['transformers.models.ernie4_5_moe'] = sys.modules[__name__]
    sys.modules['transformers.models.ernie4_5_moe.configuration_ernie4_5_moe'] = sys.modules.get(f'{__name__}.configuration_paddleocr_vl')
    sys.modules['transformers.models.ernie4_5_moe.modeling_ernie4_5_moe'] = sys.modules.get(f'{__name__}.modeling_paddleocr_vl')
'''
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)
    
    # 2. 修补 modeling_paddleocr_vl.py 文件（注释掉 @check_model_inputs 装饰器）
    modeling_file = os.path.join(model_path, 'modeling_paddleocr_vl.py')
    if os.path.exists(modeling_file):
        with open(modeling_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否需要修补
        if '@check_model_inputs' in content and '# @check_model_inputs' not in content:
            # 替换 @check_model_inputs 为注释
            content = content.replace(
                '    @check_model_inputs',
                '    # @check_model_inputs  # 注释掉此装饰器以避免参数检查问题'
            )
            
            with open(modeling_file, 'w', encoding='utf-8') as f:
                f.write(content)


def register_ernie_modules(model_path: str):
    """
    注册 ernie4_5 模块映射
    
    Args:
        model_path: 模型目录路径
    """
    # 预先导入模块以注册 ernie4_5 映射
    if os.path.exists(model_path):
        sys.path.insert(0, model_path)
        try:
            import __init__ as paddleocr_vl_init
        except:
            pass
        finally:
            if model_path in sys.path:
                sys.path.remove(model_path)
