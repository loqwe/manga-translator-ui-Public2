# 开发者指南

本文档介绍项目结构、开发环境配置和构建打包流程。

---

## ⚠️ 重要提示

运行开发命令前，请先在项目目录激活虚拟环境：

```bash
# Windows / Linux / macOS
conda activate manga-env
```

---

## 👨‍💻 项目结构

```
manga-translator-ui-package/
├── desktop_qt_ui/          # PyQt6 桌面应用（主界面）
│   ├── main.py            # 应用入口
│   ├── main_window.py     # 主窗口
│   ├── app_logic.py       # 业务逻辑
│   ├── editor/            # 可视化编辑器
│   ├── services/          # 服务层
│   └── widgets/           # UI 组件
├── manga_translator/       # 核心翻译引擎
│   ├── translators/       # 翻译器实现
│   ├── ocr/              # OCR 模块
│   ├── detection/        # 文本检测
│   ├── inpainting/       # 图像修复
│   └── rendering/        # 文本渲染
├── fonts/                 # 字体文件
├── models/                # AI 模型
├── examples/              # 配置示例
└── requirements_*.txt     # 依赖列表
```

---

## ⚙️ 环境配置

### 系统要求

- **Python 3.12**
- **Windows 10/11** 或 **Linux**
- **Git**（用于克隆代码）

### 安装依赖

**CPU 版本：**
```bash
pip install -r requirements_cpu.txt
```

**GPU 版本（需要 CUDA 12.x）：**
```bash
pip install -r requirements_gpu.txt
```

---

## 🚀 运行开发版

### 运行 PyQt6 界面

```bash
python -m desktop_qt_ui.main
```

### 运行旧版 CustomTkinter 界面

```bash
python -m desktop-ui.main
```

---

## 📦 构建打包

### 安装 PyInstaller

```bash
pip install pyinstaller
```

### 构建脚本位置

构建脚本位于 `packaging/` 目录：
- `packaging/build_packages.py` - 构建脚本
- `packaging/manga-translator-cpu.spec` - CPU 版本配置
- `packaging/manga-translator-gpu.spec` - GPU 版本配置

### 构建 CPU 版本

```bash
cd packaging
python build_packages.py <version> --build cpu
```

### 构建 GPU 版本

```bash
cd packaging
python build_packages.py <version> --build gpu
```

### 示例：构建 1.6.0 版本

```bash
cd packaging
python build_packages.py 1.6.0 --build cpu
```

---

## 🔧 开发流程

### 1. 克隆仓库

```bash
git clone https://github.com/hgmzhn/manga-translator-ui.git
cd manga-translator-ui
```

### 2. 创建虚拟环境

```bash
python -m venv venv
```

**Windows：**
```bash
venv\Scripts\activate
```

**Linux/Mac：**
```bash
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements_cpu.txt
```

或

```bash
pip install -r requirements_gpu.txt
```

### 4. 运行开发版

```bash
python -m desktop_qt_ui.main
```

---

## 🐛 调试技巧

### 启用详细日志

在"基础设置"中勾选"详细日志"，查看详细的处理过程。

### 查看中间结果

开启详细日志后，程序会生成调试文件，包括：
- 检测结果
- OCR 识别结果
- 蒙版生成过程
- 修复效果

详见 [调试指南](DEBUGGING.md)。

---

## 📝 代码规范

### Python 代码风格

- 遵循 PEP 8 规范
- 使用 4 空格缩进
- 函数和变量使用 snake_case
- 类名使用 PascalCase

### 提交规范

- 使用清晰的 commit message
- 一个 commit 只做一件事
- 提交前测试代码

---

## 🔗 相关资源

### 核心引擎

- [manga-image-translator](https://github.com/zyddnys/manga-image-translator) - 核心翻译引擎

### 依赖库

- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架
- [Pillow](https://pillow.readthedocs.io/) - 图像处理
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

---

## 📄 许可证

本项目基于 **GPL-3.0** 许可证开源。

核心翻译引擎来自 [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)。

---

## 🙏 致谢

- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) - 核心翻译引擎
- 所有贡献者和用户的支持

