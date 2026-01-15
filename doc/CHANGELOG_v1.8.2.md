# 更新日志 v1.8.2

## 🎉 新功能

### AMD ROCm GPU 支持（实验性）

- ✨ **AMD 显卡加速支持**
  - 支持 AMD ROCm PyTorch
  - 支持 RX 5000/6000/7000/9000 系列（RDNA 1/2/3/4架构）
  - 自动识别AMD显卡型号和 gfx 版本
  - 新增 `requirements_amd.txt` 依赖文件
  - ⚠️ AMD GPU 仅支持安装脚本方式，不支持打包版本

### 增强的 GPU 检测系统

- 🔍 **多层GPU检测机制**（不再依赖 wmic）
  - PowerShell CIM（Windows 8+）
  - wmic（兼容老系统）
  - PowerShell WMI（更老的PowerShell）
  - 注册表查询（最底层）
  - wmi Python 库（按需安装）

- 🎯 **NVIDIA CUDA 版本检测**
  - 使用 nvidia-smi 自动检测 CUDA 版本
  - 自动验证 CUDA >= 12 版本要求
  - CUDA < 12 时提示用户更新驱动或使用 CPU 版本
  - 显示驱动版本信息

### 脚本改进

- 🗑️ **pip 缓存清理选项**
  - 步骤1-首次安装.bat 完成后询问是否清理缓存
  - 帮助释放磁盘空间

- 🔄 **AMD 依赖更新支持**
  - 步骤4-更新维护.bat 现在支持 AMD ROCm PyTorch 更新
  - detect_torch_type.py 支持检测 AMD ROCm PyTorch

## 🐛 Bug 修复

### 编辑器相关修复

- 修复了编辑器中的若干 bug
- 改进了图形项目处理逻辑
- 优化了编辑器控制器功能

## 📝 文档更新

- 📖 **README.md**
  - 添加 AMD GPU 支持说明
  - 更新安装方式说明
  - 警告 AMD 不支持打包版本

- 📖 **doc/INSTALLATION.md**
  - 详细的 AMD GPU 配置要求
  - AMD 显卡支持列表
  - 安装流程中的 GPU 检测说明

- 🙏 **致谢**
  - 添加 Real-CUGAN 模型来源（bilibili/ailab）

## 🔧 技术细节

### 新增文件

- `requirements_amd.txt` - AMD GPU 专用依赖文件（包含详细的 gfx 版本对应表）

### 修改文件

- `packaging/launch.py` - 核心依赖安装逻辑
- `packaging/detect_torch_type.py` - PyTorch 类型检测
- `packaging/VERSION` - 版本号更新
- `步骤1-首次安装.bat` - 安装脚本

## ⚠️ 注意事项

### AMD GPU 用户

- AMD ROCm PyTorch 是实验性功能，稳定性可能不如 NVIDIA CUDA 版本
- Windows 上 ROCm 支持有限，Linux 下体验更好
- 首次运行可能需要编译某些操作，请耐心等待
- 如果遇到问题，建议使用 CPU 版本

### NVIDIA GPU 用户

- **CUDA 12.x 要求**：GPU 版本现在要求 CUDA 12.x
- 驱动要求：>= 525.60.13
- 如果您的 CUDA 版本低于 12，安装脚本会提示您更新驱动

---

## 🚀 升级方法

### 已安装用户

双击 `步骤4-更新维护.bat` → 选择"完整更新"

### 新用户

请参考 [安装指南](INSTALLATION.md)

