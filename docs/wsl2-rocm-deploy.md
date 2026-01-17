# WSL2 + ROCm 部署指南 (RX 9070 GRE / gfx1201)

本指南帮助你在 Windows 11 上通过 WSL2 运行漫画翻译器项目，使用 AMD ROCm 进行 GPU 加速。

## 前置条件

- Windows 11 22H2 或更高版本
- AMD Radeon RX 9070 GRE (gfx1201)
- 至少 16GB 内存
- 至少 50GB 可用磁盘空间

---

## 第一阶段：Windows 端配置

### 1.1 安装支持 WSL 的 AMD 驱动

**重要**：必须安装支持 WSL2 的特定驱动版本。

1. 访问 [AMD 驱动下载页面](https://www.amd.com/en/support)
2. 搜索 "Adrenalin Edition for WSL2" 或下载最新的 Adrenalin 驱动（25.x.x 版本）
3. 安装时选择"完整安装"
4. **重启电脑**

### 1.2 启用 WSL2

打开 PowerShell（管理员）：

```powershell
# 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用虚拟机功能
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 重启电脑
Restart-Computer
```

重启后继续：

```powershell
# 设置 WSL2 为默认版本
wsl --set-default-version 2

# 更新 WSL 内核
wsl --update
```

### 1.3 安装 Ubuntu 24.04

```powershell
# 安装 Ubuntu 24.04
wsl --install Ubuntu-24.04

# 等待安装完成，设置用户名和密码
```

### 1.4 验证 WSL2 安装

```powershell
# 检查 WSL 版本
wsl -l -v

# 应该看到:
# NAME            STATE           VERSION
# Ubuntu-24.04    Running         2
```

---

## 第二阶段：WSL2 内配置 ROCm

### 2.1 进入 WSL

```powershell
wsl -d Ubuntu-24.04
```

以下所有命令都在 WSL Ubuntu 内执行。

### 2.2 系统更新

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.3 安装 ROCm

```bash
# 下载 amdgpu-install 脚本
wget https://repo.radeon.com/amdgpu-install/6.4.2.1/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb

# 安装脚本
sudo apt install ./amdgpu-install_6.4.60402-1_all.deb

# 安装 ROCm（WSL 模式，不需要 DKMS）
sudo amdgpu-install --usecase=wsl,rocm --no-dkms -y

# 清理下载文件
rm amdgpu-install_6.4.60402-1_all.deb
```

### 2.4 验证 ROCm 安装

```bash
# 检查 GPU 是否被识别
rocminfo

# 应该看到类似:
# Agent 2
#   Name:                    gfx1201
#   Marketing Name:          AMD Radeon RX 9070 GRE
#   Vendor Name:             AMD

# 检查 rocm-smi
rocm-smi

# 应该显示 GPU 信息和温度
```

**如果 rocminfo 没有检测到 GPU**：
1. 确认 Windows 端 AMD 驱动是支持 WSL 的版本
2. 尝试禁用集成显卡（在设备管理器或 BIOS 中）
3. 重启 WSL：`wsl --shutdown` 然后重新进入

---

## 第三阶段：Python 环境配置

### 3.1 安装 Miniconda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 初始化
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 清理
rm Miniconda3-latest-Linux-x86_64.sh
```

### 3.2 创建项目环境

```bash
# 创建 Python 3.12 环境
conda create -n manga-wsl python=3.12 -y
conda activate manga-wsl
```

### 3.3 安装 PyTorch ROCm 版本

```bash
# 安装 PyTorch ROCm 6.4 版本（官方稳定版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# 或者使用 AMD 官方仓库的版本（可能更稳定）
# pip install torch torchvision torchaudio --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/
```

### 3.4 验证 PyTorch GPU 支持

```bash
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA/ROCm 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
    
    # 简单测试
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f'GPU 计算测试: 成功')
"
```

---

## 第四阶段：部署漫画翻译器项目

### 4.1 克隆项目

```bash
# 创建工作目录
mkdir -p ~/projects
cd ~/projects

# 从 Gitee 克隆（国内更快）
git clone https://gitee.com/long-xin12/manga-translator-ui.git manga-translator

# 或从 Windows 复制（如果网络慢）
# cp -r /mnt/d/漫画/12 ~/projects/manga-translator

cd manga-translator
```

### 4.2 安装项目依赖

```bash
# 确保在正确的环境
conda activate manga-wsl

# 安装项目依赖
pip install -r requirements.txt

# 如果有 requirements-gpu.txt
# pip install -r requirements-gpu.txt
```

### 4.3 配置模型路径

```bash
# 创建符号链接到 Windows 上的模型文件（避免重复下载）
# 假设模型在 D:\漫画\12\models

ln -s /mnt/d/漫画/12/models ~/projects/manga-translator/models

# 或者复制配置文件
cp /mnt/d/漫画/12/examples/config.json ~/projects/manga-translator/examples/
```

### 4.4 测试运行

```bash
# 运行简单测试
python -c "from manga_translator import MangaTranslator; print('导入成功')"

# 运行 CLI 测试（如果有）
# python main.py --help
```

---

## 第五阶段：配置 Qt GUI（可选）

WSL2 支持通过 WSLg 显示 Linux GUI 应用。

### 5.1 安装 Qt 依赖

```bash
# 安装 Qt 运行时依赖
sudo apt install -y \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libfontconfig1 \
    libdbus-1-3

# 安装 PyQt6
pip install PyQt6
```

### 5.2 测试 GUI 显示

```bash
# 测试 GUI 是否正常
python -c "
from PyQt6.QtWidgets import QApplication, QLabel
app = QApplication([])
label = QLabel('WSL2 GUI 测试成功!')
label.show()
app.exec()
"
```

### 5.3 运行翻译器 GUI

```bash
# 启动 GUI 界面
python desktop_qt_ui/main.py

# 或者根据项目实际入口
# python main_gui.py
```

---

## 常见问题排查

### Q1: rocminfo 不显示 GPU

```bash
# 检查 HSA 运行时
ls -la /dev/kfd /dev/dri

# 检查用户权限
groups  # 应该包含 video 和 render

# 添加用户到组
sudo usermod -aG video,render $USER
# 重新登录 WSL
```

### Q2: PyTorch 报错 "HIP error"

```bash
# 设置环境变量
export HSA_OVERRIDE_GFX_VERSION=11.0.1
export HIP_VISIBLE_DEVICES=0

# 添加到 .bashrc 永久生效
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.1' >> ~/.bashrc
echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc
```

### Q3: GUI 无法显示

```bash
# 检查 DISPLAY 变量
echo $DISPLAY  # 应该有值，如 :0

# 检查 WSLg
ls /mnt/wslg  # 应该存在

# 重启 WSL
# 在 PowerShell 中: wsl --shutdown
```

### Q4: 内存不足

```bash
# 在 Windows 端创建 .wslconfig 限制内存
# 文件位置: C:\Users\<用户名>\.wslconfig

[wsl2]
memory=12GB
swap=8GB
```

### Q5: MIOpen 编译错误

```bash
# 禁用即时编译，使用预编译内核
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# 或者安装预编译数据库
sudo apt install miopen-hip-gfx1201  # 如果有的话
```

---

## 性能优化

### 文件访问优化

```bash
# WSL2 访问 /mnt/d 等 Windows 路径较慢
# 建议将项目和模型放在 WSL 文件系统内

# 复制模型到 WSL
cp -r /mnt/d/漫画/12/models ~/projects/manga-translator/
```

### GPU 性能调优

```bash
# 设置性能模式
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

---

## 快速启动脚本

创建启动脚本 `~/start-manga.sh`：

```bash
#!/bin/bash

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate manga-wsl

# 设置环境变量
export HSA_OVERRIDE_GFX_VERSION=11.0.1
export HIP_VISIBLE_DEVICES=0
export MIOPEN_FIND_MODE=3

# 进入项目目录
cd ~/projects/manga-translator

# 启动 GUI
python desktop_qt_ui/main.py
```

```bash
# 添加执行权限
chmod +x ~/start-manga.sh

# 使用
~/start-manga.sh
```

---

## 从 Windows 直接启动

在 PowerShell 中创建快捷方式：

```powershell
# 创建启动脚本
@"
wsl -d Ubuntu-24.04 -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate manga-wsl && cd ~/projects/manga-translator && python desktop_qt_ui/main.py"
"@ | Out-File -FilePath "$env:USERPROFILE\Desktop\漫画翻译器-WSL.ps1" -Encoding UTF8
```

---

## 更新日志

- 2026-01-16: 初始版本，基于 ROCm 6.4.x 和 RX 9070 GRE (gfx1201)
