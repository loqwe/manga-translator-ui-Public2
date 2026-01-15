# 更新日志 v1.8.6

## Bug 修复

- **修复 conda 编码错误**：修复 conda 读取 environments.txt 时的 UnicodeDecodeError，解决环境检测和激活失败的问题
- **优化环境检测逻辑**：将 `conda env list` 改为 `conda info --envs`，提高环境检测可靠性
- **简化环境创建**：移除对旧版本路径环境的兼容代码，统一使用命名环境 manga-env
- **优化输出信息**：移除调试日志，精简安装过程输出，保留关键提示信息

## 升级方法

已安装用户：双击 `步骤4-更新维护.bat` → 选择"完整更新"
