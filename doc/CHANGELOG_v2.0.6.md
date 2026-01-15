# v2.0.6 更新日志

发布日期：2026-01-06

## 🐛 修复

- 修复 Qt 编辑器导出 PSD 文件相关问题：
  - 修复导出图片时未导出 PSD 文件的问题：在 export_service 中添加 cli 配置传递，确保 PSD 导出配置被正确传递给后端渲染引擎
  - 修复 PSD 文件保存路径错误的问题：PSD 文件现在会正确保存到原图所在目录的 `manga_translator_work/psd/` 下，而不是临时目录，避免导出后文件丢失
  - 修复 PSD 导出时无法找到 inpainted 图片的问题：现在使用原图路径查找 `manga_translator_work/inpainted/` 下的修复图，而不是在临时目录查找
- 修复 Qt 编辑器导出 CMYK 模式图片失败的问题：保存 PNG/WEBP 格式时自动将 CMYK 模式转换为 RGB 模式
- 修复替换翻译模式检测到无字图时不输出原图的问题：无字图现在会直接输出生肉原图
- 修复 ctd_replace 模块导入错误：将 `det_rearrange_forward` 从正确的位置导入
- 修复 OpenAI 翻译器使用新模型（如 o1、gpt-4.1）时闪退的问题：当 `max_tokens` 为空时不再传递该参数
- 修复某些情况下 UI 调用翻译程序闪退的问题
- 修复渲染时透视变换矩阵计算失败导致崩溃的问题：当 `findHomography` 返回无效矩阵时跳过该文本区域

## ✨ 新功能

- 新增 PaddleOCR-VL-For-Manga OCR 模型支持：针对日文漫画优化的视觉语言模型，效果最好但最吃配置

## 🔧 优化

- 优化替换翻译直接粘贴模式的蒙版处理：DenseCRF 边缘优化结果与扩张蒙版叠加，保留更完整的文字区域
- 增加替换翻译模式蒙版膨胀配置：支持通过 `kernel_size` 和 `mask_dilation_offset` 参数控制蒙版膨胀力度
