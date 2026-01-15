---
frameworks:
- ""
tasks: []
license: CC-BY-NC-4.0
---
# Manga Translator UI - 模型文件托管仓库

<div align="center">

[![主项目](https://img.shields.io/badge/%E4%B8%BB%E9%A1%B9%E7%9B%AE-manga--translator--ui-green)](https://github.com/hgmzhn/manga-translator-ui)
[![基于](https://img.shields.io/badge/%E5%9F%BA%E4%BA%8E-manga--image--translator-blue)](https://github.com/zyddnys/manga-image-translator)
[![模型](https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-Real--CUGAN-orange)](https://github.com/bilibili/ailab)
[![模型](https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-MangaJaNai-orange)](https://github.com/the-database/MangaJaNai)
[![OCR](https://img.shields.io/badge/OCR-PaddleOCR-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![OCR](https://img.shields.io/badge/OCR-MangaOCR-blue)](https://github.com/kha-white/manga-ocr)
[![OCR](https://img.shields.io/badge/OCR-PaddleOCR--VL--For--Manga-blue)](https://github.com/jzhang533/PaddleOCR-VL-For-Manga)
[![许可证](https://img.shields.io/badge/%E8%AE%B8%E5%8F%AF%E8%AF%81-CC--BY--NC--4.0-red)](LICENSE)

</div>

## 📦 仓库说明

这是 [Manga Translator UI](https://github.com/hgmzhn/manga-translator-ui) 项目的**模型文件托管仓库**。

本仓库托管了漫画翻译软件运行所需的所有 AI 模型文件，包括：
- 文字检测模型
- OCR 识别模型
- 图像修复模型
- 图像超分辨率模型
- 图像上色模型

## 🎯 使用说明

**用户无需手动下载本仓库的文件！**

当你运行 Manga Translator UI 软件时，程序会**自动检测缺失的模型**并从本仓库下载所需文件。

## 📋 模型列表

### 文字检测模型 (Detection)
- `detect-20241225.ckpt` - 默认文字检测器
- `comictextdetector.pt` / `comictextdetector.pt.onnx` - 漫画文字检测器
- `craft_mlt_25k.pth` / `craft_refiner_CTW1500.pth` - CRAFT 检测器
- `ysgyolo_1.2_OS1.0.onnx` - YOLO OBB 检测器

### OCR 识别模型
- `ocr.zip` - 32px OCR 模型
- `ocr_ar_48px.ckpt` + `alphabet-all-v7.txt` - 48px OCR 模型
- `ocr-ctc.zip` - CTC OCR 模型
- `manga_ocr_model.7z` - MangaOCR 模型（日文专用）
- `ch_PP-OCRv5_rec_server_infer.onnx` + `ppocrv5_dict.txt` - PaddleOCR 中文模型
- `korean_PP-OCRv5_rec_mobile_infer.onnx` + `ppocrv5_korean_dict.txt` - PaddleOCR 韩文模型
- `latin_PP-OCRv5_rec_mobile_infer.onnx` + `ppocrv5_latin_dict.txt` - PaddleOCR 拉丁文模型
- `PaddleOCR-VL-For-Manga` - PaddleOCR-VL-For-Manga 模型（日文漫画效果最好）

### 图像修复模型 (Inpainting)
- `inpainting.ckpt` - AOT 修复器
- `inpainting_lama_mpe.ckpt` - LAMA MPE 修复器
- `lama_large_512px.ckpt` - LAMA Large 修复器
- `lama_mpe_inpainting.onnx` - LAMA MPE ONNX 版本
- `lama_large_512px_inpainting.onnx` - LAMA Large ONNX 版本

### 图像超分辨率模型 (Upscaling)

#### Real-ESRGAN
- `4xESRGAN.pth` - 4倍超分模型
- `realesrgan-ncnn-vulkan` - NCNN 版本（Windows/macOS/Ubuntu）

#### Real-CUGAN (17 个模型)
- SE 系列：`up2x/3x/4x-latest-conservative/denoise1x/denoise2x/denoise3x/no-denoise.pth`
- PRO 系列：`pro-conservative/denoise3x/no-denoise-up2x/3x.pth`

#### MangaJaNai (17 个模型)
- MangaJaNai 2x 系列：`2x_MangaJaNai_1200p/1300p/1400p/1500p/1600p/1920p/2048p_V1_ESRGAN.pth`
- MangaJaNai 4x 系列：`4x_MangaJaNai_1200p/1300p/1400p/1500p/1600p/1920p/2048p_V1_ESRGAN.pth`
- IllustrationJaNai 系列：`2x/4x_IllustrationJaNai_V1_ESRGAN.pth`、`4x_IllustrationJaNai_V1_DAT2.pth`

#### Waifu2x
- `waifu2x-ncnn-vulkan` - NCNN 版本（Windows/macOS/Ubuntu）

### 图像上色模型 (Colorization)
- `manga-colorization-v2-generator.zip` - 上色生成器
- `manga-colorization-v2-net_rgb.pth` - RGB 网络

## 📊 统计信息

- **模型总数**：64 个文件
- **总大小**：约 5-8 GB（取决于选择的模型）
- **来源**：GitHub Release + HuggingFace

## 🔗 相关链接

- **主项目地址**：https://github.com/hgmzhn/manga-translator-ui
- **原始项目**：https://github.com/zyddnys/manga-image-translator
- **问题反馈**：https://github.com/hgmzhn/manga-translator-ui/issues

## 📝 模型来源与协议

本仓库的模型文件来自以下开源项目，**各模型遵守其原始项目的开源协议**：

- [manga-image-translator](https://github.com/zyddnys/manga-image-translator) - 主要模型来源
- [manga-ocr](https://github.com/kha-white/manga-ocr) - 日文 OCR 模型
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 多语言 OCR 模型
- [PaddleOCR-VL-For-Manga](https://github.com/jzhang533/PaddleOCR-VL-For-Manga) - 日文漫画 OCR 模型
- [Real-CUGAN](https://github.com/bilibili/ailab) - B站 AI Lab 超分模型
- [MangaJaNai](https://github.com/the-database/MangaJaNai) - 漫画专用超分模型 **(CC BY-NC 4.0，仅限非商业用途)**
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 通用超分模型
- [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) - 动漫图像超分模型

## 🙏 致谢

感谢所有开源项目的作者和贡献者，让这个项目得以实现！

- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) - 核心翻译引擎
- [bilibili/ailab](https://github.com/bilibili/ailab) - Real-CUGAN 超分辨率模型
- [the-database/MangaJaNai](https://github.com/the-database/MangaJaNai) - MangaJaNai/IllustrationJaNai 超分辨率模型
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - PaddleOCR 模型支持
- [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr) - MangaOCR 模型支持
- [jzhang533/PaddleOCR-VL-For-Manga](https://github.com/jzhang533/PaddleOCR-VL-For-Manga) - 提供 PaddleOCR-VL-For-Manga 模型支持
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Real-ESRGAN 超分模型
- [nihui/waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) - Waifu2x 超分模型

---

**最后更新时间**：2025-01-08
