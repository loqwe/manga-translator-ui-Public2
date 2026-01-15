# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Manga Translator UI - 漫画翻译工具，支持自动检测、OCR识别、翻译和嵌字。包含两个主要组件：
- **desktop_qt_ui/** - PyQt6 桌面应用（主界面）
- **manga_translator/** - 核心翻译引擎（CLI/Web API）

## Development Commands

### Environment Setup
```bash
# Activate conda environment (required before all commands)
conda activate manga-env

# Install dependencies (choose one)
pip install -r requirements_gpu.txt   # NVIDIA GPU (CUDA 12.x)
pip install -r requirements_amd.txt   # AMD GPU (RX 7000/9000 only)
pip install -r requirements_cpu.txt   # CPU version
```

### Running the Application
```bash
# Desktop UI (PyQt6)
python -m desktop_qt_ui.main

# CLI translation
python -m manga_translator local -i <image_or_folder> [-o output_dir]
python -m manga_translator -i <image>  # shorthand for local mode

# Web server (API + Web UI)
python -m manga_translator web --host 127.0.0.1 --port 8000 [--use-gpu]
```

### Linting
```bash
# Run ruff linter on desktop_qt_ui
ruff check desktop_qt_ui/

# Auto-fix issues
ruff check --fix desktop_qt_ui/
```

### Building Packages
```bash
cd packaging
python build_packages.py <version> --build cpu  # CPU version
python build_packages.py <version> --build gpu  # GPU version
```

## Architecture

### Core Translation Pipeline (`manga_translator/`)
```
manga_translator/
├── __main__.py          # CLI entry point, dispatches to modes
├── args.py              # Argument parsing for all modes
├── manga_translator.py  # Core MangaTranslator class
├── concurrent_pipeline.py # Async processing pipeline
├── detection/           # Text bubble detection (AI models)
├── ocr/                 # OCR engines (PaddleOCR, MangaOCR, etc.)
├── translators/         # Translation backends
│   ├── openai.py/openai_hq.py   # OpenAI GPT (text/multimodal)
│   ├── gemini.py/gemini_hq.py   # Google Gemini (text/multimodal)
│   └── sakura.py                # Japanese-optimized translator
├── inpainting/          # Image restoration (remove text)
├── rendering/           # Text rendering/typesetting
├── upscaling/           # Image super-resolution
├── server/              # Web API (FastAPI)
│   ├── routes/          # API endpoints
│   └── static/          # Web UI assets
└── mode/                # Execution modes
    ├── local.py         # CLI batch processing
    ├── ws.py            # WebSocket mode
    └── share.py         # Shared/API mode
```

### Desktop UI (`desktop_qt_ui/`)
```
desktop_qt_ui/
├── main.py              # App entry point
├── main_window.py       # Main window container
├── app_logic.py         # Business logic (connects UI to translator)
├── editor/              # Visual editor for text boxes
├── services/            # Service layer (config, state management)
├── widgets/             # Reusable UI components
└── locales/             # i18n translations (zh_CN, en_US, ja_JP, etc.)
```

### Data Flow
1. **Desktop UI**: `main_window.py` → `app_logic.py` → `manga_translator/` (via subprocess or direct call)
2. **CLI**: `__main__.py` → `mode/local.py` → `concurrent_pipeline.py` → individual modules
3. **Web API**: `server/main.py` → `routes/` → `manga_translator.py`

## Key Technical Details

- **Python**: 3.12 required
- **GUI Framework**: PyQt6
- **Deep Learning**: PyTorch (with CUDA/ROCm/MPS/CPU backends)
- **Web Framework**: FastAPI (for web mode)
- **Config**: JSON-based configs in `examples/` and `presets/`
- **Environment Variables**: `.env` file for API keys (OPENAI_API_KEY, etc.)

## Code Style

- PEP 8 compliant
- Ruff linter config at `desktop_qt_ui/ruff.toml` (ignores E501 line length, E701, E402)
- Comments in English, UI strings support i18n
- snake_case for functions/variables, PascalCase for classes
