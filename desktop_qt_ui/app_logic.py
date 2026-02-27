
"""
应用业务逻辑层
处理应用的核心业务逻辑，与UI层分离
"""
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from manga_translator.config import (
    Alignment,
    Colorizer,
    Detector,
    Direction,
    Inpainter,
    InpaintPrecision,
    Ocr,
    Renderer,
    Translator,
    Upscaler,
)
from manga_translator.save import OUTPUT_FORMATS
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog

from services import (
    get_config_service,
    get_file_service,
    get_i18n_manager,
    get_logger,
    get_preset_service,
    get_state_manager,
    get_translation_service,
)
from services.state_manager import AppStateKey


@dataclass
class AppConfig:
    """应用配置信息"""
    window_size: tuple = (1200, 800)
    theme: str = "dark"
    language: str = "zh_CN"
    auto_save: bool = True
    max_recent_files: int = 10

class MainAppLogic(QObject):
    """主页面业务逻辑控制器"""
    files_added = pyqtSignal(list)
    files_cleared = pyqtSignal()
    file_removed = pyqtSignal(str)
    config_loaded = pyqtSignal(dict)
    output_path_updated = pyqtSignal(str)
    task_completed = pyqtSignal(list)
    task_file_completed = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    render_setting_changed = pyqtSignal()
    failed_files_recorded = pyqtSignal(dict)  # 失败文件记录信号: {folder_path: [filename, ...]}

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.config_service = get_config_service()
        self.translation_service = get_translation_service()
        self.file_service = get_file_service()
        self.state_manager = get_state_manager()
        self.i18n = get_i18n_manager()
        self.preset_service = get_preset_service()

        self.thread = None
        self.worker = None
        self.saved_files_count = 0
        self.saved_files_list = []  # 收集所有保存的文件路径

        self.source_files: List[str] = [] # Holds both files and folders
        self.file_to_folder_map: Dict[str, Optional[str]] = {} # 记录文件来自哪个文件夹
        self.excluded_subfolders: set = set() # 记录被删除的子文件夹路径
        self.folder_tree_cache: Dict[str, dict] = {} # 缓存文件夹的完整树结构 {top_folder: tree_structure}

        self.app_config = AppConfig()
        self._ui_log("主页面应用业务逻辑初始化完成")
    
    def _t(self, key: str, **kwargs) -> str:
        """翻译辅助方法"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key
    
    def _ui_log(self, message: str, level: str = "INFO"):
        """
        输出到UI日志列表和日志文件
        使用 root logger 确保写入 main.py 配置的日志文件
        """
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        formatted_msg = f"{timestamp} - {level} - {message}"
        
        # 使用 root logger 输出到日志文件和控制台
        try:
            root_logger = logging.getLogger()
            if level == "ERROR":
                root_logger.error(message)
            elif level == "DEBUG":
                root_logger.debug(message)
            elif level == "WARNING":
                root_logger.warning(message)
            else:
                root_logger.info(message)
        except Exception:
            # 如果logging失败，使用print作为备用
            print(formatted_msg)
        
        # 输出到UI日志列表
        try:
            self.log_message.emit(formatted_msg)
        except Exception:
            pass  # UI可能已崩溃，忽略错误


    @pyqtSlot(dict)
    def on_file_completed(self, result):
        """处理单个文件处理完成的信号并保存"""
        if not result.get('success') or not result.get('image_data'):
            self.logger.error(f"Skipping save for failed item: {result.get('original_path')}")
            return

        try:
            config = self.config_service.get_config()
            output_format = config.cli.format
            save_quality = config.cli.save_quality
            output_folder = config.app.last_output_path
            save_to_source_dir = config.cli.save_to_source_dir

            original_path = result['original_path']
            base_filename = os.path.basename(original_path)

            # 检查是否启用了"输出到原图目录"模式
            if save_to_source_dir:
                # 输出到原图所在目录的 manga_translator_work/result 子目录
                source_dir = os.path.dirname(original_path)
                final_output_folder = os.path.join(source_dir, 'manga_translator_work', 'result')
            else:
                # 原有逻辑：使用配置的输出目录
                if not output_folder:
                    self.logger.error(self._t("log_output_dir_not_set"))
                    self.state_manager.set_status_message(self._t("error_output_dir_not_set"))
                    return

                # 检查文件是否来自文件夹或压缩包
                source_folder = self.file_to_folder_map.get(original_path)

                if source_folder:
                    # 检查是否来自压缩包
                    if self.file_service.is_archive_file(source_folder):
                        # 文件来自压缩包，使用压缩包名称（不含扩展名）作为输出子目录
                        archive_name = os.path.splitext(os.path.basename(source_folder))[0]
                        final_output_folder = os.path.join(output_folder, archive_name)
                    else:
                        # 文件来自文件夹，保持相对路径结构
                        parent_dir = os.path.normpath(os.path.dirname(original_path))
                        relative_path = os.path.relpath(parent_dir, source_folder)
                        
                        # Normalize path and avoid adding '.' as a directory component
                        if relative_path == '.':
                            final_output_folder = os.path.join(output_folder, os.path.basename(source_folder))
                        else:
                            final_output_folder = os.path.join(output_folder, os.path.basename(source_folder), relative_path)
                    final_output_folder = os.path.normpath(final_output_folder)
                else:
                    # 文件是单独添加的，直接保存到输出目录
                    final_output_folder = output_folder

            # 确定文件扩展名
            if output_format and output_format != self._t("format_not_specified"):
                file_extension = f".{output_format}"
                output_filename = os.path.splitext(base_filename)[0] + file_extension
            else:
                # 保持原扩展名
                output_filename = base_filename

            final_output_path = os.path.join(final_output_folder, output_filename)

            os.makedirs(final_output_folder, exist_ok=True)

            save_kwargs = {}
            image_to_save = result['image_data']

            # Convert RGBA to RGB for JPEG format
            if final_output_path.lower().endswith(('.jpg', '.jpeg')):
                if image_to_save.mode == 'RGBA':
                    image_to_save = image_to_save.convert('RGB')
                save_kwargs['quality'] = save_quality
            elif final_output_path.lower().endswith('.webp'):
                save_kwargs['quality'] = save_quality

            image_to_save.save(final_output_path, **save_kwargs)

            # 更新translation_map.json
            self._update_translation_map(original_path, final_output_path)

            self.saved_files_count += 1
            self.saved_files_list.append(final_output_path)  # 收集保存的文件路径
            self.logger.info(self._t("log_file_saved_successfully", path=final_output_path))
            self.task_file_completed.emit({'path': final_output_path})

        except Exception as e:
            self.logger.error(self._t("log_file_save_error", path=result['original_path'], error=e))

    def _update_translation_map(self, source_path: str, translated_path: str):
        """在输出目录创建或更新 translation_map.json"""
        try:
            import json
            output_dir = os.path.dirname(translated_path)
            map_path = os.path.join(output_dir, 'translation_map.json')

            # 规范化路径以确保一致性
            source_path_norm = os.path.normpath(source_path)
            translated_path_norm = os.path.normpath(translated_path)

            translation_map = {}
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    try:
                        translation_map = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Could not decode {map_path}, creating a new one.")

            # 使用翻译后的路径作为键，确保唯一性
            translation_map[translated_path_norm] = source_path_norm

            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(translation_map, f, ensure_ascii=False, indent=4)

            self.logger.info(f"Updated translation_map.json: {translated_path_norm} -> {source_path_norm}")
        except Exception as e:
            self.logger.error(f"Failed to update translation_map.json: {e}")

    def _calculate_output_path(self, image_path: str, save_info: dict) -> str:
        """
        计算输出文件的完整路径（用于预检查文件是否存在）
        
        Args:
            image_path: 输入图片的路径
            save_info: 包含输出配置的字典
                - output_folder: 输出文件夹
                - format: 输出格式（可选）
                - save_to_source_dir: 是否输出到原图目录
                
        Returns:
            str: 计算后的输出文件完整路径
        """
        output_folder = save_info.get('output_folder')
        output_format = save_info.get('format')
        save_to_source_dir = save_info.get('save_to_source_dir', False)
        
        file_path = image_path
        parent_dir = os.path.normpath(os.path.dirname(file_path))
        
        # 检查是否启用了"输出到原图目录"模式
        if save_to_source_dir:
            # 输出到原图所在目录的 manga_translator_work/result 子目录
            final_output_dir = os.path.join(parent_dir, 'manga_translator_work', 'result')
        else:
            # 原有逻辑：使用配置的输出目录
            final_output_dir = output_folder
            
            # 检查文件是否来自文件夹
            source_folder = self.file_to_folder_map.get(image_path)
            if source_folder:
                # 检查是否来自压缩包
                if self.file_service.is_archive_file(source_folder):
                    # 文件来自压缩包，使用压缩包名称（不含扩展名）作为输出子目录
                    archive_name = os.path.splitext(os.path.basename(source_folder))[0]
                    final_output_dir = os.path.join(output_folder, archive_name)
                else:
                    # 文件来自文件夹，保持相对路径结构
                    relative_path = os.path.relpath(parent_dir, source_folder)
                    # Normalize path and avoid adding '.' as a directory component
                    if relative_path == '.':
                        final_output_dir = os.path.join(output_folder, os.path.basename(source_folder))
                    else:
                        final_output_dir = os.path.join(output_folder, os.path.basename(source_folder), relative_path)
                final_output_dir = os.path.normpath(final_output_dir)
        
        # 处理输出文件名和格式
        base_filename, _ = os.path.splitext(os.path.basename(file_path))
        if output_format and output_format.strip() and output_format.lower() not in ['none', '不指定']:
            output_filename = f"{base_filename}.{output_format}"
        else:
            output_filename = os.path.basename(file_path)
        
        final_output_path = os.path.join(final_output_dir, output_filename)
        return final_output_path

    @pyqtSlot(str)
    def on_worker_log(self, message):
        self.log_message.emit(message)

    @pyqtSlot()
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(None, self._t("Select Output Directory"))
        if folder:
            self.update_single_config('app.last_output_path', folder)
            self.output_path_updated.emit(folder)

    @pyqtSlot()
    def open_output_folder(self):
        import subprocess
        import sys
        output_dir = self.config_service.get_config().app.last_output_path
        if not output_dir or not os.path.isdir(output_dir):
            self.logger.warning(f"Output path is not a valid directory: {output_dir}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(os.path.realpath(output_dir))
            elif sys.platform == "darwin":
                subprocess.run(["open", output_dir])
            else:
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            self.logger.error(f"Failed to open output folder: {e}")

    @pyqtSlot()
    def open_source_folder(self):
        """
        从翻译后文件夹中查找 _source_path.txt，自动定位并打开翻译前的源文件夹。
        查找逻辑：在输出目录下查找最近修改的章节文件夹中的 _source_path.txt
        """
        import subprocess
        import sys
        from pathlib import Path
        
        output_dir = self.config_service.get_config().app.last_output_path
        if not output_dir or not os.path.isdir(output_dir):
            self.logger.warning(f"输出目录无效: {output_dir}")
            return
        
        source_path = self._find_source_path(output_dir)
        
        if not source_path:
            self.logger.warning(f"未找到 _source_path.txt 文件")
            self._ui_log("⚠️ 未找到源文件夹信息 (_source_path.txt)")
            return
        
        if not os.path.isdir(source_path):
            self.logger.warning(f"源文件夹不存在: {source_path}")
            self._ui_log(f"⚠️ 源文件夹不存在: {source_path}")
            return
        
        try:
            if sys.platform == "win32":
                os.startfile(os.path.realpath(source_path))
            elif sys.platform == "darwin":
                subprocess.run(["open", source_path])
            else:
                subprocess.run(["xdg-open", source_path])
            self._ui_log(f"📂 已打开源文件夹: {os.path.basename(source_path)}")
        except Exception as e:
            self.logger.error(f"打开源文件夹失败: {e}")
    
    def _find_source_path(self, output_dir: str) -> str:
        """
        在输出目录下递归查找 _source_path.txt，优先查找最近修改的目录。
        
        Returns:
            源文件夹路径，未找到则返回 None
        """
        from pathlib import Path
        
        # 策略1: 直接在输出目录下查找
        direct_source_file = os.path.join(output_dir, "_source_path.txt")
        if os.path.isfile(direct_source_file):
            return self._read_source_path_file(direct_source_file)
        
        # 策略2: 在子目录中查找，按修改时间排序（最新优先）
        subdirs = []
        try:
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    mtime = os.path.getmtime(item_path)
                    subdirs.append((item_path, mtime))
        except Exception as e:
            self.logger.error(f"扫描输出目录失败: {e}")
            return None
        
        # 按修改时间降序排序
        subdirs.sort(key=lambda x: x[1], reverse=True)
        
        # 在最近修改的目录中查找 _source_path.txt
        for subdir, _ in subdirs:
            source_file = os.path.join(subdir, "_source_path.txt")
            if os.path.isfile(source_file):
                return self._read_source_path_file(source_file)
            
            # 再向下查找一层（处理 作品名/Chapter X 结构）
            try:
                for sub_item in os.listdir(subdir):
                    sub_item_path = os.path.join(subdir, sub_item)
                    if os.path.isdir(sub_item_path):
                        nested_source_file = os.path.join(sub_item_path, "_source_path.txt")
                        if os.path.isfile(nested_source_file):
                            return self._read_source_path_file(nested_source_file)
            except Exception:
                continue
        
        return None
    
    def _read_source_path_file(self, source_file: str) -> str:
        """读取 _source_path.txt 文件内容并返回源路径"""
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source_path = f.read().strip()
            if source_path:
                return source_path
        except Exception as e:
            self.logger.error(f"读取 _source_path.txt 失败: {e}")
        return None

    def open_font_directory(self):
        import subprocess
        import sys
        # fonts目录在_internal里（打包后）或项目根目录（开发时）
        fonts_dir = os.path.join(self.config_service.root_dir, 'fonts')
        try:
            if not os.path.exists(fonts_dir):
                os.makedirs(fonts_dir)
            if sys.platform == "win32":
                os.startfile(fonts_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", fonts_dir])
            else:
                subprocess.run(["xdg-open", fonts_dir])
        except Exception as e:
            self.logger.error(f"Error opening font directory: {e}")

    def open_dict_directory(self):
        import subprocess
        import sys
        # dict目录在_internal里（打包后）或项目根目录（开发时）
        dict_dir = os.path.join(self.config_service.root_dir, 'dict')
        try:
            if not os.path.exists(dict_dir):
                os.makedirs(dict_dir)
            if sys.platform == "win32":
                os.startfile(dict_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", dict_dir])
            else:
                subprocess.run(["xdg-open", dict_dir])
        except Exception as e:
            self.logger.error(f"Error opening dict directory: {e}")

    def get_hq_prompt_options(self) -> List[str]:
        try:
            # dict目录在_internal里（打包后）或项目根目录（开发时）
            dict_dir = os.path.join(self.config_service.root_dir, 'dict')
            if not os.path.isdir(dict_dir):
                return []
            prompt_files = sorted([
                f for f in os.listdir(dict_dir)
                if f.lower().endswith('.json') and f not in [
                    'system_prompt_hq.json',
                    'system_prompt_line_break.json',
                    'glossary_extraction_prompt.json'
                ]
            ])
            return prompt_files
        except Exception as e:
            self.logger.error(f"Error scanning prompt directory: {e}")
            return []

    @pyqtSlot(str, str)
    def save_env_var(self, key: str, value: str):
        self.config_service.save_env_var(key, value)
        # 不再输出日志，避免刷屏

    # region 预设管理
    def get_presets_list(self) -> List[str]:
        """获取所有预设名称列表"""
        return self.preset_service.get_presets_list()
    
    @pyqtSlot(str)
    def save_preset(self, preset_name: str, copy_current: bool = False) -> bool:
        """保存预设
        
        Args:
            preset_name: 预设名称
            copy_current: 是否复制当前配置。False=创建空白预设，True=复制当前配置
        """
        try:
            if copy_current:
                # 复制当前配置模式：保存.env中所有翻译器的环境变量
                all_env_vars = self.config_service.load_env_vars()
                
                # 保存所有非空的环境变量
                filtered_env_vars = {}
                for key, value in all_env_vars.items():
                    if value and value.strip():
                        filtered_env_vars[key] = value
                
                success = self.preset_service.save_preset(preset_name, filtered_env_vars)
                if success:
                    # 不输出日志，避免刷屏
                    pass
            else:
                # 创建空白预设模式：只保存当前翻译器的空环境变量结构
                current_translator = self.config_service.get_config().translator.translator
                translator_env_vars = self.config_service.get_all_env_vars(current_translator)
                
                # 创建空白的环境变量字典
                empty_env_vars = {key: "" for key in translator_env_vars}
                
                success = self.preset_service.save_preset(preset_name, empty_env_vars)
                if success:
                    self._ui_log(f"预设已创建: {preset_name} (空白预设)")
            
            if not success:
                self._ui_log(f"保存预设失败: {preset_name}", "ERROR")
            return success
        except Exception as e:
            self.logger.error(f"保存预设失败: {e}")
            self._ui_log(f"保存预设失败: {e}", "ERROR")
            return False
    
    @pyqtSlot(str)
    def load_preset(self, preset_name: str) -> bool:
        """加载预设并更新环境变量到.env"""
        try:
            # 加载预设文件
            preset_env_vars = self.preset_service.load_preset(preset_name)
            if preset_env_vars is None:
                self._ui_log(f"加载预设失败: {preset_name}", "ERROR")
                return False
            
            # 汇总所有翻译器的必填+可选环境变量键，防止旧值残留
            all_env_keys = set()
            for cfg in self.config_service.get_translator_configs().values():
                all_env_keys.update(getattr(cfg, "required_env_vars", []) or [])
                all_env_keys.update(getattr(cfg, "optional_env_vars", []) or [])
            # 预设文件里出现的额外键也一并纳入（如 MAX_REQUESTS_PER_MINUTE）
            all_env_keys.update(preset_env_vars.keys())
            
            # 准备要更新的环境变量：未在预设内的键置空，避免继续使用旧值
            env_vars_to_update = {}
            for key in sorted(all_env_keys):
                env_vars_to_update[key] = preset_env_vars.get(key, "")
            
            # 写入 .env
            success = self.config_service.save_env_vars(env_vars_to_update)
            if not success:
                self._ui_log(f"应用预设失败: {preset_name}", "ERROR")
                return False

            # 成功日志（UI + 文件）
            self._ui_log(f"已应用预设: {preset_name}")
            self.logger.info(f"Preset applied: {preset_name}")
            
            # 清除翻译器缓存，确保下次翻译时重新创建实例（读取新的环境变量）
            try:
                from manga_translator.translators import translator_cache
                translator_cache.clear()
                self.logger.debug("翻译器缓存已清除")
            except Exception as e:
                self.logger.warning(f"清除翻译器缓存失败: {e}")
            
            return success
        except Exception as e:
            self.logger.error(f"加载预设失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._ui_log(f"加载预设失败: {e}", "ERROR")
            return False
    
    @pyqtSlot(str)
    def delete_preset(self, preset_name: str) -> bool:
        """删除预设"""
        try:
            success = self.preset_service.delete_preset(preset_name)
            if success:
                self._ui_log(f"预设已删除: {preset_name}")
            else:
                self._ui_log(f"删除预设失败: {preset_name}", "ERROR")
            return success
        except Exception as e:
            self.logger.error(f"删除预设失败: {e}")
            self._ui_log(f"删除预设失败: {e}", "ERROR")
            return False
    # endregion
    
    # region API测试
    async def test_api_connection_async(self, translator_key: str, api_key: str, api_base: str = None, model: str = None) -> tuple[bool, str]:
        """异步测试API连接（如果指定了模型，会测试该模型是否可用）"""
        try:
            if "openai" in translator_key.lower():
                # 尝试使用 curl_cffi 客户端绕过 TLS 指纹检测
                try:
                    from manga_translator.translators.common import AsyncOpenAICurlCffi
                    client = AsyncOpenAICurlCffi(
                        api_key=api_key,
                        base_url=api_base or "https://api.openai.com/v1",
                        impersonate="chrome110",
                        timeout=30.0
                    )
                except ImportError:
                    # 如果 curl_cffi 不可用，回退到标准客户端
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=api_base or "https://api.openai.com/v1"
                    )
                
                try:
                    # 如果指定了模型，测试该模型是否可用
                    if model and model.strip():
                        try:
                            # 尝试用该模型发送一个简单的测试请求（不传递 max_tokens 以兼容所有模型）
                            await client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": "test"}]
                            )
                            return True, f"连接成功，模型 {model} 可用"
                        except Exception as e:
                            # 如果模型测试失败，返回详细错误
                            return False, f"连接成功但模型 {model} 不可用: {str(e)}"
                    else:
                        # 没有指定模型，只测试连接
                        await client.models.list()
                        return True, "连接成功"
                finally:
                    await client.close()
            
            elif "gemini" in translator_key.lower():
                # 统一使用 AsyncGeminiCurlCffi（Google 认证方式 + curl_cffi TLS 指纹伪装）
                base_url = api_base.strip() if api_base and api_base.strip() else "https://generativelanguage.googleapis.com"

                try:
                    from manga_translator.translators.common import AsyncGeminiCurlCffi
                    client = AsyncGeminiCurlCffi(
                        api_key=api_key,
                        base_url=base_url,
                        impersonate="chrome110",
                        timeout=30.0
                    )

                    try:
                        # 如果指定了模型，测试该模型
                        if model and model.strip():
                            try:
                                await client.models.generate_content(
                                    model=model,
                                    contents="test"
                                )
                                return True, f"连接成功，模型 {model} 可用"
                            except Exception as e:
                                return False, f"连接成功但模型 {model} 不可用: {str(e)}"
                        else:
                            # 尝试列出模型
                            await client.models.list()
                            return True, "连接成功"
                    finally:
                        await client.close()

                except ImportError:
                    # 如果 curl_cffi 不可用，回退到标准 SDK
                    from google import genai
                    import asyncio
                    loop = asyncio.get_event_loop()

                    def sync_test():
                        # 新版 SDK 使用 genai.Client
                        if base_url != "https://generativelanguage.googleapis.com":
                            from google.genai import types
                            client = genai.Client(
                                api_key=api_key,
                                http_options=types.HttpOptions(base_url=base_url)
                            )
                        else:
                            client = genai.Client(api_key=api_key)

                        # 如果指定了模型，测试该模型
                        if model and model.strip():
                            client.models.generate_content(
                                model=model,
                                contents="test"
                            )
                            return True, f"连接成功，模型 {model} 可用"
                        else:
                            # 列出模型
                            list(client.models.list())
                            return True, "连接成功"

                    try:
                        return await loop.run_in_executor(None, sync_test)
                    except Exception as e:
                        return False, f"连接失败: {str(e)}"
            
            elif "sakura" in translator_key.lower():
                # Sakura使用OpenAI兼容API
                from openai import AsyncOpenAI
                if not api_base:
                    return False, "请先配置SAKURA_API_BASE"
                client = AsyncOpenAI(
                    api_key="sk-114514",  # Sakura使用固定密钥
                    base_url=api_base
                )
                
                try:
                    # 如果指定了模型，测试该模型
                    if model and model.strip():
                        try:
                            # 不传递 max_tokens 以兼容所有模型
                            await client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": "test"}]
                            )
                            return True, f"连接成功，模型 {model} 可用"
                        except Exception as e:
                            return False, f"连接成功但模型 {model} 不可用: {str(e)}"
                    else:
                        await client.models.list()
                        return True, "连接成功"
                finally:
                    await client.close()
            
            else:
                return False, "该翻译器不支持API测试"
                
        except Exception as e:
            return False, f"连接失败: {str(e)}"
    
    async def get_available_models_async(self, translator_key: str, api_key: str, api_base: str = None) -> tuple[bool, List[str], str]:
        """异步获取可用模型列表"""
        try:
            if "openai" in translator_key.lower():
                # 尝试使用 curl_cffi 客户端绕过 TLS 指纹检测
                try:
                    from manga_translator.translators.common import AsyncOpenAICurlCffi
                    client = AsyncOpenAICurlCffi(
                        api_key=api_key,
                        base_url=api_base or "https://api.openai.com/v1",
                        impersonate="chrome110",
                        timeout=60.0
                    )
                except ImportError:
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=api_base or "https://api.openai.com/v1",
                        timeout=60.0,
                    )
                
                try:
                    models_response = await client.models.list()
                    
                    # 获取所有模型ID，不过滤
                    model_ids = [m.id for m in models_response.data]
                    model_ids.sort(reverse=True)  # 新模型在前
                    
                    return True, model_ids, "获取成功"
                finally:
                    await client.close()
            
            elif "gemini" in translator_key.lower():
                # Gemini API - 使用 curl_cffi 绕过 TLS 指纹检测，使用 Google Gemini 认证格式
                try:
                    from manga_translator.translators.common import AsyncGeminiCurlCffi

                    # 确定 base_url
                    base_url = api_base.strip() if api_base and api_base.strip() else "https://generativelanguage.googleapis.com"

                    client = AsyncGeminiCurlCffi(
                        api_key=api_key,
                        base_url=base_url,
                        impersonate="chrome110",
                        timeout=60.0
                    )
                    try:
                        models_response = await client.models.list()
                        model_ids = [m.id for m in models_response]
                        return True, model_ids, "获取成功"
                    finally:
                        await client.close()
                except ImportError:
                    # 如果 curl_cffi 不可用，回退到标准客户端
                    from google import genai
                    from google.genai import types
                    import asyncio
                    loop = asyncio.get_event_loop()

                    # 检查是否是自定义API
                    is_custom_api = (
                        api_base
                        and api_base.strip()
                        and api_base.strip() not in ["https://generativelanguage.googleapis.com", "https://generativelanguage.googleapis.com/"]
                    )

                    if is_custom_api:
                        # 自定义 API 使用 http_options
                        def sync_get_models():
                            client = genai.Client(
                                api_key=api_key,
                                http_options=types.HttpOptions(base_url=api_base.strip())
                            )
                            models = list(client.models.list())
                            model_names = [m.name.replace("models/", "") for m in models]
                            return True, model_names, "获取成功"
                    else:
                        def sync_get_models():
                            client = genai.Client(api_key=api_key)
                            models = list(client.models.list())
                            model_names = [m.name.replace("models/", "") for m in models]
                            return True, model_names, "获取成功"

                    return await loop.run_in_executor(None, sync_get_models)
            
            elif "sakura" in translator_key.lower():
                # Sakura使用OpenAI兼容API
                from openai import AsyncOpenAI
                if not api_base:
                    return False, [], "请先配置SAKURA_API_BASE"
                client = AsyncOpenAI(
                    api_key="sk-114514",
                    base_url=api_base
                )
                try:
                    models_response = await client.models.list()
                    model_ids = [m.id for m in models_response.data]
                    return True, model_ids, "获取成功"
                finally:
                    await client.close()
            
            else:
                return False, [], "该翻译器不支持获取模型列表"
                
        except Exception as e:
            return False, [], f"获取失败: {str(e)}"
    # endregion

    # region 配置管理
    def load_config_file(self, config_path: str) -> bool:
        try:
            success = self.config_service.load_config_file(config_path)
            if success:
                config = self.config_service.get_config()
                self.state_manager.set_current_config(config)
                self.state_manager.set_state(AppStateKey.CONFIG_PATH, config_path)
                self.logger.info(self._t("log_config_loaded_successfully", path=config_path))
                self.config_loaded.emit(config.dict())
                if config.app.last_output_path:
                    self.output_path_updated.emit(config.app.last_output_path)
                return True
            else:
                self.logger.error(self._t("log_config_load_failed", path=config_path))
                return False
        except Exception as e:
            self.logger.error(self._t("log_config_load_exception", error=e))
            return False
    
    def save_config_file(self, config_path: str = None) -> bool:
        try:
            success = self.config_service.save_config_file(config_path)
            if success:
                self.logger.info(self._t("log_config_saved_successfully"))
                return True
            return False
        except Exception as e:
            self.logger.error(self._t("log_config_save_exception", error=e))
            return False
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        try:
            self.config_service.update_config(config_updates)
            updated_config = self.config_service.get_config()
            self.state_manager.set_current_config(updated_config)
            self.logger.info(self._t("log_config_updated_successfully"))
            return True
        except Exception as e:
            self.logger.error(self._t("log_config_update_exception", error=e))
            return False

    def update_single_config(self, full_key: str, value: Any):
        self.logger.debug(f"update_single_config: '{full_key}' = '{value}'")
        try:
            config_obj = self.config_service.get_config()
            keys = full_key.split('.')
            parent_obj = config_obj
            for key in keys[:-1]:
                parent_obj = getattr(parent_obj, key)
            setattr(parent_obj, keys[-1], value)
            
            self.config_service.set_config(config_obj)
            self.config_service.save_config_file()
            self.logger.debug(self._t("log_config_saved", config_key=full_key, value=value))

            # 当翻译器设置被更改时，直接更新翻译服务的内部状态
            if full_key == 'translator.translator':
                self.logger.debug(self._t("log_translator_switched", value=value))
                self.translation_service.set_translator(value)
            
            # 当目标语言被更改时，更新翻译服务的目标语言
            if full_key == 'translator.target_lang':
                self.logger.debug(f"Target language switched to: {value}")
                self.translation_service.set_target_language(value)

            # 当渲染设置被更改时，通知编辑器刷新
            if full_key.startswith('render.'):
                self.logger.debug(self._t("log_render_setting_changed", config_key=full_key))
                self.render_setting_changed.emit()

        except Exception as e:
            self.logger.error(f"Error saving single config change for {full_key}: {e}")
    # endregion

    # region UI数据提供
    def get_display_mapping(self, key: str) -> Optional[Dict[str, str]]:
        # 每次都动态生成翻译映射，确保语言切换时能正确更新
        display_name_maps = {
            "alignment": {
                "auto": self._t("alignment_auto"),
                "left": self._t("alignment_left"),
                "center": self._t("alignment_center"),
                "right": self._t("alignment_right")
            },
            "direction": {
                "auto": self._t("direction_auto"),
                "h": self._t("direction_horizontal"),
                "v": self._t("direction_vertical")
            },
            "upscaler": {
                "waifu2x": "Waifu2x",
                "esrgan": "ESRGAN",
                "4xultrasharp": "4x UltraSharp",
                "realcugan": "Real-CUGAN",
                "mangajanai": "MangaJaNai"
            },
            "layout_mode": {
                'smart_scaling': self._t("layout_mode_smart_scaling"),
                'strict': self._t("layout_mode_strict"),
                'balloon_fill': self._t("layout_mode_balloon_fill")
            },
                "realcugan_model": {
                    "2x-conservative": self._t("realcugan_2x_conservative"),
                    "2x-conservative-pro": self._t("realcugan_2x_conservative_pro"),
                    "2x-no-denoise": self._t("realcugan_2x_no_denoise"),
                    "2x-denoise1x": self._t("realcugan_2x_denoise1x"),
                    "2x-denoise2x": self._t("realcugan_2x_denoise2x"),
                    "2x-denoise3x": self._t("realcugan_2x_denoise3x"),
                    "2x-denoise3x-pro": self._t("realcugan_2x_denoise3x_pro"),
                    "3x-conservative": self._t("realcugan_3x_conservative"),
                    "3x-conservative-pro": self._t("realcugan_3x_conservative_pro"),
                    "3x-no-denoise": self._t("realcugan_3x_no_denoise"),
                    "3x-no-denoise-pro": self._t("realcugan_3x_no_denoise_pro"),
                    "3x-denoise3x": self._t("realcugan_3x_denoise3x"),
                    "3x-denoise3x-pro": self._t("realcugan_3x_denoise3x_pro"),
                    "4x-conservative": self._t("realcugan_4x_conservative"),
                    "4x-no-denoise": self._t("realcugan_4x_no_denoise"),
                    "4x-denoise3x": self._t("realcugan_4x_denoise3x"),
                },
                "translator": {
                    "openai": "OpenAI",
                    "openai_hq": self._t("translator_openai_hq"),
                    "gemini": "Google Gemini",
                    "gemini_hq": self._t("translator_gemini_hq"),
                    "sakura": "Sakura",
                    "none": self._t("translator_none"),
                    "original": self._t("translator_original"),
                },
                "target_lang": self.translation_service.get_target_languages(),
                "ocr_vl_language_hint": {
                    "auto": self._t("ocr_lang_auto"),
                    "multilingual": self._t("ocr_lang_multilingual"),
                    "Arabic": self._t("ocr_lang_arabic"),
                    "Simplified Chinese": self._t("ocr_lang_simplified_chinese"),
                    "Traditional Chinese": self._t("ocr_lang_traditional_chinese"),
                    "English": self._t("ocr_lang_english"),
                    "Japanese": self._t("ocr_lang_japanese"),
                    "Korean": self._t("ocr_lang_korean"),
                    "Spanish": self._t("ocr_lang_spanish"),
                    "French": self._t("ocr_lang_french"),
                    "German": self._t("ocr_lang_german"),
                    "Russian": self._t("ocr_lang_russian"),
                    "Portuguese": self._t("ocr_lang_portuguese"),
                    "Italian": self._t("ocr_lang_italian"),
                    "Thai": self._t("ocr_lang_thai"),
                    "Vietnamese": self._t("ocr_lang_vietnamese"),
                    "Indonesian": self._t("ocr_lang_indonesian"),
                    "Turkish": self._t("ocr_lang_turkish"),
                    "Polish": self._t("ocr_lang_polish"),
                    "Ukrainian": self._t("ocr_lang_ukrainian"),
                },
                "labels": {
                    "enable_watermark_filter": self._t("label_filter_text_enabled"),
                    "kernel_size": self._t("label_kernel_size"),
                    "mask_dilation_offset": self._t("label_mask_dilation_offset"),
                    "translator": self._t("label_translator"),
                    "target_lang": self._t("label_target_lang"),
                    "no_text_lang_skip": self._t("label_no_text_lang_skip"),
                    "gpt_config": self._t("label_gpt_config"),
                    "high_quality_prompt_path": self._t("label_high_quality_prompt_path"),
                    "extract_glossary": self._t("label_extract_glossary"),
                    "use_custom_api_params": self._t("label_use_custom_api_params"),
                    "use_stream": self._t("label_use_stream"),
                    "use_mocr_merge": self._t("label_use_mocr_merge"),
                    "ocr": self._t("label_ocr"),
                    "use_hybrid_ocr": self._t("label_use_hybrid_ocr"),
                    "secondary_ocr": self._t("label_secondary_ocr"),
                    "min_text_length": self._t("label_min_text_length"),
                    "ignore_bubble": self._t("label_ignore_bubble"),
                    "use_model_bubble_filter": self._t("label_use_model_bubble_filter"),
                    "model_bubble_overlap_threshold": self._t("label_model_bubble_overlap_threshold"),
                    "use_model_bubble_repair_intersection": self._t("label_use_model_bubble_repair_intersection"),
                    "limit_mask_dilation_to_bubble_mask": self._t("label_limit_mask_dilation_to_bubble_mask"),
                    "prob": self._t("label_prob"),
                    "merge_gamma": self._t("label_merge_gamma"),
                    "merge_sigma": self._t("label_merge_sigma"),
                    "merge_edge_ratio_threshold": self._t("label_merge_edge_ratio_threshold"),
                    "merge_special_require_full_wrap": self._t("label_merge_special_require_full_wrap"),
                    "yolo_restrict_merge": self._t("label_yolo_restrict_merge"),
                    "ocr_vl_language_hint": self._t("label_ocr_vl_language_hint"),
                    "ocr_vl_custom_prompt": self._t("label_ocr_vl_custom_prompt"),
                    "detector": self._t("label_detector"),
                    "detection_size": self._t("label_detection_size"),
                    "text_threshold": self._t("label_text_threshold"),
                    "det_rotate": self._t("label_det_rotate"),
                    "det_auto_rotate": self._t("label_det_auto_rotate"),
                    "det_invert": self._t("label_det_invert"),
                    "det_gamma_correct": self._t("label_det_gamma_correct"),
                    "use_yolo_obb": self._t("label_use_yolo_obb"),
                    "yolo_obb_conf": self._t("label_yolo_obb_conf"),
                    "yolo_obb_iou": self._t("label_yolo_obb_iou"),
                    "yolo_obb_overlap_threshold": self._t("label_yolo_obb_overlap_threshold"),
                    "box_threshold": self._t("label_box_threshold"),
                    "unclip_ratio": self._t("label_unclip_ratio"),
                    "min_box_area_ratio": self._t("label_min_box_area_ratio"),
                    "inpainter": self._t("label_inpainter"),
                    "inpainting_size": self._t("label_inpainting_size"),
                    "inpainting_precision": self._t("label_inpainting_precision"),
                    "inpainting_split_ratio": self._t("label_inpainting_split_ratio"),
                    "force_use_torch_inpainting": self._t("label_force_use_torch_inpainting"),
                    "renderer": self._t("label_renderer"),
                    "alignment": self._t("label_alignment"),
                    "disable_font_border": self._t("label_disable_font_border"),
                    "disable_auto_wrap": self._t("label_disable_auto_wrap"),
                    "font_size_offset": self._t("label_font_size_offset"),
                    "font_size_minimum": self._t("label_font_size_minimum"),
                    "max_font_size": self._t("label_max_font_size"),
                    "font_scale_ratio": self._t("label_font_scale_ratio"),
                    "stroke_width": self._t("label_stroke_width"),
                    "center_text_in_bubble": self._t("label_center_text_in_bubble"),
                    "optimize_line_breaks": self._t("label_optimize_line_breaks"),
                    "check_br_and_retry": self._t("label_check_br_and_retry"),
                    "strict_smart_scaling": self._t("label_strict_smart_scaling"),
                    "direction": self._t("label_direction"),
                    "uppercase": self._t("label_uppercase"),
                    "lowercase": self._t("label_lowercase"),
                    "font_path": self._t("label_font_path"),
                    "no_hyphenation": self._t("label_no_hyphenation"),
                    "font_color": self._t("label_font_color"),
                    "auto_rotate_symbols": self._t("label_auto_rotate_symbols"),
                    "rtl": self._t("label_rtl"),
                    "layout_mode": self._t("label_layout_mode"),
                    "upscaler": self._t("label_upscaler"),
                    "upscale_ratio": self._t("label_upscale_ratio"),
                    "realcugan_model": self._t("label_realcugan_model"),
                    "tile_size": self._t("label_tile_size"),
                    "revert_upscaling": self._t("label_revert_upscaling"),
                    "colorization_size": self._t("label_colorization_size"),
                    "denoise_sigma": self._t("label_denoise_sigma"),
                    "colorizer": self._t("label_colorizer"),
                    "verbose": self._t("label_verbose"),
                    "attempts": self._t("label_attempts"),
                    "ignore_errors": self._t("label_ignore_errors"),
                    "use_gpu": self._t("label_use_gpu"),
                    "context_size": self._t("label_context_size"),
                    "format": self._t("label_format"),
                    "overwrite": self._t("label_overwrite"),
                    "skip_no_text": self._t("label_skip_no_text"),
                    "save_text": self._t("label_save_text"),
                    "load_text": self._t("label_load_text"),
                    "template": self._t("label_template"),
                    "save_quality": self._t("label_save_quality"),
                    "batch_size": self._t("label_batch_size"),
                    "batch_concurrent": self._t("label_batch_concurrent"),
                    "translation_concurrency": self._t("label_translation_concurrency"),
                    "inpaint_concurrency": self._t("label_inpaint_concurrency"),
                    "generate_and_export": self._t("label_generate_and_export"),
                    "last_output_path": self._t("label_last_output_path"),
                    "save_to_source_dir": self._t("label_save_to_source_dir"),
                    "line_spacing": self._t("label_line_spacing"),
                    "font_size": self._t("label_font_size"),
                    "OPENAI_API_KEY": self._t("label_OPENAI_API_KEY"),
                    "OPENAI_MODEL": self._t("label_OPENAI_MODEL"),
                    "OPENAI_API_BASE": self._t("label_OPENAI_API_BASE"),
                    "OPENAI_GLOSSARY_PATH": self._t("label_OPENAI_GLOSSARY_PATH"),
                    "GEMINI_API_KEY": self._t("label_GEMINI_API_KEY"),
                    "GEMINI_MODEL": self._t("label_GEMINI_MODEL"),
                    "GEMINI_API_BASE": self._t("label_GEMINI_API_BASE"),
                    "SAKURA_API_BASE": self._t("label_SAKURA_API_BASE"),
                    "SAKURA_DICT_PATH": self._t("label_SAKURA_DICT_PATH"),
                    "SAKURA_VERSION": self._t("label_SAKURA_VERSION"),
                    "MAX_REQUESTS_PER_MINUTE": self._t("label_MAX_REQUESTS_PER_MINUTE"),
                    "CUSTOM_OPENAI_API_BASE": self._t("label_CUSTOM_OPENAI_API_BASE"),
                    "CUSTOM_OPENAI_MODEL": self._t("label_CUSTOM_OPENAI_MODEL"),
                    "CUSTOM_OPENAI_API_KEY": self._t("label_CUSTOM_OPENAI_API_KEY"),
                    "CUSTOM_OPENAI_MODEL_CONF": self._t("label_CUSTOM_OPENAI_MODEL_CONF")
                }
            }
        return display_name_maps.get(key)

    def get_options_for_key(self, key: str) -> Optional[List[str]]:
        options_map = {
            "format": [self._t("format_not_specified")] + list(OUTPUT_FORMATS.keys()),
            "renderer": [member.value for member in Renderer],
            "alignment": [member.value for member in Alignment],
            "direction": [member.value for member in Direction],
            "upscaler": [member.value for member in Upscaler],
            "upscale_ratio": [self._t("upscale_ratio_not_use"), "2", "3", "4"],
            "realcugan_model": [
                "2x-conservative",
                "2x-conservative-pro",
                "2x-no-denoise",
                "2x-denoise1x",
                "2x-denoise2x",
                "2x-denoise3x",
                "2x-denoise3x-pro",
                "3x-conservative",
                "3x-conservative-pro",
                "3x-no-denoise",
                "3x-no-denoise-pro",
                "3x-denoise3x",
                "3x-denoise3x-pro",
                "4x-conservative",
                "4x-no-denoise",
                "4x-denoise3x",
            ],
            "translator": [member.value for member in Translator],
            "detector": [member.value for member in Detector],
            "colorizer": [member.value for member in Colorizer],
            "inpainter": [member.value for member in Inpainter],
            "inpainting_precision": [member.value for member in InpaintPrecision],
            "ocr": [member.value for member in Ocr],
            "secondary_ocr": [member.value for member in Ocr]
        }
        return options_map.get(key)
    @pyqtSlot()
    def export_config(self):
        """导出配置（排除敏感信息）"""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        
        try:
            # 选择保存位置
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                self._t("Export Config"),
                "manga_translator_config.json",
                "JSON Files (*.json)"
            )
            
            if not file_path:
                return
            
            # 获取当前配置
            config = self.config_service.get_config()
            config_dict = config.dict()
            
            # 排除敏感信息和临时状态
            # 1. 排除 app 配置（包含路径等临时信息）
            if 'app' in config_dict:
                del config_dict['app']
            
            # 2. 排除 CLI 中的临时状态
            if 'cli' in config_dict:
                # 保留 CLI 配置，但排除某些临时字段
                cli_exclude = ['verbose']  # 可以根据需要添加更多
                for key in cli_exclude:
                    if key in config_dict['cli']:
                        del config_dict['cli'][key]
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(self._t("log_config_exported", path=file_path))
            QMessageBox.information(
                None,
                self._t("Export Success"),
                self._t("Config exported successfully to:\n{path}\n\nNote: Sensitive information like API keys are not included.", path=file_path)
            )
            
        except Exception as e:
            self.logger.error(self._t("log_config_export_failed", error=e))
            QMessageBox.critical(
                None,
                self._t("Export Failed"),
                self._t("Error occurred while exporting config:\n{error}", error=str(e))
            )
    
    @pyqtSlot()
    def import_config(self):
        """导入配置（保留现有的敏感信息）"""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        
        try:
            # 选择要导入的文件
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                self._t("Import Config"),
                "",
                "JSON Files (*.json)"
            )
            
            if not file_path:
                return
            
            # 读取导入的配置
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # 获取当前配置
            current_config = self.config_service.get_config()
            current_dict = current_config.dict()
            
            # 保留当前的 app 配置（路径等临时信息）
            preserved_app = current_dict.get('app', {})
            
            # 深度合并配置
            def deep_update(target, source):
                for key, value in source.items():
                    if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                        deep_update(target[key], value)
                    else:
                        target[key] = value
            
            # 合并导入的配置到当前配置
            deep_update(current_dict, imported_config)
            
            # 恢复 app 配置
            current_dict['app'] = preserved_app
            
            # 更新配置
            from core.config_models import AppSettings
            new_config = AppSettings.parse_obj(current_dict)
            self.config_service.set_config(new_config)
            self.config_service.save_config_file()
            
            # 通知UI更新 - 使用转换后的配置字典
            config_dict_for_ui = self.config_service._convert_config_for_ui(new_config.dict())
            self.config_loaded.emit(config_dict_for_ui)
            
            self.logger.info(self._t("log_config_imported", path=file_path))
            QMessageBox.information(
                None,
                self._t("Import Success"),
                self._t("Config imported successfully!\n\nSource: {path}\n\nNote: Your API keys and sensitive information have been preserved.", path=file_path)
            )
            
        except Exception as e:
            self.logger.error(self._t("log_config_import_failed", error=e))
            QMessageBox.critical(
                None,
                self._t("Import Failed"),
                self._t("Error occurred while importing config:\n{error}\n\nPlease ensure the file format is correct.", error=str(e))
            )
    # endregion

    # region 文件管理
    def add_files(self, file_paths: List[str]):
        """
        Adds files/folders to the list for processing.
        只添加有效的图片文件或文件夹，过滤掉非图片文件。
        """
        new_paths = []
        skipped_files = []
        for path in file_paths:
            norm_path = os.path.normpath(path)
            if norm_path not in self.source_files:
                # 文件夹直接添加
                if os.path.isdir(norm_path):
                    new_paths.append(norm_path)
                # 文件需要验证是否为有效的图片或压缩包
                elif os.path.isfile(norm_path):
                    if self.file_service.validate_image_file(norm_path):
                        new_paths.append(norm_path)
                    else:
                        skipped_files.append(norm_path)

        if skipped_files:
            self._ui_log(f"⚠️ 跳过 {len(skipped_files)} 个非图片文件", "WARNING")
            for f in skipped_files[:3]:  # 只显示前3个
                self._ui_log(f"   - {os.path.basename(f)}")
            if len(skipped_files) > 3:
                self._ui_log(f"   ... 还有 {len(skipped_files) - 3} 个")

        if new_paths:
            self.source_files.extend(new_paths)
            self.logger.info(f"Added {len(new_paths)} files/folders to the list.")
            self.files_added.emit(new_paths)

    def get_last_open_dir(self) -> str:
        path = self.config_service.get_config().app.last_open_dir
        self.logger.info(f"Retrieved last open directory: {path}")
        return path

    def set_last_open_dir(self, path: str):
        self.logger.info(f"Saving last open directory: {path}")
        self.update_single_config('app.last_open_dir', path)

    def add_folder(self):
        """Opens a dialog to select folders (supports multiple selection) and adds their paths to the list."""
        last_dir = self.get_last_open_dir()

        # 使用自定义的现代化文件夹选择器
        from widgets.folder_dialog import select_folders

        folders = select_folders(
            parent=None,
            start_dir=last_dir,
            multi_select=True,
            config_service=self.config_service
        )

        if folders:
            self.set_last_open_dir(folders[0])  # 保存第一个文件夹的路径
            self.add_files(folders)
    
    def add_folders(self):
        """Alias for add_folder for backward compatibility."""
        self.add_folder()

    def remove_file(self, file_path: str):
        try:
            norm_file_path = os.path.normpath(file_path)
            
            # 尝试在 source_files 中找到匹配的路径（不区分大小写，处理路径分隔符）
            matched_path = None
            for source_path in self.source_files:
                if os.path.normpath(source_path).lower() == norm_file_path.lower():
                    matched_path = source_path
                    break
            
            # 情况1：直接在 source_files 中（文件夹或单独添加的文件）
            if matched_path:
                self.source_files.remove(matched_path)
                # 如果是文件，清理 file_to_folder_map
                if matched_path in self.file_to_folder_map:
                    del self.file_to_folder_map[matched_path]
                
                # 如果是文件夹，清理排除列表中该文件夹下的所有子文件夹
                if os.path.isdir(matched_path):
                    excluded_to_remove = set()
                    for excluded_folder in self.excluded_subfolders:
                        try:
                            # 检查 excluded_folder 是否在被删除的文件夹内
                            common = os.path.commonpath([matched_path, excluded_folder])
                            if common == os.path.normpath(matched_path):
                                excluded_to_remove.add(excluded_folder)
                        except ValueError:
                            continue
                    self.excluded_subfolders -= excluded_to_remove
                
                self.file_removed.emit(file_path)
                return
            
            # 情况2：文件夹路径（可能是顶层文件夹或子文件夹）
            if os.path.isdir(norm_file_path):
                # 检查是否是某个顶层文件夹的子文件夹
                parent_folder = None
                for folder in self.source_files:
                    if os.path.isdir(folder):
                        try:
                            # 检查 norm_file_path 是否是 folder 的子文件夹
                            common = os.path.commonpath([folder, norm_file_path])
                            if common == os.path.normpath(folder) and norm_file_path != os.path.normpath(folder):
                                parent_folder = folder
                                break
                        except ValueError:
                            continue
                
                if parent_folder:
                    # 这是子文件夹，添加到排除列表
                    self.excluded_subfolders.add(norm_file_path)
                    # 发射删除信号让 FileListView 处理
                    # FileListView 会自动更新树形结构和文件数量
                    self.file_removed.emit(file_path)
                    return
                
                # 不是子文件夹，可能是通过单独添加文件自动分组的文件夹
                # 删除该文件夹下的所有文件
                files_to_remove = []
                for source_file in self.source_files:
                    if os.path.isfile(source_file):
                        try:
                            # 检查文件是否在这个文件夹内
                            common = os.path.commonpath([norm_file_path, source_file])
                            if common == norm_file_path:
                                files_to_remove.append(source_file)
                        except ValueError:
                            # 不同驱动器，跳过
                            continue
                
                # 移除所有找到的文件
                for f in files_to_remove:
                    self.source_files.remove(f)
                    # 同时清理 file_to_folder_map
                    if f in self.file_to_folder_map:
                        del self.file_to_folder_map[f]
                
                if files_to_remove:
                    self.file_removed.emit(file_path)
                    return
            
            # 情况3：文件夹内的单个文件（只处理文件，不处理文件夹）
            if os.path.isfile(norm_file_path):
                # 检查这个文件是否来自某个文件夹
                parent_folder = None
                for folder in self.source_files:
                    if os.path.isdir(folder):
                        # 检查文件是否在这个文件夹内
                        try:
                            common = os.path.commonpath([folder, norm_file_path])
                            # 确保文件在文件夹内，而不是文件夹本身
                            if common == os.path.normpath(folder) and norm_file_path != os.path.normpath(folder):
                                parent_folder = folder
                                break
                        except ValueError:
                            # 不同驱动器，跳过
                            continue
                
                if parent_folder:
                    # 这是文件夹内的文件，需要将其添加到排除列表
                    # 由于当前架构不支持排除单个文件，我们需要：
                    # 1. 移除整个文件夹
                    # 2. 添加文件夹内的其他文件
                    
                    # 获取文件夹内的所有图片文件
                    folder_files = self.file_service.get_image_files_from_folder(parent_folder, recursive=True)
                    
                    # 移除要删除的文件
                    remaining_files = [f for f in folder_files if os.path.normpath(f) != norm_file_path]
                    
                    # 从 source_files 中移除文件夹
                    self.source_files.remove(parent_folder)
                    
                    # 如果还有剩余文件，将它们作为单独的文件添加回去
                    if remaining_files:
                        self.source_files.extend(remaining_files)
                        # 更新 file_to_folder_map：这些文件现在仍然属于原文件夹
                        # 保持文件夹映射关系，以便输出路径计算正确
                        for f in remaining_files:
                            self.file_to_folder_map[f] = parent_folder
                    
                    self.file_removed.emit(file_path)
                    return
            
            # 如果到这里还没有处理，说明路径不存在
            self.logger.warning(f"Path not found in list for removal: {file_path}")
        except Exception as e:
            self._ui_log(f"移除路径时发生异常: {e}", "ERROR")

    def clear_file_list(self):
        if not self.source_files:
            return
        # TODO: Add confirmation dialog
        self.source_files.clear()
        self.file_to_folder_map.clear()  # 清空文件夹映射
        self.excluded_subfolders.clear()  # 清空排除列表
        self.files_cleared.emit()
        self.logger.info("File list cleared by user.")
    # endregion

    # region 核心任务逻辑
    def get_folder_tree_structure(self) -> dict:
        """
        获取完整的文件夹树结构
        返回: {
            'files': [所有文件列表],
            'tree': {
                'folder_path': {
                    'files': [该文件夹直接包含的文件],
                    'subfolders': [子文件夹路径列表]
                }
            }
        }
        """
        tree = {}
        all_files = []
        
        # 处理每个顶层文件夹
        for source_path in self.source_files:
            if os.path.isdir(source_path):
                norm_folder = os.path.normpath(source_path)
                # 递归构建该文件夹的树结构
                folder_files = self._build_folder_tree(norm_folder, tree)
                all_files.extend(folder_files)
            elif os.path.isfile(source_path):
                # 单独添加的文件
                all_files.append(source_path)
        
        return {
            'files': all_files,
            'tree': tree
        }
    
    def _build_folder_tree(self, folder_path: str, tree: dict) -> List[str]:
        """
        递归构建文件夹树结构
        返回该文件夹及其子文件夹中的所有文件列表
        """
        # 检查是否被排除
        if folder_path in self.excluded_subfolders:
            return []
        
        norm_folder = os.path.normpath(folder_path)
        
        # 初始化该文件夹的树节点
        if norm_folder not in tree:
            tree[norm_folder] = {
                'files': [],
                'subfolders': []
            }
        
        all_files = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif'}
        
        try:
            items = os.listdir(folder_path)
            subdirs = []
            files = []
            
            for item in items:
                if item == 'manga_translator_work':
                    continue
                
                item_path = os.path.join(folder_path, item)
                norm_item_path = os.path.normpath(item_path)
                
                if os.path.isdir(item_path):
                    # 检查是否被排除
                    if norm_item_path not in self.excluded_subfolders:
                        subdirs.append(norm_item_path)
                        tree[norm_folder]['subfolders'].append(norm_item_path)
                elif os.path.splitext(item)[1].lower() in image_extensions:
                    files.append(norm_item_path)
            
            # 排序
            subdirs.sort(key=self.file_service._natural_sort_key)
            files.sort(key=self.file_service._natural_sort_key)
            
            # 添加该文件夹直接包含的文件
            tree[norm_folder]['files'] = files
            all_files.extend(files)
            
            # 递归处理子文件夹
            for subdir in subdirs:
                subdir_files = self._build_folder_tree(subdir, tree)
                all_files.extend(subdir_files)
        
        except Exception as e:
            self.logger.error(f"Error building tree for folder {folder_path}: {e}")
        
        return all_files
    
    def _resolve_input_files(self) -> List[str]:
        """
        Expands folders in self.source_files into a list of image files.
        同时记录文件和文件夹的映射关系。
        按文件夹分组排序：先对文件夹进行排序，然后对每个文件夹内的图片排序。
        支持 PDF、EPUB、CBZ 等压缩包格式，自动解压提取图片。
        """
        resolved_files = []
        # 保存旧的映射，用于处理删除文件后的情况
        old_map = self.file_to_folder_map.copy()
        self.file_to_folder_map.clear()
        
        # 记录压缩包到临时目录的映射，用于输出时保持结构
        self.archive_to_temp_map = getattr(self, 'archive_to_temp_map', {})

        # 分离文件和文件夹
        folders = []
        individual_files = []
        archive_files = []
        
        for path in self.source_files:
            if os.path.isdir(path):
                folders.append(path)
            elif os.path.isfile(path):
                if self.file_service.is_archive_file(path):
                    archive_files.append(path)
                elif self.file_service.validate_image_file(path):
                    individual_files.append(path)
        
        # 处理压缩包文件
        if archive_files:
            from desktop_qt_ui.utils.archive_extractor import extract_images_from_archive
            for archive_path in archive_files:
                try:
                    self._ui_log(f"正在解压: {os.path.basename(archive_path)}")
                    images, temp_dir = extract_images_from_archive(archive_path)
                    if images:
                        self.archive_to_temp_map[archive_path] = temp_dir
                        # 将解压出的图片添加到处理列表
                        for img_path in images:
                            resolved_files.append(img_path)
                            # 记录这些文件来自这个压缩包（使用压缩包路径作为"文件夹"）
                            self.file_to_folder_map[img_path] = archive_path
                        self._ui_log(f"从 {os.path.basename(archive_path)} 提取了 {len(images)} 张图片")
                    else:
                        self._ui_log(f"警告: {os.path.basename(archive_path)} 中没有找到图片", "WARNING")
                except Exception as e:
                    self._ui_log(f"解压 {os.path.basename(archive_path)} 失败: {e}", "ERROR")
        
        # 清理排除列表：移除不再属于任何 source_files 文件夹的排除项
        if self.excluded_subfolders:
            excluded_to_remove = set()
            for excluded_folder in self.excluded_subfolders:
                # 检查这个排除的文件夹是否还在某个 source_files 的文件夹内
                is_valid = False
                for folder in folders:
                    try:
                        common = os.path.commonpath([folder, excluded_folder])
                        if common == os.path.normpath(folder):
                            is_valid = True
                            break
                    except ValueError:
                        continue
                if not is_valid:
                    excluded_to_remove.add(excluded_folder)
            self.excluded_subfolders -= excluded_to_remove
        
        # 对文件夹进行自然排序
        folders.sort(key=self.file_service._natural_sort_key)
        
        # 按文件夹分组处理
        for folder in folders:
            # 获取文件夹中的所有图片（递归查找所有子文件夹，已经使用自然排序）
            folder_files = self.file_service.get_image_files_from_folder(folder, recursive=True)
            
            # 过滤掉被排除的子文件夹中的文件
            if self.excluded_subfolders:
                filtered_files = []
                for file_path in folder_files:
                    # 检查文件是否在被排除的子文件夹中
                    is_excluded = False
                    for excluded_folder in self.excluded_subfolders:
                        try:
                            common = os.path.commonpath([excluded_folder, file_path])
                            if common == excluded_folder:
                                is_excluded = True
                                break
                        except ValueError:
                            continue
                    if not is_excluded:
                        filtered_files.append(file_path)
                folder_files = filtered_files
            
            resolved_files.extend(folder_files)
            # 记录这些文件来自这个文件夹
            for file_path in folder_files:
                self.file_to_folder_map[file_path] = folder
        
        # 处理单独添加的文件（使用自然排序）
        # 获取当前输出目录，用于检测文件是否已在输出目录中
        output_folder = self.config_service.get_config().app.last_output_path
        output_folder_norm = os.path.normpath(output_folder) if output_folder else None
        
        individual_files.sort(key=self.file_service._natural_sort_key)
        for file_path in individual_files:
            resolved_files.append(file_path)
            # 检查是否有旧的文件夹映射（删除文件后剩余的文件）
            if file_path in old_map and old_map[file_path] is not None:
                # 保留原来的文件夹映射
                self.file_to_folder_map[file_path] = old_map[file_path]
            else:
                # 检查文件是否已经在输出目录中（重试场景）
                # 如果文件已在输出目录下，直接覆盖原位置，不重新创建目录结构
                file_path_norm = os.path.normpath(file_path)
                is_in_output_folder = False
                if output_folder_norm:
                    try:
                        common = os.path.commonpath([output_folder_norm, file_path_norm])
                        if common == output_folder_norm:
                            is_in_output_folder = True
                    except ValueError:
                        pass  # 不同驱动器
                
                if is_in_output_folder:
                    # 文件已在输出目录中（重试场景）
                    # 检查父目录是否是 Chapter 目录
                    parent_dir = os.path.dirname(file_path)
                    parent_name = os.path.basename(parent_dir) if parent_dir else ''
                    if 'chapter' in parent_name.lower():
                        # 父目录是 Chapter，使用输出目录作为映射
                        # 这样输出路径只会是: 输出目录/Chapter XX/文件名
                        self.file_to_folder_map[file_path] = output_folder_norm
                    else:
                        # 父目录不是 Chapter，直接放到输出根目录
                        self.file_to_folder_map[file_path] = None
                else:
                    # 尝试从文件路径推断文件夹映射（主要用于失败记录导入的文件）
                    # 只有当父目录名包含 "Chapter" 时才保留目录结构
                    parent_dir = os.path.dirname(file_path)
                    parent_name = os.path.basename(parent_dir) if parent_dir else ''
                    if parent_dir and 'chapter' in parent_name.lower():
                        # 父目录是 Chapter 目录，使用父目录作为映射
                        # 这样输出路径为: output_folder/Chapter XX/文件名
                        self.file_to_folder_map[file_path] = parent_dir
                    else:
                        # 父目录不是 Chapter，直接放到输出根目录
                        self.file_to_folder_map[file_path] = None

        return list(dict.fromkeys(resolved_files)) # Return unique files

    def start_backend_task(self):
        """
        Resolves input paths and uses a 'Worker-to-Thread' model to start the translation task.
        """
        # 通过调用配置服务的 reload_config 方法，强制全面重新加载所有配置
        try:
            self._ui_log("即将开始后台任务，强制重新加载所有配置...")
            self.config_service.reload_config()
            self._ui_log("配置已刷新，继续执行任务。")
        except Exception as e:
            self._ui_log(f"重新加载配置时发生严重错误: {e}", "ERROR")
            # 根据需要，这里可以决定是否要中止任务
            # from PyQt6.QtWidgets import QMessageBox
            # QMessageBox.critical(None, "配置错误", f"无法加载最新配置: {e}")
            # return

        # 检查是否有任务在运行（基于状态而不是线程）
        if self.state_manager.is_translating():
            self._ui_log("一个任务已经在运行中。", "WARNING")
            return
        
        # 如果有旧线程还在运行，等待它结束（不使用 terminate）
        if self.thread is not None and self.thread.isRunning():
            self._ui_log("检测到旧线程还在运行，正在请求停止...", "WARNING")
            self.state_manager.set_status_message("正在停止旧任务...")
            
            # 通知 worker 停止
            if self.worker:
                try:
                    self.worker.stop()
                except Exception as e:
                    self._ui_log(f"停止worker时出错: {e}", "WARNING")
            
            # 请求线程退出
            self.thread.quit()
            
            # 等待最多5秒（给渲染任务足够的时间完成）
            wait_time = 5000  # 5秒
            if not self.thread.wait(wait_time):
                self._ui_log(f"旧线程在{wait_time}ms内未停止，强制终止", "ERROR")
                # 最后手段：强制终止（可能导致资源泄漏，但比线程冲突好）
                self.thread.terminate()
                self.thread.wait()  # 等待终止完成
                self._ui_log("旧线程已被强制终止", "WARNING")
            else:
                self._ui_log("旧线程已正常停止")
            
            # 清理引用
            self.thread = None
            self.worker = None
            
            # 重置状态
            self.state_manager.set_translating(False)
            self.state_manager.set_status_message("就绪")

        # 检查文件列表是否为空
        files_to_process = self._resolve_input_files()
        if not files_to_process:
            self._ui_log("没有找到有效的图片文件，任务中止", "WARNING")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                self._t("File List Empty"),
                self._t("Please add image files to translate!\n\nYou can add files by:\n• Click 'Add Files' button\n• Click 'Add Folder' button\n• Drag and drop files to the list")
            )
            return

        # 检查输出目录是否合法
        output_path = self.config_service.get_config().app.last_output_path
        if not output_path or not os.path.isdir(output_path):
            self._ui_log(f"输出目录不合法: {output_path}", "WARNING")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                self._t("Invalid Output Directory"),
                self._t("Please set a valid output directory!\n\nYou can set it by:\n• Click 'Browse...' button to select directory\n• Enter path directly in the output directory field")
            )
            return

        self.saved_files_count = 0
        self.saved_files_list = []  # 重置保存文件列表
        self.thread = QThread()
        self.worker = TranslationWorker(
            files=files_to_process,
            config_dict=self.config_service.get_config().dict(),
            output_folder=self.config_service.get_config().app.last_output_path,
            root_dir=self.config_service.root_dir,
            file_to_folder_map=self.file_to_folder_map.copy()  # 传递文件到文件夹的映射
        )
        
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.process)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.on_task_finished)
        self.worker.error.connect(self.on_task_error)
        self.worker.progress.connect(self.on_task_progress)
        self.worker.log_received.connect(self.on_worker_log)
        self.worker.file_processed.connect(self.on_file_completed)
        self.worker.failed_files_recorded.connect(self._on_failed_files_recorded)

        self.thread.start()
        self._ui_log("翻译工作线程已启动。")
        self.state_manager.set_translating(True)
        self.state_manager.set_status_message("正在翻译...")

    @pyqtSlot(dict)
    def _on_failed_files_recorded(self, failed_files: dict):
        """处理失败文件记录信号"""
        if failed_files:
            self._ui_log(f"📋 记录失败文件: {sum(len(f) for f in failed_files.values())} 张")
            # 转发到UI层
            self.failed_files_recorded.emit(failed_files)
    
    @pyqtSlot(dict)
    def retry_failed_record(self, record: dict):
        """重试失败记录，使用顺序处理模式（逐张翻译）
        
        延迟重试流程：
        1. 文件已由 FileListContainer._on_failed_retry_requested 添加到列表
        2. 这里使用顺序模式启动翻译（禁用并发和批量）
        """
        # 检查是否有任务在运行
        if self.state_manager.is_translating():
            self._ui_log("当前有任务在运行，无法启动重试", "WARNING")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                self._t("Warning"),
                self._t("A task is already running. Please wait for it to complete.")
            )
            return
        
        total_count = record.get("total_count", 0)
        self._ui_log(f"🔄 开始重试 {total_count} 个失败文件（顺序模式）...")
        
        # 使用 QTimer 延迟启动翻译，确保 UI 更新完成
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._start_retry_task)
    
    def _start_retry_task(self):
        """启动重试任务，使用顺序处理模式（禁用并发和批量）"""
        # 重新加载配置
        try:
            self.config_service.reload_config()
        except Exception as e:
            self._ui_log(f"重新加载配置时发生错误: {e}", "ERROR")
        
        # 检查文件列表
        files_to_process = self._resolve_input_files()
        if not files_to_process:
            self._ui_log("没有找到有效的图片文件，重试中止", "WARNING")
            return
        
        # 检查输出目录
        output_path = self.config_service.get_config().app.last_output_path
        if not output_path or not os.path.isdir(output_path):
            self._ui_log(f"输出目录不合法: {output_path}", "WARNING")
            return
        
        # 从失败记录中提取历史上下文（用于重试时恢复翻译上下文）
        historical_context = None
        try:
            from manga_translator.utils.failure_record import load_failure_records, prepare_retry_context
            
            # 用第一个文件定位工作目录
            if files_to_process:
                records = load_failure_records(files_to_process[0])
                if records:
                    # 合并所有失败记录的历史上下文（它们应该来自同一个作品目录）
                    all_translations = []
                    all_originals = []
                    context_size = 3  # 默认值
                    
                    for record in records:
                        ctx = prepare_retry_context(record)
                        hist_ctx = ctx.get('historical_context', {})
                        if hist_ctx:
                            all_trans = hist_ctx.get('all_page_translations', [])
                            all_orig = hist_ctx.get('_original_page_texts', [])
                            context_size = hist_ctx.get('context_size', context_size)
                            
                            # 合并上下文（去重）
                            for t in all_trans:
                                if t not in all_translations:
                                    all_translations.append(t)
                            for o in all_orig:
                                if o not in all_originals:
                                    all_originals.append(o)
                    
                    if all_translations or all_originals:
                        historical_context = {
                            'all_page_translations': all_translations,
                            '_original_page_texts': all_originals,
                            'context_size': context_size
                        }
                        self._ui_log(
                            f"🔄 从失败记录中提取历史上下文: "
                            f"{len(all_translations)} 页翻译, {len(all_originals)} 页原文"
                        )
        except Exception as e:
            self._ui_log(f"提取历史上下文时出错: {e}", "WARNING")
        
        # 获取配置并修改为顺序模式
        config_dict = self.config_service.get_config().dict()
        # 禁用并发模式
        if 'cli' in config_dict:
            config_dict['cli']['batch_concurrent'] = False
        # 设置批量大小为 1，强制逐张处理
        if 'cli' in config_dict:
            config_dict['cli']['batch_size'] = 1
        
        self.saved_files_count = 0
        self.saved_files_list = []
        self.thread = QThread()
        self.worker = TranslationWorker(
            files=files_to_process,
            config_dict=config_dict,
            output_folder=output_path,
            root_dir=self.config_service.root_dir,
            file_to_folder_map=self.file_to_folder_map.copy(),
            historical_context=historical_context  # 传递历史上下文
        )
        
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.process)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.on_task_finished)
        self.worker.error.connect(self.on_task_error)
        self.worker.progress.connect(self.on_task_progress)
        self.worker.log_received.connect(self.on_worker_log)
        self.worker.file_processed.connect(self.on_file_completed)
        self.worker.failed_files_recorded.connect(self._on_failed_files_recorded)
        
        self.thread.start()
        self._ui_log("重试任务已启动（顺序模式）")
        self.state_manager.set_translating(True)
        self.state_manager.set_status_message("正在重试...")

    def on_task_finished(self, results):
        """处理任务完成信号，并根据需要保存批量任务的结果"""
        saved_files = []
        results_summary = None

        # Separate summary items from actual result items.
        result_items = []
        if results:
            try:
                for r in list(results):
                    if isinstance(r, dict) and r.get('results_summary'):
                        results_summary = r.get('results_summary')
                        continue
                    result_items.append(r)
            except Exception:
                # Fallback: treat all as result items
                result_items = list(results)
                results_summary = None

        self._ui_log(f"任务完成回调：收到 {len(result_items) if result_items else 0} 条结果")

        is_batch = isinstance(results_summary, dict)

        # Batch jobs: backend already saved files; UI only acknowledges.
        # Sequential jobs: saving is handled in `on_file_completed`.
        if is_batch and result_items:
            self._ui_log(f"批量翻译任务完成，收到 {len(result_items)} 个结果。正在整理...")
            try:
                config = self.config_service.get_config()
                output_format = config.cli.format
                save_quality = config.cli.save_quality
                output_folder = config.app.last_output_path
                save_to_source_dir = config.cli.save_to_source_dir

                if not output_folder and not save_to_source_dir:
                    self._ui_log("输出目录未设置，无法整理结果。", "ERROR")
                    self.state_manager.set_status_message("错误：输出目录未设置！")
                else:
                    for result in result_items:
                        if not isinstance(result, dict):
                            continue
                        if not result.get('success'):
                            continue

                        original_path = result.get('original_path')
                        if not original_path:
                            continue

                        # In batch mode, image_data is None because the backend already saved the file.
                        # We just need to acknowledge it.
                        if result.get('image_data') is None:
                            if save_to_source_dir:
                                source_dir = os.path.dirname(original_path)
                                final_output_folder = os.path.join(source_dir, 'manga_translator_work', 'result')
                                translated_file = os.path.join(final_output_folder, os.path.basename(original_path))
                            else:
                                # 构造翻译后的图片路径
                                source_folder = self.file_to_folder_map.get(original_path)
                                if source_folder:
                                    folder_name = os.path.basename(source_folder)
                                    final_output_folder = os.path.join(output_folder, folder_name)
                                    translated_file = os.path.join(final_output_folder, os.path.basename(original_path))
                                else:
                                    translated_file = os.path.join(output_folder, os.path.basename(original_path))

                            translated_file = os.path.normpath(translated_file)
                            saved_files.append(translated_file)
                        else:
                            # This handles cases where a result with image_data is present in a batch
                            try:
                                if not output_folder:
                                    raise ValueError("输出目录未设置")

                                base_filename = os.path.splitext(os.path.basename(original_path))[0]
                                file_extension = f".{output_format}" if output_format and output_format != "不指定" else ".png"
                                output_filename = f"{base_filename}_translated{file_extension}"
                                final_output_path = os.path.join(output_folder, output_filename)
                                os.makedirs(output_folder, exist_ok=True)

                                save_kwargs = {}
                                image_to_save = result['image_data']

                                # Convert RGBA to RGB for JPEG format
                                if file_extension in ['.jpg', '.jpeg']:
                                    if image_to_save.mode == 'RGBA':
                                        image_to_save = image_to_save.convert('RGB')
                                    save_kwargs['quality'] = save_quality
                                elif file_extension == '.webp':
                                    save_kwargs['quality'] = save_quality

                                image_to_save.save(final_output_path, **save_kwargs)
                                saved_files.append(final_output_path)
                                self._ui_log(f"成功保存文件: {final_output_path}")
                            except Exception as e:
                                self._ui_log(f"保存文件 {original_path} 时出错: {e}", "ERROR")

            except Exception as e:
                self._ui_log(f"处理批量任务结果时发生严重错误: {e}", "ERROR")

        # For sequential mode, include already-saved files and also skipped-existing outputs (if any).
        if not is_batch:
            if self.saved_files_list:
                saved_files = list(self.saved_files_list)

            try:
                config = self.config_service.get_config()
                output_folder = config.app.last_output_path
                save_to_source_dir = config.cli.save_to_source_dir

                for result in result_items:
                    if not isinstance(result, dict):
                        continue
                    if not result.get('success'):
                        continue
                    if not result.get('skipped'):
                        continue

                    original_path = result.get('original_path')
                    if not original_path:
                        continue

                    if save_to_source_dir:
                        source_dir = os.path.dirname(original_path)
                        final_output_folder = os.path.join(source_dir, 'manga_translator_work', 'result')
                        translated_file = os.path.join(final_output_folder, os.path.basename(original_path))
                    else:
                        if not output_folder:
                            continue
                        source_folder = self.file_to_folder_map.get(original_path)
                        if source_folder:
                            folder_name = os.path.basename(source_folder)
                            final_output_folder = os.path.join(output_folder, folder_name)
                            translated_file = os.path.join(final_output_folder, os.path.basename(original_path))
                        else:
                            translated_file = os.path.join(output_folder, os.path.basename(original_path))

                    translated_file = os.path.normpath(translated_file)
                    saved_files.append(translated_file)
            except Exception:
                # Keep best-effort behavior; do not fail the completion path.
                pass
        else:
            if not saved_files and self.saved_files_list:
                saved_files = self.saved_files_list.copy()

        # Compute a consistent summary line.
        summary_total = None
        summary_success = None
        summary_failed = None
        summary_skipped = None
        summary_skipped_backend = None
        if isinstance(results_summary, dict):
            try:
                summary_total = int(results_summary.get('total', 0) or 0)
                summary_success = int(results_summary.get('success', 0) or 0)
                summary_failed = int(results_summary.get('failed', 0) or 0)
                summary_skipped = int(results_summary.get('skipped', 0) or 0)
                summary_skipped_backend = int(results_summary.get('skipped_backend', 0) or 0)
            except Exception:
                summary_total = None

        # Fallback summary if not provided
        if summary_total is None:
            try:
                # Count only dict items
                ok = 0
                fail = 0
                skip = 0
                for r in result_items:
                    if not isinstance(r, dict):
                        continue
                    if r.get('success'):
                        ok += 1
                        if r.get('skipped'):
                            skip += 1
                    else:
                        fail += 1
                summary_total = ok + fail
                summary_success = ok - skip
                summary_failed = fail
                summary_skipped = skip
                summary_skipped_backend = 0
            except Exception:
                summary_total = 0
                summary_success = 0
                summary_failed = 0
                summary_skipped = 0
                summary_skipped_backend = 0

        extra_skipped = summary_skipped_backend or 0
        if extra_skipped > 0:
            completion_msg = f"翻译任务完成：总计 {summary_total} 个文件，成功 {summary_success} 个，跳过 {summary_skipped} 个（已存在）+ {extra_skipped} 个（无文本等），失败 {summary_failed} 个。"
        else:
            completion_msg = f"翻译任务完成：总计 {summary_total} 个文件，成功 {summary_success} 个，跳过 {summary_skipped} 个（已存在），失败 {summary_failed} 个。"

        self._ui_log(completion_msg)

        try:
            self.state_manager.set_translating(False)
            self.state_manager.set_status_message(completion_msg)

            # 重置主视图的进度条
            if hasattr(self, 'main_view') and self.main_view:
                self.main_view.reset_progress()

            # 播放系统提示音
            try:
                from PyQt6.QtWidgets import QApplication
                QApplication.beep()
            except Exception:
                pass

            # 使用列表副本发送信号，避免引用问题
            payload = list(saved_files)
            if isinstance(results_summary, dict):
                payload.append({'results_summary': results_summary})
            self.task_completed.emit(payload)
        except Exception as e:
            self._ui_log(f"完成任务状态更新或信号发射时发生致命错误: {e}", "ERROR")
            import traceback
            traceback.print_exc()

        # 注意：将清理逻辑移出 finally 块，使用 QTimer 延迟执行
        # 这样可以确保信号有足够时间被主线程处理
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._cleanup_after_task)
    
    def _cleanup_after_task(self):
        """延迟清理任务相关资源"""
        try:
            # 清理线程引用（线程应该已经通过deleteLater自动清理）
            # 只在线程仍在运行时进行额外处理
            # 使用 sip 检查对象是否已被删除
            thread_valid = False
            try:
                if self.thread is not None:
                    # 尝试访问线程属性来检查是否有效
                    thread_valid = self.thread.isRunning()
            except RuntimeError:
                # C++ 对象已被删除
                thread_valid = False
            
            if thread_valid:
                self._ui_log("任务完成但线程仍在运行，请求退出...", "WARNING")
                self.thread.quit()
                # 不阻塞UI太久，最多等待500ms
                if not self.thread.wait(500):
                    self._ui_log("线程未在500ms内停止，将由Qt事件循环自动清理", "WARNING")
            
            # 清理压缩包解压的临时文件
            if hasattr(self, 'archive_to_temp_map') and self.archive_to_temp_map:
                try:
                    from desktop_qt_ui.utils.archive_extractor import cleanup_archive_temp
                    for archive_path in list(self.archive_to_temp_map.keys()):
                        cleanup_archive_temp(archive_path)
                    self.archive_to_temp_map.clear()
                    self._ui_log("已清理压缩包临时文件")
                except Exception as cleanup_error:
                    self._ui_log(f"清理临时文件时出错: {cleanup_error}", "WARNING")

            # 翻译任务完成后释放 CUDA 缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self._ui_log("翻译完成后已调用 torch.cuda.empty_cache()", "DEBUG")
            except Exception as memory_cleanup_error:
                self._ui_log(f"调用 torch.cuda.empty_cache() 失败: {memory_cleanup_error}", "WARNING")
        except Exception as e:
            # 忽略 C++ 对象已删除的错误
            if "has been deleted" not in str(e):
                self._ui_log(f"清理任务资源时出错: {e}", "WARNING")
        finally:
            # 清理引用，让Qt的deleteLater机制处理实际的对象销毁
            self.thread = None
            self.worker = None
    
    def on_task_error(self, error_message):
        # 错误信息已在 worker 中通过详细错误提示框显示，这里不再重复输出
        self.state_manager.set_translating(False)
        self.state_manager.set_status_message(f"任务失败: {error_message}")
        
        # 重置主视图的进度条
        if hasattr(self, 'main_view') and self.main_view:
            self.main_view.reset_progress()
        
        # 清理线程
        if self.thread and self.thread.isRunning():
            self._ui_log("错误发生但线程仍在运行，请求退出...", "WARNING")
            self.thread.quit()
            if not self.thread.wait(2000):
                self._ui_log("线程未在2秒内停止，将由Qt事件循环自动清理", "WARNING")
        
        # 清理引用
        self.thread = None
        self.worker = None

    def on_task_progress(self, current, total, message):
        self._ui_log(f"[进度] {current}/{total}: {message}")
        percentage = (current / total) * 100 if total > 0 else 0
        self.state_manager.set_translation_progress(percentage)
        self.state_manager.set_status_message(f"[{current}/{total}] {message}")
        
        # 更新主视图的进度条
        if hasattr(self, 'main_view') and self.main_view:
            self.main_view.update_progress(current, total, message)

    def stop_task(self) -> bool:
        """停止翻译任务（优雅停止，不使用 terminate）"""
        if self.thread and self.thread.isRunning():
            self._ui_log("正在请求停止翻译线程...")

            # 立即更新按钮为"停止中..."状态并禁用，防止重复点击
            # 但不更新 is_translating 状态，等线程真正结束后再更新
            self.state_manager.set_status_message("正在停止...")
            if hasattr(self, 'main_view') and self.main_view:
                self.main_view.set_stopping_state()

            # 1. 通知 worker 停止（设置标志）
            if self.worker:
                try:
                    self.worker.stop()
                except:
                    pass

            # 2. 请求线程退出事件循环
            self.thread.quit()
            
            # 3. 连接 finished 信号以清理资源，并在线程真正结束后更新状态
            def on_thread_finished():
                self._ui_log("翻译线程已正常停止")
                # 线程结束后才更新翻译状态，与进度条重置时机一致
                self.state_manager.set_translating(False)
                self.state_manager.set_status_message("任务已停止")
                # 重置进度条
                if hasattr(self, 'main_view') and self.main_view:
                    self.main_view.reset_progress()
                self.thread = None
                self.worker = None
            
            try:
                self.thread.finished.disconnect()
            except:
                pass
            self.thread.finished.connect(on_thread_finished)

            return True
        
        self._ui_log("请求停止任务，但没有正在运行的线程。", "WARNING")
        self.state_manager.set_translating(False)
        return False
    # endregion

    # region 应用生命周期
    def initialize(self) -> bool:
        try:
            # The config is already loaded at startup. We just need to ensure the UI
            # reflects the loaded state without triggering a full, blocking rebuild.
            
            # Get the already loaded config
            config = self.config_service.get_config()

            # Manually emit the signal to populate UI options
            self.config_loaded.emit(config.dict())

            # Manually emit the signal to update the output path display in the UI
            if config.app.last_output_path:
                self.output_path_updated.emit(config.app.last_output_path)
            
            # Ensure the config path is stored in the state manager
            default_config_path = self.config_service.get_default_config_path()
            if os.path.exists(default_config_path):
                self.state_manager.set_state(AppStateKey.CONFIG_PATH, default_config_path)

            self.state_manager.set_app_ready(True)
            self.state_manager.set_status_message("就绪")
            self._ui_log("应用初始化完成")
            return True
        except Exception as e:
            self._ui_log(f"应用初始化异常: {e}", "ERROR")
            return False
    
    def shutdown(self):
        """应用关闭时的清理"""
        try:
            if self.state_manager.is_translating() or (self.thread and self.thread.isRunning()):
                self._ui_log("应用关闭中，停止翻译任务...")
                
                # 通知 worker 停止
                if self.worker:
                    try:
                        self.worker.stop()
                    except Exception as e:
                        self._ui_log(f"停止worker时出错: {e}", "WARNING")
                
                # 请求线程退出并等待（最多3秒）
                if self.thread and self.thread.isRunning():
                    self.thread.quit()
                    if not self.thread.wait(3000):
                        self._ui_log("线程3秒内未停止，强制终止", "WARNING")
                        self.thread.terminate()
                        self.thread.wait()
                    else:
                        self._ui_log("翻译线程已正常停止")
                
                self.thread = None
                self.worker = None
                self.state_manager.set_translating(False)
            
            # 关闭缩略图加载线程池
            try:
                from desktop_qt_ui.widgets.file_list_view import shutdown_thumbnail_executor
                shutdown_thumbnail_executor()
            except Exception:
                pass
            
            if self.translation_service:
                pass
        except Exception as e:
            self._ui_log(f"应用关闭异常: {e}", "ERROR")
    # endregion

class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class TranslationWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    log_received = pyqtSignal(str)
    file_processed = pyqtSignal(dict)
    failed_files_recorded = pyqtSignal(dict)  # 失败文件记录信号: {folder_path: [filename, ...]}

    def __init__(self, files, config_dict, output_folder, root_dir, file_to_folder_map=None, historical_context=None):
        super().__init__()
        self.files = files
        self.config_dict = config_dict
        self.output_folder = output_folder
        self.root_dir = root_dir
        self.file_to_folder_map = file_to_folder_map or {}  # 文件到文件夹的映射
        self.historical_context = historical_context  # 重试时恢复的历史上下文
        self._is_running = True
        self._current_task = None  # 保存当前运行的异步任务
        self._finished_emitted = False
        self.results_summary = None
        self.i18n = get_i18n_manager()
        self.logger = get_logger(__name__)
        self.file_service = get_file_service()
    
    def _t(self, key: str, **kwargs) -> str:
        """翻译辅助方法"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key
    
    def _calculate_output_path(self, image_path: str, save_info: dict) -> str:
        """
        计算输出文件的完整路径（用于预检查文件是否存在）
        
        Args:
            image_path: 输入图片的路径
            save_info: 包含输出配置的字典
                
        Returns:
            str: 计算后的输出文件完整路径
        """
        from manga_translator.utils.path_manager import build_unique_folder_aliases
        
        output_folder = save_info.get('output_folder')
        output_format = save_info.get('format')
        save_to_source_dir = save_info.get('save_to_source_dir', False)
        
        file_path = image_path
        parent_dir = os.path.normpath(os.path.dirname(file_path))
        
        # 检查是否启用了"输出到原图目录"模式
        if save_to_source_dir:
            # 输出到原图所在目录的 manga_translator_work/result 子目录
            final_output_dir = os.path.join(parent_dir, 'manga_translator_work', 'result')
        else:
            # 默认输出到输出目录根
            final_output_dir = output_folder
            
            # 获取图片的直接父目录名（如 "Chapter 74"）
            parent_name = os.path.basename(parent_dir)
            
            # 检查是否需要保留目录结构
            # 只有当父目录名包含 "chapter" 时才创建子目录
            if parent_name and 'chapter' in parent_name.lower():
                # 生成所有图片父目录的唯一别名，避免同名目录冲突
                parent_dir_aliases = save_info.get('parent_dir_aliases')
                if parent_dir_aliases is None:
                    # 收集所有图片的父目录
                    all_parent_dirs = save_info.get('all_parent_dirs', set())
                    if not all_parent_dirs:
                        # 如果没有预先收集，仅使用当前父目录
                        all_parent_dirs = {parent_dir}
                    parent_dir_aliases = build_unique_folder_aliases(all_parent_dirs)
                    save_info['parent_dir_aliases'] = parent_dir_aliases
                
                # 获取当前父目录的别名
                folder_alias = parent_dir_aliases.get(parent_dir) or parent_name
                final_output_dir = os.path.join(output_folder, folder_alias)
            # 如果父目录不是 Chapter，直接放到输出根目录（final_output_dir 保持为 output_folder）
            
            # 特殊处理：压缩包文件
            source_folder = self.file_to_folder_map.get(image_path)
            if source_folder and self.file_service.is_archive_file(source_folder):
                # 文件来自压缩包，使用压缩包名称（不含扩展名）作为输出子目录
                archive_name = os.path.splitext(os.path.basename(source_folder))[0]
                final_output_dir = os.path.join(output_folder, archive_name)
            
            final_output_dir = os.path.normpath(final_output_dir)
        
        # 处理输出文件名和格式
        base_filename, _ = os.path.splitext(os.path.basename(file_path))
        if output_format and output_format.strip() and output_format.lower() not in ['none', '不指定']:
            output_filename = f"{base_filename}.{output_format}"
        else:
            output_filename = os.path.basename(file_path)
        
        final_output_path = os.path.join(final_output_dir, output_filename)
        return final_output_path

    def _emit_failed_files_record(self, results: list):
        """
        收集失败文件信息并发射信号
        
        Args:
            results: 翻译结果列表 [{'success': bool, 'original_path': str, ...}, ...]
        """
        self.log_received.emit(f"📋 [失败记录] 开始收集失败文件，results数量: {len(results)}")
        
        # 收集失败的文件，按文件夹分组
        failed_files = {}  # {folder_path: [filename, ...]}
        
        for result in results:
            if not result.get('success', False):
                original_path = result.get('original_path', '')
                self.log_received.emit(f"📋 [失败记录] 找到失败文件: {original_path}")
                if original_path and original_path != 'Unknown':
                    folder_path = os.path.dirname(original_path)
                    filename = os.path.basename(original_path)
                    if folder_path not in failed_files:
                        failed_files[folder_path] = []
                    failed_files[folder_path].append(filename)
        
        self.log_received.emit(f"📋 [失败记录] 收集完成，失败文件数: {sum(len(f) for f in failed_files.values()) if failed_files else 0}")
        
        if failed_files:
            self.log_received.emit(f"📋 [失败记录] 发射信号: {failed_files}")
            self.failed_files_recorded.emit(failed_files)
        else:
            self.log_received.emit("📋 [失败记录] 没有找到失败文件，不发射信号")

    def stop(self):
        self.log_received.emit("--- Stop request received.")
        self._is_running = False
        # 取消当前运行的异步任务
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        
        # 使用统一的内存清理模块
        try:
            from desktop_qt_ui.utils.memory_cleanup import full_memory_cleanup
            full_memory_cleanup(log_callback=lambda msg: self.log_received.emit(msg))
        except Exception as e:
            self.log_received.emit(f"--- [CLEANUP] Warning: Failed to cleanup: {e}")

    def _build_friendly_error_message(self, error_message: str, error_traceback: str) -> str:
        """
        根据错误信息构建友好的中文错误提示
        """
        friendly_msg = "\n" + "="*80 + "\n"
        friendly_msg += "❌ 翻译任务失败\n"
        friendly_msg += "="*80 + "\n\n"
        
        # 如果是"达到最大尝试次数"的错误，提取真正的错误原因
        real_error = error_message
        if "达到最大尝试次数" in error_message and "最后一次错误:" in error_message:
            # 提取真正的错误原因
            try:
                real_error = error_message.split("最后一次错误:")[1].strip()
            except:
                pass
        
        # 检查是否是AI断句检查失败
        if ("BR markers missing" in real_error or 
            "AI断句检查" in error_message or 
            "BRMarkersValidationException" in error_traceback or
            "_validate_br_markers" in error_traceback):
            friendly_msg += "🔍 错误原因：AI断句检查失败\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   AI翻译时未能正确添加断句标记 [BR]，导致多次重试后仍然失败。\n\n"
            friendly_msg += "💡 解决方案（选择其一）：\n"
            friendly_msg += "   1. ⭐ 关闭「AI断句检查」选项（推荐）\n"
            friendly_msg += "      - 位置：高级设置 → 渲染设置 → AI断句检查\n"
            friendly_msg += "      - 说明：允许AI在少数情况下不添加断句标记\n\n"
            friendly_msg += "   2. 增加「重试次数」\n"
            friendly_msg += "      - 位置：通用设置 → 重试次数\n"
            friendly_msg += "      - 建议：设置为 10 或更高（-1 表示无限重试）\n\n"
            friendly_msg += "   3. 更换翻译模型\n"
            friendly_msg += "      - 某些模型对断句标记的理解更好\n"
            friendly_msg += "      - 建议：尝试 gpt-4o 或 gemini-2.0-flash-exp\n\n"
            friendly_msg += "   4. 关闭「AI断句」功能\n"
            friendly_msg += "      - 位置：高级设置 → 渲染设置 → AI断句\n"
            friendly_msg += "      - 说明：使用传统的自动换行（可能导致排版不够精确）\n\n"
            friendly_msg += "   5. 减小批量大小\n"
            friendly_msg += "      - 位置：高级设置 → 批量大小\n"
            friendly_msg += "      - 建议：将批量大小减小（如从 3 减到 1 或 2）\n"
            friendly_msg += "      - 说明：批量处理的文本越少，AI越容易正确添加断句标记\n\n"
        
        # 检查是否是翻译数量不匹配错误
        elif "翻译数量不匹配" in real_error or "Translation count mismatch" in real_error:
            friendly_msg += "🔍 错误原因：翻译数量不匹配\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   AI返回的翻译条数与原文条数不一致。\n"
            friendly_msg += "   这通常是因为AI将多条文本合并翻译，或漏掉了某些文本。\n\n"
            friendly_msg += "💡 解决方案（选择其一）：\n"
            friendly_msg += "   1. ⭐ 增加「重试次数」（推荐）\n"
            friendly_msg += "      - 位置：通用设置 → 重试次数\n"
            friendly_msg += "      - 建议：设置为 10 或更高（-1 表示无限重试）\n"
            friendly_msg += "      - 说明：多次重试通常能让AI返回正确数量的翻译\n\n"
            friendly_msg += "   2. 更换翻译模型\n"
            friendly_msg += "      - 某些模型对指令的遵循能力更强\n"
            friendly_msg += "      - 建议：尝试 gpt-4o 或 gemini-2.0-flash-exp\n\n"
            friendly_msg += "   3. 减小批量大小\n"
            friendly_msg += "      - 位置：高级设置 → 批量大小\n"
            friendly_msg += "      - 建议：将批量大小减小（如从 3 减到 1 或 2）\n"
            friendly_msg += "      - 说明：批量处理的文本越少，AI越不容易出错\n\n"
        
        # 检查是否是翻译质量检查失败
        elif "翻译质量检查失败" in real_error or "Quality check failed" in real_error:
            friendly_msg += "🔍 错误原因：翻译质量检查失败\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   AI返回的翻译存在质量问题，如空翻译、合并翻译或可疑符号。\n\n"
            friendly_msg += "💡 解决方案（选择其一）：\n"
            friendly_msg += "   1. ⭐ 增加「重试次数」（推荐）\n"
            friendly_msg += "      - 位置：通用设置 → 重试次数\n"
            friendly_msg += "      - 建议：设置为 10 或更高（-1 表示无限重试）\n\n"
            friendly_msg += "   2. 更换翻译模型\n"
            friendly_msg += "      - 某些模型翻译质量更稳定\n"
            friendly_msg += "      - 建议：尝试 gpt-4o 或 gemini-2.0-flash-exp\n\n"
            friendly_msg += "   3. 减小批量大小\n"
            friendly_msg += "      - 位置：高级设置 → 批量大小\n"
            friendly_msg += "      - 建议：将批量大小减小（如从 3 减到 1 或 2）\n"
            friendly_msg += "      - 说明：批量处理的文本越少，AI翻译质量越稳定\n\n"
        
        # 检查是否是模型不支持多模态
        elif "不支持多模态" in real_error or "multimodal" in real_error.lower() or "vision" in real_error.lower():
            friendly_msg += "🔍 错误原因：模型不支持多模态输入\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   当前使用的是「高质量翻译器」（openai_hq 或 gemini_hq），\n"
            friendly_msg += "   这些翻译器需要发送图片给AI进行分析，但当前模型不支持图片输入。\n\n"
            friendly_msg += "💡 解决方案（选择其一）：\n"
            friendly_msg += "   1. ⭐ 更换为支持多模态的模型（推荐）\n"
            friendly_msg += "      - OpenAI: gpt-4o, gpt-4-turbo, gpt-4-vision-preview\n"
            friendly_msg += "      - Gemini: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash\n"
            friendly_msg += "      - 注意：DeepSeek模型不支持多模态\n\n"
            friendly_msg += "   2. 切换到普通翻译器\n"
            friendly_msg += "      - 位置：翻译设置 → 翻译器\n"
            friendly_msg += "      - 将 openai_hq 改为 openai\n"
            friendly_msg += "      - 将 gemini_hq 改为 gemini\n"
            friendly_msg += "      - 说明：普通翻译器不需要发送图片，只翻译文本\n\n"
        
        # 检查是否是404错误（API地址或模型配置错误）
        elif "API_404_ERROR" in real_error or "404" in real_error or "HTML错误页面" in real_error:
            friendly_msg += "🔍 错误原因：API返回404错误\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   API返回了HTML格式的404错误页面，而不是正常的JSON响应。\n"
            friendly_msg += "   这通常意味着API地址错误或模型名称不存在。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. ⭐ 检查API地址配置（最常见）\n"
            friendly_msg += "      - 位置：翻译设置 → 环境变量 → OPENAI_API_BASE\n"
            friendly_msg += "      - 正确格式：https://api.openai.com/v1\n"
            friendly_msg += "      - 注意：地址末尾必须是 /v1，不要多加或少加路径\n\n"
            friendly_msg += "   2. 检查模型名称是否正确\n"
            friendly_msg += "      - 位置：翻译设置 → 环境变量 → OPENAI_MODEL\n"
            friendly_msg += "      - 确认模型名称与API服务提供的模型完全匹配\n"
            friendly_msg += "      - 注意：模型名称区分大小写\n"
            friendly_msg += "      - 提示：可以使用「测试连接」功能查看可用模型列表\n\n"
            friendly_msg += "   3. 如果使用自定义API（如中转API、第三方服务）\n"
            friendly_msg += "      - 确认中转服务的API地址格式\n"
            friendly_msg += "      - 确认中转服务支持你使用的模型\n"
            friendly_msg += "      - 联系中转服务提供商确认配置\n\n"
        
        # 检查是否是API密钥错误
        elif "api key" in real_error.lower() or "authentication" in real_error.lower() or "unauthorized" in real_error.lower() or "401" in real_error:
            friendly_msg += "🔍 错误原因：API密钥验证失败\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   API密钥无效、过期或未正确配置。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. 检查API密钥是否正确\n"
            friendly_msg += "      - 位置：翻译设置 → 环境变量配置区域\n"
            friendly_msg += "      - 确认密钥没有多余的空格或换行\n\n"
            friendly_msg += "   2. 验证API密钥是否有效\n"
            friendly_msg += "      - OpenAI: https://platform.openai.com/api-keys\n"
            friendly_msg += "      - Gemini: https://aistudio.google.com/app/apikey\n\n"
            friendly_msg += "   3. 检查API额度是否用完\n"
            friendly_msg += "      - 登录对应平台查看余额和使用情况\n\n"
        
        # 检查是否是网络连接错误
        elif "connection" in real_error.lower() or "连接" in real_error or "timeout" in real_error.lower() or "超时" in real_error or "network" in real_error.lower() or "网络" in real_error:
            friendly_msg += "🔍 错误原因：网络连接失败\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   无法连接到API服务器，可能是网络问题或需要代理。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. 检查网络连接\n"
            friendly_msg += "      - 确认电脑可以正常访问互联网\n\n"
            friendly_msg += "   2. 检查API地址是否正确\n"
            friendly_msg += "      - 位置：翻译设置 → 环境变量 → API_BASE\n"
            friendly_msg += "      - 默认值：https://api.openai.com/v1\n\n"
        
        # 检查是否是速率限制错误
        elif "rate limit" in real_error.lower() or "429" in real_error or "too many requests" in real_error.lower():
            friendly_msg += "🔍 错误原因：API请求被拒绝 (HTTP 429)\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   HTTP 429 错误有多种可能原因：\n"
            friendly_msg += "   • API密钥错误或无效\n"
            friendly_msg += "   • 账户余额不足或欠费\n"
            friendly_msg += "   • 请求速率超过限制（RPM/TPM）\n"
            friendly_msg += "   • 当前账户级别不支持该模型\n\n"
            friendly_msg += "💡 解决方案（按顺序检查）：\n"
            friendly_msg += "   1. ⭐ 检查API密钥是否正确（最常见）\n"
            friendly_msg += "      - 位置：翻译设置 → 环境变量配置区域\n"
            friendly_msg += "      - 确认密钥没有多余的空格或换行\n"
            friendly_msg += "      - 使用「测试连接」功能验证密钥是否有效\n\n"
            friendly_msg += "   2. 检查账户余额和状态\n"
            friendly_msg += "      - OpenAI: https://platform.openai.com/usage\n"
            friendly_msg += "      - Gemini: https://aistudio.google.com/app/apikey\n"
            friendly_msg += "      - 确认账户余额充足且未欠费\n"
            friendly_msg += "      - 确认账户状态正常（未被限制）\n\n"
            friendly_msg += "   3. 检查模型是否支持\n"
            friendly_msg += "      - 某些模型需要特定的账户级别或付费套餐\n"
            friendly_msg += "      - 例如：GPT-4 需要付费账户，免费账户只能用 GPT-3.5\n"
            friendly_msg += "      - 尝试更换为账户支持的模型\n\n"
            friendly_msg += "   4. 降低请求速率\n"
            friendly_msg += "      - 位置：通用设置 → 每分钟最大请求数\n"
            friendly_msg += "      - 建议：设置为 3-10（取决于API套餐）\n"
            friendly_msg += "      - 免费账户建议设置为 3\n\n"
            friendly_msg += "   5. 稍后重试\n"
            friendly_msg += "      - 等待几分钟后再次尝试翻译\n\n"
            friendly_msg += "   6. 升级API套餐\n"
            friendly_msg += "      - 联系API提供商升级到更高级别的套餐\n\n"
        
        # 检查是否是403禁止访问错误
        elif "403" in real_error or "forbidden" in real_error.lower():
            friendly_msg += "🔍 错误原因：访问被拒绝 (HTTP 403)\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   服务器拒绝访问，可能是权限不足或地区限制。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. 检查API密钥权限\n"
            friendly_msg += "      - 确认API密钥有访问该服务的权限\n\n"
            friendly_msg += "   2. 检查账户状态\n"
            friendly_msg += "      - 确认账户未被封禁或限制\n\n"

        
        # 检查是否是404未找到错误
        elif "404" in real_error or "not found" in real_error.lower():
            friendly_msg += "🔍 错误原因：资源未找到 (HTTP 404)\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   请求的API端点不存在或模型名称错误。\n"
            friendly_msg += "   也可能是翻译器类型与API地址不匹配。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. ⭐ 检查翻译器类型是否匹配API地址（最常见）\n"
            friendly_msg += "      - 如果API地址是 xxxx/v1 格式（OpenAI兼容接口）\n"
            friendly_msg += "        → 应选择「OpenAI」或「OpenAI高质量」翻译器\n"
            friendly_msg += "      - 如果使用 Gemini 官方 API (generativelanguage.googleapis.com)\n"
            friendly_msg += "        → 应选择「Gemini」或「Gemini高质量」翻译器\n"
            friendly_msg += "      - 位置：翻译设置 → 翻译器\n\n"
            friendly_msg += "   2. 检查API地址是否正确\n"
            friendly_msg += "      - 位置：翻译设置 → 环境变量 → API_BASE\n"
            friendly_msg += "      - OpenAI默认：https://api.openai.com/v1\n"
            friendly_msg += "      - Gemini默认：https://generativelanguage.googleapis.com\n"
            friendly_msg += "      - 注意：地址末尾的 /v1 不要多加或少加\n\n"
            friendly_msg += "   3. 检查模型名称\n"
            friendly_msg += "      - 位置：翻译设置 → 环境变量 → MODEL\n"
            friendly_msg += "      - 确认模型名称拼写正确（如 gpt-4o 不是 gpt4o）\n"
            friendly_msg += "      - 使用「测试连接」功能查看可用模型列表\n\n"
            friendly_msg += "   4. 验证模型可用性\n"
            friendly_msg += "      - 某些模型可能已下线或更名\n"
            friendly_msg += "      - 访问官方文档查看可用模型列表\n\n"
        
        # 检查是否是500服务器错误
        elif "500" in real_error or "internal server error" in real_error.lower():
            friendly_msg += "🔍 错误原因：服务器内部错误 (HTTP 500)\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   API服务器遇到内部错误，这通常是临时问题。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. ⭐ 增加重试次数（推荐）\n"
            friendly_msg += "      - 位置：通用设置 → 重试次数\n"
            friendly_msg += "      - 建议：设置为 10 或更高\n"
            friendly_msg += "      - 服务器错误通常是临时的，重试可能成功\n\n"
            friendly_msg += "   2. 稍后重试\n"
            friendly_msg += "      - 等待几分钟，让服务器恢复正常\n\n"
            friendly_msg += "   3. 检查API服务状态\n"
            friendly_msg += "      - OpenAI: https://status.openai.com/\n"
            friendly_msg += "      - 查看是否有大规模服务中断\n\n"
        
        # 检查是否是502/503/504网关错误
        elif any(code in real_error for code in ["502", "503", "504"]) or "bad gateway" in real_error.lower() or "service unavailable" in real_error.lower() or "gateway timeout" in real_error.lower():
            error_code = "502/503/504"
            if "502" in real_error:
                error_code = "502"
            elif "503" in real_error:
                error_code = "503"
            elif "504" in real_error:
                error_code = "504"
            
            friendly_msg += f"🔍 错误原因：网关/服务不可用 (HTTP {error_code})\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   - 502: 网关接收到无效响应\n"
            friendly_msg += "   - 503: 服务暂时不可用（通常是维护或过载）\n"
            friendly_msg += "   - 504: 网关超时\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. ⭐ 等待后重试（推荐）\n"
            friendly_msg += "      - 这些错误通常是临时的\n"
            friendly_msg += "      - 等待5-10分钟后重新翻译\n\n"
            friendly_msg += "   2. 增加重试次数\n"
            friendly_msg += "      - 位置：通用设置 → 重试次数\n"
            friendly_msg += "      - 建议：设置为 10 或更高\n\n"
            friendly_msg += "   3. 检查API服务状态\n"
            friendly_msg += "      - 访问API提供商的状态页面\n"
            friendly_msg += "      - OpenAI: https://status.openai.com/\n\n"
            friendly_msg += "   4. 更换API地址\n"
            friendly_msg += "      - 如果使用第三方API中转，尝试更换地址\n\n"
        
        # 检查是否是内容过滤错误
        elif "content filter" in real_error.lower() or "content_filter" in real_error:
            friendly_msg += "🔍 错误原因：内容被安全策略拦截\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   AI检测到内容可能违反使用政策。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. 检查图片内容\n"
            friendly_msg += "      - 某些敏感内容可能被API拒绝处理\n\n"
            friendly_msg += "   2. 更换翻译器\n"
            friendly_msg += "      - 尝试使用其他翻译器（如 Gemini、DeepL）\n\n"
            friendly_msg += "   3. 增加重试次数\n"
            friendly_msg += "      - 位置：通用设置 → 重试次数\n"
            friendly_msg += "      - 有时重试可以解决临时的过滤问题\n\n"
        
        # 检查是否是语言不支持错误
        elif "language not supported" in real_error.lower() or "LanguageUnsupportedException" in error_traceback:
            friendly_msg += "🔍 错误原因：翻译器不支持当前语言\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. 更换翻译器\n"
            friendly_msg += "      - 位置：翻译设置 → 翻译器\n"
            friendly_msg += "      - 建议：使用支持更多语言的翻译器（如 OpenAI、Gemini）\n\n"
            friendly_msg += "   2. 检查目标语言设置\n"
            friendly_msg += "      - 位置：翻译设置 → 目标语言\n"
            friendly_msg += "      - 确认选择的语言被当前翻译器支持\n\n"
        
        # 检查是否是请求被拦截错误
        elif "blocked" in real_error.lower() or "request was blocked" in real_error.lower():
            friendly_msg += "🔍 错误原因：请求被API服务商拦截\n\n"
            friendly_msg += "📝 详细说明：\n"
            friendly_msg += "   API服务商（可能是第三方中转）拦截了你的请求。\n"
            friendly_msg += "   这通常是中转服务的反滥用机制或内容审核导致的。\n\n"
            friendly_msg += "💡 解决方案：\n"
            friendly_msg += "   1. ⭐ 更换API服务商（推荐）\n"
            friendly_msg += "      - 如果使用第三方中转API，尝试更换其他服务商\n"
            friendly_msg += "      - 或者使用官方API（如 api.openai.com）\n\n"
            friendly_msg += "   2. 切换到普通翻译器\n"
            friendly_msg += "      - 位置：翻译设置 → 翻译器\n"
            friendly_msg += "      - 将 openai_hq 改为 openai（不发送图片）\n"
            friendly_msg += "      - 某些中转服务不支持多模态（图片+文本）请求\n\n"
            friendly_msg += "   3. 检查API密钥状态\n"
            friendly_msg += "      - 确认API密钥未被封禁或限制\n"
            friendly_msg += "      - 联系API服务商确认账户状态\n\n"
        
        # 通用错误
        else:
            friendly_msg += "🔍 错误原因：\n"
            friendly_msg += f"   {error_message}\n\n"
            friendly_msg += "💡 通用解决方案：\n"
            friendly_msg += "   1. 检查配置是否正确\n"
            friendly_msg += "      - 翻译器、API密钥、模型名称等\n\n"
            friendly_msg += "   2. 增加重试次数\n"
            friendly_msg += "      - 位置：通用设置 → 重试次数\n"
            friendly_msg += "      - 建议：设置为 10 或更高\n\n"
            friendly_msg += "   3. 查看详细日志\n"
            friendly_msg += "      - 在日志框中查找更多错误信息\n\n"
        
        friendly_msg += "="*80 + "\n"
        friendly_msg += "📋 原始错误信息：\n"
        friendly_msg += "-"*80 + "\n"
        friendly_msg += f"{error_message}\n"
        if error_traceback and "Traceback" in error_traceback:
            friendly_msg += "\n" + "-"*80 + "\n"
            friendly_msg += "详细错误：\n"
            friendly_msg += "-"*80 + "\n"
            
            # 只保留API详细错误信息（不保留代码路径）
            lines = error_traceback.split('\n')
            api_error_lines = []
            
            for line in lines:
                # 只保留API错误信息行（包含详细的错误内容）
                if line.strip() and any(keyword in line for keyword in ['BadRequest', 'Error code:', "'error':", "'message':", "{'error':"]):
                    api_error_lines.append(line.strip())
            
            if api_error_lines:
                friendly_msg += '\n'.join(api_error_lines) + "\n"
                
        friendly_msg += "="*80 + "\n"
        
        return friendly_msg

    async def _do_processing(self):
        log_handler = QtLogHandler(self.log_received)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)
        manga_logger = logging.getLogger('manga_translator')
        manga_logger.addHandler(log_handler)
        
        # 根据 verbose 配置设置日志级别
        verbose = self.config_dict.get('cli', {}).get('verbose', False)
        log_level = logging.DEBUG if verbose else logging.INFO
        manga_logger.setLevel(log_level)
        
        # 根日志器设为 DEBUG 以允许所有日志通过
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 文件处理器始终为 DEBUG，其他处理器根据 verbose 设置
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG)  # 文件日志始终 DEBUG
            else:
                handler.setLevel(log_level)  # 控制台根据 verbose 设置

        results = []
        try:
            from manga_translator.config import (
                ColorizerConfig,
                Config,
                DetectorConfig,
                InpainterConfig,
                OcrConfig,
                RenderConfig,
                Translator,
                TranslatorConfig,
                UpscaleConfig,
            )
            from manga_translator.manga_translator import MangaTranslator
            from PIL import Image

            self.log_received.emit("--- 正在初始化翻译器...")
            translator_params = self.config_dict.get('cli', {})
            translator_params.update(self.config_dict)
            
            # 根据 verbose 设置设置日志级别
            verbose = translator_params.get('verbose', False)
            if hasattr(self, 'log_service') and self.log_service:
                self.log_service.set_console_log_level(verbose)
            
            font_filename = self.config_dict.get('render', {}).get('font_path')
            if font_filename:
                font_full_path = os.path.join(self.root_dir, 'fonts', font_filename)
                if os.path.exists(font_full_path):
                    translator_params['font_path'] = font_full_path
                    # 同时更新 config_dict 中的 font_path
                    self.config_dict['render']['font_path'] = font_full_path

            translator = MangaTranslator(params=translator_params)
            self.log_received.emit("--- 翻译器初始化完成")
            
            # 注入历史上下文（重试模式时恢复之前翻译的上下文）
            if self.historical_context:
                all_trans = self.historical_context.get('all_page_translations', [])
                all_orig = self.historical_context.get('_original_page_texts', [])
                if all_trans or all_orig:
                    translator.all_page_translations = all_trans.copy()
                    translator._original_page_texts = all_orig.copy()
                    self.log_received.emit(
                        f"--- 🔄 已恢复历史上下文: 携带 {len(all_trans)} 页翻译记录, "
                        f"{len(all_orig)} 页原文记录"
                    )
            
            # 注册进度钩子，接收后端的批次进度
            progress_signal = self.progress  # 捕获信号引用
            
            async def progress_hook(state: str, finished: bool):
                try:
                    if state.startswith("batch:"):
                        # 解析批次进度: "batch:start:end:total"
                        parts = state.split(":")
                        if len(parts) == 4:
                            batch_end = int(parts[2])
                            total = int(parts[3])
                            # 进度条显示图片数量
                            progress_signal.emit(batch_end, total, "")
                except Exception:
                    pass  # 忽略进度更新错误，不影响翻译流程
            
            translator.add_progress_hook(progress_hook)

            explicit_keys = {'render', 'upscale', 'translator', 'detector', 'colorizer', 'inpainter', 'ocr'}
            remaining_config = {
                k: v for k, v in self.config_dict.items() 
                if k in Config.__fields__ and k not in explicit_keys
            }

            render_config_data = self.config_dict.get('render', {}).copy()

            # 转换 direction 值：'h' -> 'horizontal', 'v' -> 'vertical'
            if 'direction' in render_config_data:
                direction_value = render_config_data['direction']
                if direction_value == 'h':
                    render_config_data['direction'] = 'horizontal'
                elif direction_value == 'v':
                    render_config_data['direction'] = 'vertical'

            translator_config_data = self.config_dict.get('translator', {}).copy()
            hq_prompt_path = translator_config_data.get('high_quality_prompt_path')
            if hq_prompt_path and not os.path.isabs(hq_prompt_path):
                full_prompt_path = os.path.join(self.root_dir, hq_prompt_path)
                if os.path.exists(full_prompt_path):
                    translator_config_data['high_quality_prompt_path'] = full_prompt_path
                else:
                    self.log_received.emit(f"--- WARNING: High quality prompt file not found at {full_prompt_path}")
            
            # 将 CLI 配置中的 attempts 复制到 translator 配置中
            cli_attempts = self.config_dict.get('cli', {}).get('attempts', -1)
            translator_config_data['attempts'] = cli_attempts
            self.log_received.emit(f"--- Setting translator attempts to: {cli_attempts} (from UI config)")

            # 转换超分倍数：'不使用' -> None, '2'/'4' -> int
            upscale_config_data = self.config_dict.get('upscale', {}).copy()
            if 'upscale_ratio' in upscale_config_data:
                ratio_value = upscale_config_data['upscale_ratio']
                if ratio_value == '不使用' or ratio_value is None:
                    upscale_config_data['upscale_ratio'] = None
                elif isinstance(ratio_value, str) and ratio_value in ('x2', 'x4', 'DAT2 x4'):
                    # mangajanai 的字符串选项，直接保留
                    upscale_config_data['upscale_ratio'] = ratio_value
                else:
                    try:
                        upscale_config_data['upscale_ratio'] = int(ratio_value)
                    except (ValueError, TypeError):
                        upscale_config_data['upscale_ratio'] = None

            config = Config(
                render=RenderConfig(**render_config_data),
                upscale=UpscaleConfig(**upscale_config_data),
                translator=TranslatorConfig(**translator_config_data),
                detector=DetectorConfig(**self.config_dict.get('detector', {})),
                colorizer=ColorizerConfig(**self.config_dict.get('colorizer', {})),
                inpainter=InpainterConfig(**self.config_dict.get('inpainter', {})),
                ocr=OcrConfig(**self.config_dict.get('ocr', {})),
                **remaining_config
            )
            self.log_received.emit(f"--- 配置对象创建完成: enable_watermark_filter={config.enable_watermark_filter}")

            translator_type = config.translator.translator
            is_hq = translator_type in [Translator.openai_hq, Translator.gemini_hq]
            batch_size = self.config_dict.get('cli', {}).get('batch_size', 1)

            # 准备save_info（所有模式都需要）
            output_format = self.config_dict.get('cli', {}).get('format')
            if not output_format or output_format == "不指定":
                output_format = None # Set to None to preserve original extension

            # 收集输入文件夹列表（从file_to_folder_map中获取）
            input_folders = set()
            for file_path in self.files:
                folder = self.file_to_folder_map.get(file_path)
                if folder:
                    input_folders.add(os.path.normpath(folder))
            
            # 收集所有图片的父目录（用于计算输出路径时避免同名目录冲突）
            all_parent_dirs = set()
            for file_path in self.files:
                parent_dir = os.path.normpath(os.path.dirname(file_path))
                if parent_dir:
                    all_parent_dirs.add(parent_dir)

            save_info = {
                'output_folder': self.output_folder,
                'format': output_format,
                'overwrite': self.config_dict.get('cli', {}).get('overwrite', True),
                'input_folders': input_folders,
                'all_parent_dirs': all_parent_dirs,
                'save_to_source_dir': self.config_dict.get('cli', {}).get('save_to_source_dir', False)
            }

            # Filter out existing files if overwrite is False
            original_files = self.files
            skipped_files = []
            files_to_process = []
            
            # 获取 cli_config（用于检查特殊模式）
            cli_config = self.config_dict.get('cli', {})
            
            if not save_info['overwrite']:
                self.log_received.emit("--- 🔍 检查已存在的文件（覆盖检测已禁用）...")
                self.logger.info("检查已存在的文件（覆盖检测已禁用）")
                
                for file_path in self.files:
                    try:
                        should_skip = False
                        
                        # 检查导出原文/翻译的TXT文件（如果启用）
                        if cli_config.get('template', False) and cli_config.get('save_text', False):
                            # 导出原文模式 - 检查TXT文件
                            from manga_translator.utils.path_manager import get_original_txt_path
                            txt_path = get_original_txt_path(file_path, create_dir=False)
                            if os.path.exists(txt_path):
                                should_skip = True
                        elif cli_config.get('generate_and_export', False):
                            # 导出翻译模式 - 检查TXT文件
                            from manga_translator.utils.path_manager import get_translated_txt_path
                            txt_path = get_translated_txt_path(file_path, create_dir=False)
                            if os.path.exists(txt_path):
                                should_skip = True
                        else:
                            # 普通翻译模式 - 检查图片文件
                            output_path = self._calculate_output_path(file_path, save_info)
                            if os.path.exists(output_path):
                                should_skip = True
                        
                        if should_skip:
                            skipped_files.append(file_path)
                            results.append({'success': True, 'original_path': file_path, 'image_data': None, 'skipped': True})
                        else:
                            files_to_process.append(file_path)
                    except Exception as e:
                        # If check fails, assume it needs processing
                        self.logger.error(f"检查文件时出错 {file_path}: {e}")
                        files_to_process.append(file_path)
                
                if skipped_files:
                    skip_msg = f"⏭️ 已跳过 {len(skipped_files)} 个已存在的文件"
                    self.log_received.emit(skip_msg)
                    self.log_received.emit("--- ℹ️ 跳过的文件将不会被处理，如需重新翻译请启用「覆盖已存在文件」选项")
                    self.logger.info(f"已跳过 {len(skipped_files)} 个已存在的文件（覆盖检测已禁用）")
                    # Update files list to only include those needing processing
                    self.files = files_to_process
                else:
                    self.log_received.emit("--- ✅ 未发现已存在的文件，将处理所有文件")
                    self.logger.info("未发现已存在的文件，将处理所有文件")
            
            # Update total count for progress bar logic
            total_original_count = len(original_files)
            skipped_existing_count = len(skipped_files)
            
            # 确定翻译流程模式
            workflow_mode = self._t("Normal Translation")
            workflow_tip = ""
            cli_config = self.config_dict.get('cli', {})
            if cli_config.get('upscale_only', False):
                workflow_mode = self._t("Upscale Only")
                workflow_tip = self._t("💡 Tip: Only upscale images, no detection, OCR, translation or rendering")
            elif cli_config.get('colorize_only', False):
                workflow_mode = self._t("Colorize Only")
                workflow_tip = self._t("💡 Tip: Only colorize images, no detection, OCR, translation or rendering")
            elif cli_config.get('generate_and_export', False):
                workflow_mode = self._t("Export Translation")
                workflow_tip = self._t("💡 Tip: After exporting, check manga_translator_work/translations/ for imagename_translated.txt files")
            elif cli_config.get('template', False):
                workflow_mode = self._t("Export Original Text")
                workflow_tip = self._t("💡 Tip: After exporting, manually translate imagename_original.txt in manga_translator_work/originals/, then use 'Import Translation and Render' mode")
            elif cli_config.get('load_text', False):
                workflow_mode = self._t("Import Translation and Render")
                workflow_tip = self._t("💡 Tip: Will read TXT files from manga_translator_work/originals/ or translations/ and render (prioritize _original.txt)")
                
                # TXT导入JSON的预处理已经统一到翻译器入口（manga_translator.py），这里不再需要

            # 检查是否启用并发模式
            batch_concurrent = self.config_dict.get('cli', {}).get('batch_concurrent', False)
            
            if is_hq or (len(self.files) > 0 and batch_size > 1):
                self.log_received.emit(f"--- 开始批量处理 ({'高质量模式' if is_hq else '批量模式'})")

                # 输出批量处理信息
                # total_images is the number of files to process
                total_images = len(self.files)
                
                # 如果启用并发模式，不分批加载（并发流水线内部会按需加载）
                if batch_concurrent:
                    self.log_received.emit(self._t("📊 Concurrent pipeline mode: {total} images (Total: {orig})", total=total_images, orig=total_original_count))
                    self.log_received.emit(self._t("🔧 Translation workflow: {mode}", mode=workflow_mode))
                    self.log_received.emit(self._t("📁 Output directory: {dir}", dir=self.output_folder))
                    if workflow_tip:
                        self.log_received.emit(workflow_tip)
                    self.log_received.emit(self._t("🚀 Starting translation..."))
                    
                    # 初始化进度条 (start from skipped_existing_count)
                    self.progress.emit(skipped_existing_count, total_original_count, "")
                    
                    if total_images > 0:
                        # 并发模式：直接传递所有文件路径，不预加载图片
                        images_with_configs = [(file_path, config) for file_path in self.files]
                        
                        # 调用翻译（并发流水线会自动处理）
                        all_contexts = await translator.translate_batch(
                            images_with_configs,
                            save_info=save_info,
                            global_offset=skipped_existing_count,
                            global_total=total_original_count
                        )
                    else:
                        all_contexts = []
                else:
                    # 非并发模式：前端分批加载（用于内存管理）
                    frontend_batch_size = 10  # 每次最多加载10张图片到内存
                    # Calculate batches based on remaining files
                    total_frontend_batches = (total_images + frontend_batch_size - 1) // frontend_batch_size if total_images > 0 else 0
                    
                    # 计算后端总批次数（用于显示统一的进度）
                    # Note: This is an estimation for logging purposes
                    backend_total_batches = (total_images + batch_size - 1) // batch_size if batch_size > 0 else total_images
                    
                    # 显示批量处理信息
                    if skipped_existing_count > 0:
                        self.log_received.emit(self._t("📊 Batch processing mode: {total} images in {batches} batches", total=total_images, batches=backend_total_batches))
                        self.log_received.emit(f"--- ℹ️ 另有 {skipped_existing_count} 个文件已跳过（原始总数：{total_original_count}）")
                    else:
                        self.log_received.emit(self._t("📊 Batch processing mode: {total} images in {batches} batches", total=total_images, batches=backend_total_batches))
                    
                    self.log_received.emit(self._t("🔧 Translation workflow: {mode}", mode=workflow_mode))
                    self.log_received.emit(self._t("📁 Output directory: {dir}", dir=self.output_folder))
                    if workflow_tip:
                        self.log_received.emit(workflow_tip)

                    # 按批次加载和处理图片（节省内存）
                    self.log_received.emit(self._t("🚀 Starting translation..."))
                    
                    # 初始化进度条
                    self.progress.emit(skipped_existing_count, total_original_count, "")
                    
                    all_contexts = []
                    processed_images_count = skipped_existing_count  # Start count from skipped files
                    
                    for frontend_batch_num in range(total_frontend_batches):
                        if not self._is_running: raise asyncio.CancelledError("Task stopped by user.")
                        
                        batch_start = frontend_batch_num * frontend_batch_size
                        batch_end = min(batch_start + frontend_batch_size, total_images)
                        current_batch_files = self.files[batch_start:batch_end]
                        
                        # 加载当前批次的图片（静默加载，不显示前端批次信息）
                        images_with_configs = []
                        for file_path in current_batch_files:
                            if not self._is_running: raise asyncio.CancelledError("Task stopped by user.")
                            try:
                                # 使用二进制模式读取以避免Windows路径编码问题
                                with open(file_path, 'rb') as f:
                                    image = Image.open(f)
                                    image.load()  # 立即加载图片数据，避免文件句柄关闭后无法访问
                                image.name = file_path
                                images_with_configs.append((image, config))
                            except Exception as e:
                                self.log_received.emit(f"⚠️ 无法加载图片 {os.path.basename(file_path)}: {e}")
                                self.logger.error(f"Error loading image {file_path}: {e}")
                                # 创建错误上下文
                                from manga_translator.utils import Context
                                error_ctx = Context()
                                error_ctx.image_name = file_path
                                error_ctx.translation_error = str(e)
                                all_contexts.append(error_ctx)
                        
                        if images_with_configs:
                            # 传递全局偏移量给后端，让后端显示正确的全局图片编号
                            batch_contexts = await translator.translate_batch(
                                images_with_configs, 
                                save_info=save_info,
                                global_offset=processed_images_count,  # 传递已处理的图片数
                                global_total=total_original_count  # 传递总图片数
                            )
                            all_contexts.extend(batch_contexts)
                            processed_images_count += len(images_with_configs)
                            
                            # 批次处理完成后，立即清理图片对象
                            for image, _ in images_with_configs:
                                if hasattr(image, 'close'):
                                    try:
                                        image.close()
                                    except:
                                        pass
                            images_with_configs.clear()
                            
                            # 强制垃圾回收
                            import gc
                            gc.collect()
                    
                    # 非并发模式：所有批次处理完成
                
                # 并发模式和非并发模式都会到这里
                contexts = all_contexts
                
                # 🔍 调试：检查并发模式返回的contexts
                self.log_received.emit(f"📋 [调试] contexts数量: {len(contexts)}")
                for i, ctx in enumerate(contexts[:5]):  # 只显示前5个
                    has_error = hasattr(ctx, 'translation_error') and ctx.translation_error
                    has_success = hasattr(ctx, 'success') and ctx.success
                    self.log_received.emit(f"📋 [调试] ctx[{i}]: image_name={getattr(ctx, 'image_name', 'N/A')}, translation_error={has_error}, success={has_success}")

                # The backend now handles saving for batch jobs. We just need to collect the paths/status.
                processed_success_count = 0
                processed_failed_count = 0
                skipped_backend_count = 0
                for ctx in contexts:
                    # ✅ 不在结果收集阶段检查取消状态
                    # process_batch 已经处理了取消逻辑，这里只需要收集所有结果
                    if ctx:
                        if hasattr(ctx, 'skipped') and ctx.skipped:
                            skipped_backend_count += 1
                        # 检查是否有翻译错误
                        if hasattr(ctx, 'translation_error') and ctx.translation_error:
                            results.append({'success': False, 'original_path': ctx.image_name, 'error': ctx.translation_error})
                            processed_failed_count += 1
                            # 输出详细的错误信息（包含原始错误）
                            self.log_received.emit(f"\n⚠️ 图片 {os.path.basename(ctx.image_name)} 翻译失败：")
                            self.log_received.emit(ctx.translation_error)
                        elif hasattr(ctx, 'success') and ctx.success:
                            # 优先检查success标志（因为result可能被清理了）
                            results.append({'success': True, 'original_path': ctx.image_name, 'image_data': None})
                            processed_success_count += 1
                        elif ctx.result:
                            results.append({'success': True, 'original_path': ctx.image_name, 'image_data': None})
                            processed_success_count += 1
                        else:
                            results.append({'success': False, 'original_path': ctx.image_name, 'error': '翻译结果为空'})
                            processed_failed_count += 1
                    else:
                        results.append({'success': False, 'original_path': 'Unknown', 'error': 'Batch translation returned no context'})
                        processed_failed_count += 1

                # Summary fields are used by UI for consistent counting.
                # Keep legacy keys (success/failed/skipped) for compatibility.
                self.results_summary = {
                    'total': total_original_count,
                    'to_process': total_images,
                    'success': processed_success_count,
                    'failed': processed_failed_count,
                    'skipped': skipped_existing_count,
                    'skipped_existing': skipped_existing_count,
                    'skipped_backend': skipped_backend_count,
                }

                if processed_failed_count > 0:
                    self.log_received.emit(self._t("\n⚠️ Batch translation completed: {success}/{total} succeeded, {failed}/{total} failed", success=processed_success_count, total=total_images, failed=processed_failed_count))
                    # 🔍 调试：检查results中的失败项
                    self.log_received.emit(f"📋 [调试] 即将调用_emit_failed_files_record，results数量: {len(results)}")
                    failed_results = [r for r in results if not r.get('success', True)]
                    self.log_received.emit(f"📋 [调试] 失败项数量: {len(failed_results)}")
                    for r in failed_results[:3]:  # 只显示前3个
                        self.log_received.emit(f"📋 [调试] 失败项: success={r.get('success')}, original_path={r.get('original_path', 'N/A')[:50]}")
                    # 收集失败文件信息并发射信号
                    self._emit_failed_files_record(results)
                else:
                    self.log_received.emit(self._t("✅ Batch translation completed: {success}/{total} succeeded", success=processed_success_count, total=total_images))
                self.log_received.emit(self._t("💾 Files saved to: {dir}", dir=self.output_folder))

            else:
                self.log_received.emit("--- 开始顺序处理...")
                total_files = len(self.files)

                # 输出顺序处理信息
                self.log_received.emit(self._t("📊 Sequential processing mode: {total} images (Total: {orig})", total=total_files, orig=total_original_count))
                self.log_received.emit(self._t("🔧 Translation workflow: {mode}", mode=workflow_mode))
                self.log_received.emit(self._t("📁 Output directory: {dir}", dir=self.output_folder))
                if workflow_tip:
                    self.log_received.emit(workflow_tip)

                # 初始化进度条
                self.progress.emit(skipped_existing_count, total_original_count, "")

                success_count = 0
                # ✅ 顺序模式也需要把失败信息发射给UI（用于“失败记录”列表）
                sequential_failed_results = []  # [{'success': False, 'original_path': str, 'error': str}, ...]
                failed_record_emitted = False

                try:
                    for i, file_path in enumerate(self.files):
                        if not self._is_running:
                            raise asyncio.CancelledError("Task stopped by user.")

                        current_num = skipped_existing_count + i + 1
                        # 更新进度条（显示图片数量）
                        self.progress.emit(current_num, total_original_count, "")
                        self.log_received.emit(f"🔄 [{current_num}/{total_original_count}] 正在处理：{os.path.basename(file_path)}")

                        try:
                            # 使用二进制模式读取以避免Windows路径编码问题
                            with open(file_path, 'rb') as f:
                                image = Image.open(f)
                                image.load()  # 立即加载图片数据，避免文件句柄关闭后无法访问
                            image.name = file_path

                            ctx = await translator.translate(image, config, image_name=image.name, save_info=save_info)

                            if ctx and ctx.result:
                                self.file_processed.emit({'success': True, 'original_path': file_path, 'image_data': ctx.result})
                                success_count += 1
                                self.log_received.emit(f"✅ [{current_num}/{total_files}] 完成：{os.path.basename(file_path)}")
                            else:
                                err_msg = 'Translation returned no result or image'
                                # 尝试使用后端返回的错误信息（如果存在）
                                if ctx and hasattr(ctx, 'translation_error') and ctx.translation_error:
                                    err_msg = str(ctx.translation_error)
                                self.file_processed.emit({'success': False, 'original_path': file_path, 'error': err_msg})
                                sequential_failed_results.append({'success': False, 'original_path': file_path, 'error': err_msg})
                                self.log_received.emit(f"❌ [{current_num}/{total_files}] 失败：{os.path.basename(file_path)}")

                        except Exception as e:
                            self.log_received.emit(f"❌ [{current_num}/{total_files}] 错误：{os.path.basename(file_path)} - {e}")
                            self.file_processed.emit({'success': False, 'original_path': file_path, 'error': str(e)})
                            sequential_failed_results.append({'success': False, 'original_path': file_path, 'error': str(e)})

                            # ✅ 在抛出异常前，先把失败记录发给UI（避免列表为空）
                            if sequential_failed_results and not failed_record_emitted:
                                self._emit_failed_files_record(sequential_failed_results)
                                failed_record_emitted = True

                            # 抛出异常，终止整个翻译流程
                            raise

                finally:
                    # ✅ 正常结束时，统一发射一次失败记录
                    if sequential_failed_results and not failed_record_emitted:
                        self._emit_failed_files_record(sequential_failed_results)
                        failed_record_emitted = True

                self.log_received.emit(f"✅ 顺序翻译完成：成功 {success_count}/{total_files} 张")
                self.log_received.emit(f"💾 文件已保存到：{self.output_folder}")
            
            if self.results_summary is not None:
                results = list(results)
                results.append({'success': True, 'original_path': '', 'image_data': None, 'results_summary': self.results_summary})
            self.finished.emit(results)
            self._finished_emitted = True

        except asyncio.CancelledError as e:
            self.log_received.emit(f"Task cancelled: {e}")
            self.logger.warning(f"Task cancelled: {e}")
            
            # ✅ 取消时也收集已处理的结果
            if results:
                success_count = sum(1 for r in results if r.get('success', False))
                failed_count = len(results) - success_count
                self.log_received.emit(f"❌ 任务已取消，已处理: {success_count} 成功, {failed_count} 失败")
                
                self.results_summary = {
                    'success': success_count,
                    'failed': failed_count,
                    'skipped': 0,
                    'cancelled': True
                }
                
                # 发射已处理的结果
                results = list(results)
                results.append({'success': True, 'original_path': '', 'image_data': None, 'results_summary': self.results_summary})
                self.finished.emit(results)
                self._finished_emitted = True
            else:
                self.error.emit("任务已取消")
        except Exception as e:
            import traceback
            error_message = str(e)
            error_traceback = traceback.format_exc()
            
            # 记录到logger，确保命令行能看到
            self.logger.error(f"Translation error: {error_message}")
            self.logger.error(error_traceback)
            
            # 构建友好的中文错误提示
            friendly_error = self._build_friendly_error_message(error_message, error_traceback)
            
            self.log_received.emit(friendly_error)
            self.error.emit(friendly_error)
        finally:
            manga_logger.removeHandler(log_handler)

            # 翻译结束后进行完整的内存清理（特别是CPU模式）
            try:
                # 清理翻译器对象引用
                if 'translator' in locals():
                    del translator
                
                from desktop_qt_ui.utils.memory_cleanup import full_memory_cleanup
                full_memory_cleanup(log_callback=lambda msg: self.log_received.emit(msg))
            except Exception as e:
                self.log_received.emit(f"--- [CLEANUP] Warning: 内存清理时出错: {e}")

    @pyqtSlot()
    def process(self):
        loop = None
        try:
            import asyncio
            import sys
            self.log_received.emit("--- 开始处理任务...")

            # 在Windows上的工作线程中，需要手动初始化Windows Socket
            if sys.platform == 'win32':
                # 使用ctypes直接调用WSAStartup
                import ctypes
                
                try:
                    # WSADATA结构体大小
                    WSADATA_SIZE = 400
                    wsa_data = ctypes.create_string_buffer(WSADATA_SIZE)
                    # 调用WSAStartup，版本2.2
                    ws2_32 = ctypes.WinDLL('ws2_32')
                    result = ws2_32.WSAStartup(0x0202, wsa_data)
                    if result != 0:
                        self.log_received.emit(f"--- [ERROR] WSAStartup failed with code {result}")
                except Exception as e:
                    self.log_received.emit(f"--- [ERROR] Failed to initialize WSA: {e}")
                
                # 使用ProactorEventLoop（Windows默认）
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            # 创建事件循环并保存任务引用
            try:
                loop = asyncio.new_event_loop()
            except Exception as e:
                self.log_received.emit(f"--- [ERROR] Failed to create event loop: {e}")
                import traceback
                self.log_received.emit(f"--- [ERROR] Traceback: {traceback.format_exc()}")
                raise
            
            asyncio.set_event_loop(loop)
            
            self._current_task = loop.create_task(self._do_processing())
            loop.run_until_complete(self._current_task)
            # 任务处理完成，不输出日志

        except asyncio.CancelledError:
            self.log_received.emit("--- 任务已取消")
            if not self._finished_emitted:
                try:
                    results = []
                    if self.results_summary is not None:
                        results.append({'success': True, 'original_path': '', 'image_data': None, 'results_summary': self.results_summary})
                    self.finished.emit(results)
                    self._finished_emitted = True
                    self.log_received.emit("--- [FINISH] finished信号已发射（取消）")
                except Exception as e:
                    self.log_received.emit(f"--- [ERROR] Failed to emit finished signal (cancelled): {e}")
        except Exception as e:
            import traceback
            error_msg = f"An error occurred in the asyncio runner: {str(e)}\n{traceback.format_exc()}"
            # 同时记录到logger，确保命令行能看到
            self.logger.error(error_msg)
            self.error.emit(error_msg)
            if not self._finished_emitted:
                try:
                    results = []
                    if self.results_summary is not None:
                        results.append({'success': True, 'original_path': '', 'image_data': None, 'results_summary': self.results_summary})
                    self.finished.emit(results)
                    self._finished_emitted = True
                    self.log_received.emit("--- [FINISH] finished信号已发射（异常）")
                except Exception as e2:
                    self.log_received.emit(f"--- [ERROR] Failed to emit finished signal (exception): {e2}")
        finally:
            if not self._finished_emitted:
                try:
                    results = []
                    if self.results_summary is not None:
                        results.append({'success': True, 'original_path': '', 'image_data': None, 'results_summary': self.results_summary})
                    self.finished.emit(results)
                    self._finished_emitted = True
                    self.log_received.emit("--- [FINISH] finished信号已兜底发射")
                except Exception as e:
                    self.log_received.emit(f"--- [ERROR] Failed to emit finished signal: {e}")
            if loop:
                try:
                    # Cancel all remaining tasks
                    tasks = asyncio.all_tasks(loop=loop)
                    for task in tasks:
                        task.cancel()
                    
                    # Gather all tasks to let them finish cancelling
                    if tasks:
                        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

                    # Shutdown async generators
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception as e:
                    self.log_received.emit(f"--- ERROR during asyncio cleanup: {e}")
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
                    # 清理完成，不输出日志
