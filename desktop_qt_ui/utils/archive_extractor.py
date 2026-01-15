"""
压缩包/文档格式图片提取工具
支持 PDF、EPUB、CBZ 格式
"""
import os
import tempfile
import zipfile
import shutil
from typing import List, Optional, Tuple
from pathlib import Path


# 支持的压缩包/文档格式
ARCHIVE_EXTENSIONS = {'.pdf', '.epub', '.cbz', '.cbr', '.cb7', '.zip'}

# 支持的图片格式
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif', '.gif', '.tiff', '.tif'}


def is_archive_file(file_path: str) -> bool:
    """检查文件是否是支持的压缩包/文档格式"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ARCHIVE_EXTENSIONS


def get_temp_extract_dir(archive_path: str) -> str:
    """获取压缩包的临时解压目录"""
    # 使用系统临时目录下的固定子目录，便于管理
    base_temp = os.path.join(tempfile.gettempdir(), 'manga_translator_archives')
    os.makedirs(base_temp, exist_ok=True)
    
    # 使用文件名和修改时间生成唯一目录名
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    mtime = int(os.path.getmtime(archive_path)) if os.path.exists(archive_path) else 0
    unique_name = f"{archive_name}_{mtime}"
    
    return os.path.join(base_temp, unique_name)


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """从 PDF 文件中提取图片"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")
    
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 将页面渲染为图片
        # 使用较高的分辨率以保证质量
        mat = fitz.Matrix(2.0, 2.0)  # 2x 缩放
        pix = page.get_pixmap(matrix=mat)
        
        image_path = os.path.join(output_dir, f"page_{page_num + 1:04d}.png")
        pix.save(image_path)
        extracted_images.append(image_path)
    
    doc.close()
    return sorted(extracted_images)


def extract_images_from_epub(epub_path: str, output_dir: str) -> List[str]:
    """从 EPUB 文件中提取图片"""
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    
    with zipfile.ZipFile(epub_path, 'r') as zf:
        for file_info in zf.infolist():
            ext = os.path.splitext(file_info.filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                # 提取图片，保持相对路径结构
                # 但简化文件名以避免路径过长
                base_name = os.path.basename(file_info.filename)
                # 添加序号前缀以保持顺序
                idx = len(extracted_images)
                new_name = f"{idx:04d}_{base_name}"
                output_path = os.path.join(output_dir, new_name)
                
                with zf.open(file_info) as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                extracted_images.append(output_path)
    
    return sorted(extracted_images)


def extract_images_from_cbz(cbz_path: str, output_dir: str) -> List[str]:
    """从 CBZ (Comic Book ZIP) 文件中提取图片"""
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    
    with zipfile.ZipFile(cbz_path, 'r') as zf:
        # 获取所有图片文件并排序
        image_files = []
        for file_info in zf.infolist():
            if file_info.is_dir():
                continue
            ext = os.path.splitext(file_info.filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(file_info)
        
        # 按文件名自然排序
        image_files.sort(key=lambda x: natural_sort_key(x.filename))
        
        for idx, file_info in enumerate(image_files):
            base_name = os.path.basename(file_info.filename)
            # 添加序号前缀以保持顺序
            new_name = f"{idx:04d}_{base_name}"
            output_path = os.path.join(output_dir, new_name)
            
            with zf.open(file_info) as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            extracted_images.append(output_path)
    
    return extracted_images


def extract_images_from_cbr(cbr_path: str, output_dir: str) -> List[str]:
    """从 CBR (Comic Book RAR) 文件中提取图片"""
    try:
        import rarfile
    except ImportError:
        raise ImportError("需要安装 rarfile: pip install rarfile")
    
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    
    with rarfile.RarFile(cbr_path, 'r') as rf:
        image_files = []
        for file_info in rf.infolist():
            if file_info.is_dir():
                continue
            ext = os.path.splitext(file_info.filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(file_info)
        
        image_files.sort(key=lambda x: natural_sort_key(x.filename))
        
        for idx, file_info in enumerate(image_files):
            base_name = os.path.basename(file_info.filename)
            new_name = f"{idx:04d}_{base_name}"
            output_path = os.path.join(output_dir, new_name)
            
            with rf.open(file_info) as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            extracted_images.append(output_path)
    
    return extracted_images


def natural_sort_key(s: str):
    """自然排序键，支持数字排序"""
    import re
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def extract_images_from_archive(archive_path: str, output_dir: Optional[str] = None) -> Tuple[List[str], str]:
    """
    从压缩包/文档中提取图片
    
    Args:
        archive_path: 压缩包/文档路径
        output_dir: 输出目录，如果为 None 则使用临时目录
    
    Returns:
        (提取的图片路径列表, 输出目录)
    """
    if output_dir is None:
        output_dir = get_temp_extract_dir(archive_path)
    
    # 如果目录已存在且有文件，直接返回缓存的结果
    if os.path.exists(output_dir):
        existing_images = []
        for f in os.listdir(output_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                existing_images.append(os.path.join(output_dir, f))
        if existing_images:
            return sorted(existing_images), output_dir
    
    ext = os.path.splitext(archive_path)[1].lower()
    
    if ext == '.pdf':
        images = extract_images_from_pdf(archive_path, output_dir)
    elif ext == '.epub':
        images = extract_images_from_epub(archive_path, output_dir)
    elif ext in {'.cbz', '.zip'}:
        images = extract_images_from_cbz(archive_path, output_dir)
    elif ext == '.cbr':
        images = extract_images_from_cbr(archive_path, output_dir)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    return images, output_dir


def cleanup_temp_archives():
    """清理所有临时解压目录"""
    base_temp = os.path.join(tempfile.gettempdir(), 'manga_translator_archives')
    if os.path.exists(base_temp):
        shutil.rmtree(base_temp, ignore_errors=True)


def cleanup_archive_temp(archive_path: str):
    """清理指定压缩包的临时解压目录"""
    temp_dir = get_temp_extract_dir(archive_path)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
