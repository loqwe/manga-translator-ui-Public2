# YOLO OBB 修复和广告过滤优化

## 问题 1：YOLO OBB 检测错误

### 问题描述
```
2025-12-21 14:55:43,062 - ERROR - [manga-translator.DefaultDetector] - YOLO OBB辅助检测失败: 
OpenCV(4.12.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4208: 
error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'
```

### 根本原因
YOLO OBB 检测器不生成 mask 图像，返回 `(textlines, None, None)`。但当图像尺寸过小需要添加 border 时，`CommonDetector._remove_border()` 函数会尝试对 `raw_mask` 执行 `cv2.resize()`，但没有检查 `raw_mask` 是否为 `None`。

**错误链路**：
1. 主检测器检测完成，返回结果
2. `dispatch()` 调用 `yolo_detector.detect(image, ...)` 进行 YOLO OBB 辅助检测
3. `detect()` 中检测到图像尺寸过小，添加 border：`add_border = min(img_w, img_h) < 400`
4. YOLO OBB `_infer()` 成功检测，返回 `(textlines, None, None)`
5. `_remove_border()` 被调用，尝试 `cv2.resize(None, ...)` → **错误！**

**特征**：
- 错误消息来自 `[manga-translator.DefaultDetector]`，而非 YOLO OBB 内部
- 只在处理小尺寸图像时发生（< 400px）
- YOLO OBB 本身检测成功，但在后处理时崩溃

### 修复方案

#### 修复 1：修复 CommonDetector._remove_border 对 None mask 的处理

**修改文件**: `manga_translator/detection/common.py`

**修改位置**: `_remove_border` 函数（第 81-88 行）

**修改内容**:
在对 `raw_mask` 执行 `cv2.resize()` 之前，添加 None 检查。

```python
# 修改前
def _remove_border(self, image: np.ndarray, old_w: int, old_h: int, textlines: List[Quadrilateral], raw_mask, mask):
    new_h, new_w = image.shape[:2]
    raw_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # ❗ 错误：raw_mask 可能为 None
    raw_mask = raw_mask[:old_h, :old_w]
    if mask is not None:
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = mask[:old_h, :old_w]

# 修改后
def _remove_border(self, image: np.ndarray, old_w: int, old_h: int, textlines: List[Quadrilateral], raw_mask, mask):
    new_h, new_w = image.shape[:2]
    
    # ✅ 检查 raw_mask 是否为 None（某些检测器不生成 mask）
    if raw_mask is not None:
        raw_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        raw_mask = raw_mask[:old_h, :old_w]
    
    if mask is not None:
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = mask[:old_h, :old_w]
```

#### 修复 2：增强 YOLO OBB letterbox 的鲁棒性（预防性修复）

**修改文件**: `manga_translator/detection/yolo_obb.py`

**修改位置**: `letterbox` 函数（第 62-88 行）

**修改内容**:
添加输入验证和尺寸保护，确保 resize 的目标尺寸至少为 1x1。

```python
# 修改前
new_unpad_w = int(round(shape[1] * gain))
new_unpad_h = int(round(shape[0] * gain))

# 修改后
# 检查输入图像尺寸是否有效
if shape[0] <= 0 or shape[1] <= 0:
    self.logger.error(f"YOLO OBB letterbox: 输入图像尺寸无效 {shape}")
    raise ValueError(f"Invalid image shape: {shape}")

# 计算新的未填充尺寸，确保至少为1
new_unpad_w = max(1, int(round(shape[1] * gain)))
new_unpad_h = max(1, int(round(shape[0] * gain)))
```

### 验证
修复后的代码确保：
- ✅ YOLO OBB 可以正常处理小尺寸图像（< 400px）
- ✅ 不生成 mask 的检测器不会导致 _remove_border 崩溃
- ✅ 增强了 YOLO OBB letterbox 的鲁棒性，防止未来类似问题

---

## 问题 2：广告水印过滤优化

### 需求
从日志中可以看到翻译结果包含大量韩文广告水印文本：
- `저작권법에 의해 법적 조치에 처해집니다` (根据著作权法将采取法律措施)
- `복제, 변경, 번역, 출판, 방송 및 기타의 방법으로 이용할` (以复制、修改、翻译、出版、广播及其他方式使用)
- `是7468` → `最快的漫画提供网站 漫画王国 Newtoki 468`
- `HTTPS://NEWTOK1468.CO`
- `外可智是不人` → `最快的漫画提供网站`

### 解决方案

**修改文件**: `examples/config/watermark_filter.json`

**新增过滤规则**:

```json
{
  "partial_match_patterns": [
    // 原有规则...
    
    // 新增：Newtoki 系列广告
    "newtok",                    // 匹配 newtoki, newtok1468 等
    "https://newtok",            // 匹配完整 URL
    
    // 新增：韩文版权警告
    "저작권법에 의해 법적 조치에 처해집니다",
    "복제, 변경, 번역, 출판, 방송 및 기타의 방법으로 이용할",
    
    // 新增：韩文广告常用语
    "최빠 만화 제공 사이트",      // 最快的漫画提供网站
    "만화왕국"                    // 漫画王国
  ]
}
```

### 过滤效果

启用 `enable_watermark_filter: true` 后，以下文本将被自动过滤：

| 原文 | 翻译 | 状态 |
|------|------|------|
| `복제, 변경, 번역, 출판, 방송 및 기타의 방법으로 이용할 저작권법에 의해 법적 조치에 처해집니다` | ~~以复制、修改、翻译、出版、广播及其他方式使用时，将根据著作权法采取法律措施。~~ | ✅ 过滤 |
| `是7468` | ~~最快的漫画提供网站 漫画王国 Newtoki 468~~ | ✅ 过滤 |
| `HTTPS://NEWTOK1468.CO` | ~~HTTPS://NEWTOK1468.CO~~ | ✅ 过滤 |
| `外可智是不人` | ~~最快的漫画提供网站~~ | ✅ 过滤 |
| `저기…` | 那个…… | ⭕ 保留（正常对话）|
| `얘기가좀 길어지시는것같아서` | 我看你们好像聊得有点久了 | ⭕ 保留（正常对话）|

### 使用方法

1. **启用过滤功能**（在配置文件中）:
```json
{
  "enable_watermark_filter": true
}
```

2. **查看过滤列表**:
   - 主界面 → 设置 → "打开过滤列表" 按钮

3. **自定义规则**:
   - 编辑 `examples/config/watermark_filter.json`
   - 支持部分匹配、精确匹配、正则表达式

---

## 测试建议

### YOLO OBB 修复测试
```bash
# 运行程序，观察日志中是否还有 cv::resize 错误
# 测试图片：包含极小文本区域的漫画页面
```

### 广告过滤测试
```bash
# 1. 确认配置文件中启用了过滤
cat examples/config.json | grep enable_watermark_filter

# 2. 翻译包含广告的漫画
# 3. 检查翻译结果中是否还有广告水印文本
```

---

## 技术细节

### YOLO OBB letterbox 函数流程
1. 计算缩放比例：`gain = min(target_h/img_h, target_w/img_w)`
2. 计算缩放后尺寸：`new_h = int(round(img_h * gain))`
3. **新增**：确保最小尺寸：`new_h = max(1, new_h)`
4. 执行 resize：`cv2.resize(img, (new_w, new_h))`
5. 添加 padding 到目标尺寸

### 水印过滤实现位置
- 配置文件：`examples/config/watermark_filter.json`
- 后端实现：`manga_translator.py` (lines 1723-1778)
- 支持的匹配模式：
  - 子串匹配（不区分大小写）
  - 精确匹配
  - 正则表达式（带 `regex: true` 标记）

---

## 相关文件

### 修改的文件
- `manga_translator/detection/common.py` - 修复 _remove_border 对 None mask 的处理
- `manga_translator/detection/yolo_obb.py` - 增强 YOLO OBB letterbox 鲁棒性
- `examples/config/watermark_filter.json` - 广告过滤规则更新

### 配置文件
- `examples/config.json` - 启用/禁用水印过滤
- `desktop_qt_ui/core/config_models.py` - 配置模型定义

---

## 更新日志

### 2025-12-21
- ✅ **核心修复**: 修复 CommonDetector._remove_border 对 None mask 的错误处理
  - 解决 YOLO OBB 辅助检测在小尺寸图像时的崩溃问题
  - 修复了所有不生成 mask 的检测器在使用 border 时的问题
- ✅ **预防性修复**: 增强 YOLO OBB letterbox 函数的鲁棒性
  - 添加输入验证和尺寸保护
  - 确保 resize 目标尺寸至少为 1x1
- ✅ 添加韩文广告水印过滤规则（Newtoki 系列）
- ✅ 添加版权警告文本过滤规则
- ✅ 优化广告语过滤覆盖范围
