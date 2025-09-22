# 直播流帧捕获工具 (useffmpeg.py)

快速获取直播流帧并进行OCR分析的工具

## 功能特性

- 🎬 从直播流快速获取指定数量的帧
- 📸 自动保存原图和裁剪后的右上角区域
- 🔤 OCR识别文本内容
- 🕒 自动检测时间信息
- 📺 自动检测重播标识
- 📁 按时间戳组织输出文件

## 使用方法

### 1. 测试模式（使用本地图片）
```bash
uv run python app/useffmpeg.py test
```

### 2. 直播流模式
```bash
# 基本用法
uv run python app/useffmpeg.py "https://example.com/live/stream.m3u8"

# 获取5帧
uv run python app/useffmpeg.py "https://example.com/live/stream.m3u8" -n 5

# 只获取帧，不进行OCR分析
uv run python app/useffmpeg.py "https://example.com/live/stream.m3u8" --no-ocr

# 自定义输出目录
uv run python app/useffmpeg.py "https://example.com/live/stream.m3u8" -o "my_output"

# 自定义缩放宽度
uv run python app/useffmpeg.py "https://example.com/live/stream.m3u8" -w 1920
```

### 3. 参数说明
- `stream_url`: 直播流URL (HLS/DASH格式)
- `-n, --frames`: 获取帧数 (默认: 2)
- `-w, --width`: 缩放宽度 (默认: 1280)
- `--no-ocr`: 不进行OCR分析
- `-o, --output`: 输出目录 (默认: app/output)

## 输出文件结构

```
app/output/
└── 20250922_183920/          # 时间戳目录
    ├── frame_01_1758537564.jpg      # 原始帧
    ├── frame_01_cropped_1758537564.jpg  # 右上角裁剪
    ├── frame_02_1758537564.jpg      # 原始帧
    └── frame_02_cropped_1758537564.jpg  # 右上角裁剪
```

## 分析结果

工具会自动分析每一帧：
- ✅ **文本识别**: 使用PaddleOCR识别图像中的文字
- 🕒 **时间检测**: 自动检测包含时间格式的文本 (如 14:30:25)
- 📺 **重播检测**: 自动检测重播相关关键词 (如"播"、"重播"、"REPLAY")
- ⏱️ **性能统计**: 显示OCR处理时间和置信度

## 示例输出

```
🧪 测试模式 - 使用本地图片
==================================================
📁 输出目录: app\output\20250922_183920
🔤 初始化OCR引擎...
✅ OCR引擎初始化完成

🖼️  处理图片 1: 3.png
   原始尺寸: (1280, 720), 模式: RGBA
   🔄 转换为RGB模式
   💾 已保存: test_frame_01_1758537564.jpg
   ✂️  裁剪尺寸: (320, 180)
   💾 裁剪已保存: test_frame_01_cropped_1758537564.jpg
   📝 识别到 8 行文本
      1. 乘播
      2. 周决赛第2天]
      3. 秋
      4. 季
      5. 赛
   🕒 包含时间: False
   📺 包含重播标识: True
   ⏱️  OCR耗时: 315ms

📋 测试总结:
   输出目录: app\output\20250922_183920
   处理图片: 2
   📺 包含重播标识的图片: 2
```

## 依赖要求

- FFmpeg (用于流媒体处理)
- PaddleOCR (OCR识别引擎)
- PIL/Pillow (图像处理)
- numpy (数组处理)

## 常见问题

### Q: 无法连接到直播流
A: 检查流媒体URL是否正确，网络连接是否正常

### Q: OCR识别不准确
A: 可以调整裁剪比例或图像缩放参数来改善识别效果

### Q: 处理速度慢
A: 使用 `--no-ocr` 参数跳过OCR分析，或减少获取帧数

## 技术原理

1. **流媒体处理**: 使用FFmpeg从HLS/DASH流中提取JPEG帧
2. **图像处理**: 自动裁剪右上角区域用于时间检测
3. **OCR识别**: 使用PaddleOCR进行文字识别
4. **智能分析**: 基于关键词和模式匹配进行内容分析
