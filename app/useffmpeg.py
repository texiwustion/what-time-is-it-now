#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速获取直播流帧的工具
用法: python app/useffmpeg.py <stream_url> [frame_count]
"""

import subprocess
import time
import io
import sys
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import argparse

# 添加上级目录到Python路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from paddle_ocr import PaddleOCREngine
from ocr import OCRResult
from ffmpeg_玩法 import FFmpeg玩法工厂, 直播流帧捕获玩法


class StreamFrameCapture:
    """直播流帧捕获工具"""
    
    def __init__(self, stream_url: str, output_dir: str = "app/output"):
        self.stream_url = stream_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / timestamp
        self.session_dir.mkdir(exist_ok=True)
        
        # 初始化FFmpeg玩法
        self.ffmpeg_玩法 = FFmpeg玩法工厂.create_直播流帧捕获(stream_url)
        
        print(f"📁 输出目录: {self.session_dir}")
    
    def capture_frames(self, frame_count: int = 2, scale_width: int = 1280) -> list[Path]:
        """捕获帧并保存为图片文件"""
        print(f"🎬 开始从流媒体获取 {frame_count} 帧...")
        print(f"📺 流媒体URL: {self.stream_url}")
        
        # 配置FFmpeg玩法
        self.ffmpeg_玩法.set_frame_count(frame_count)
        self.ffmpeg_玩法.set_scale(scale_width)
        self.ffmpeg_玩法.set_fps(1.0)
        
        cmd = self.ffmpeg_玩法.build_command()
        print(f"🔧 FFmpeg命令: {' '.join(cmd)}")
        
        try:
            saved_files = []
            frame_number = 0
            
            print("📡 正在接收数据流...")
            
            # 使用FFmpeg玩法捕获帧
            for jpeg_data in self.ffmpeg_玩法.capture_frames(frame_count):
                try:
                    # 转换为PIL图像
                    image = Image.open(io.BytesIO(jpeg_data))
                    frame_number += 1
                    
                    # 保存图片
                    filename = f"frame_{frame_number:02d}_{int(time.time())}.jpg"
                    filepath = self.session_dir / filename
                    image.save(filepath, 'JPEG', quality=95)
                    saved_files.append(filepath)
                    
                    print(f"✅ 第{frame_number}帧已保存: {filename} (尺寸: {image.size})")
                        
                except Exception as e:
                    print(f"⚠️  解析JPEG帧失败: {e}")
                    continue
            
            print(f"🎉 成功捕获 {len(saved_files)} 帧")
            return saved_files
            
        except Exception as e:
            print(f"❌ 捕获帧时出错: {e}")
            return []
    
    def capture_and_analyze(self, frame_count: int = 2) -> dict:
        """捕获帧并进行OCR分析"""
        # 捕获帧
        saved_files = self.capture_frames(frame_count)
        
        if not saved_files:
            return {"frames": [], "analysis": []}
        
        # 初始化OCR引擎
        print("\n🔤 初始化OCR引擎...")
        try:
            ocr_engine = PaddleOCREngine()
            print("✅ OCR引擎初始化完成")
        except Exception as e:
            print(f"❌ OCR引擎初始化失败: {e}")
            return {"frames": [str(f) for f in saved_files], "analysis": []}
        
        # 分析每一帧
        analysis_results = []
        for i, filepath in enumerate(saved_files):
            print(f"\n🔍 分析第{i+1}帧: {filepath.name}")
            
            try:
                # 加载图片
                image = Image.open(filepath)
                
                # 裁剪右上角（用于时间检测）
                width, height = image.size
                crop_ratio = 0.25
                crop_width = int(width * crop_ratio)
                crop_height = int(height * crop_ratio)
                
                left = width - crop_width
                top = 0
                right = width
                bottom = crop_height
                
                cropped = image.crop((left, top, right, bottom))
                
                # 保存裁剪后的图片
                cropped_filename = f"frame_{i+1:02d}_cropped_{int(time.time())}.jpg"
                cropped_filepath = self.session_dir / cropped_filename
                cropped.save(cropped_filepath, 'JPEG', quality=95)
                
                # OCR识别
                import numpy as np
                cropped_array = np.array(cropped.convert('RGB'))
                result = ocr_engine.infer(cropped_array)
                
                # 简单分析
                texts = [line.text for line in result.texts]
                has_time = any(':' in text and any(c.isdigit() for c in text) for text in texts)
                has_replay = any('播' in text or '重播' in text or 'REPLAY' in text.upper() for text in texts)
                
                frame_analysis = {
                    "frame_file": str(filepath),
                    "cropped_file": str(cropped_filepath),
                    "texts": texts,
                    "has_time": has_time,
                    "has_replay": has_replay,
                    "ocr_time_ms": result.time_ms,
                    "avg_confidence": result.avg_confidence
                }
                
                analysis_results.append(frame_analysis)
                
                print(f"   📝 识别到 {len(texts)} 行文本")
                if texts:
                    for j, text in enumerate(texts[:3]):  # 只显示前3行
                        print(f"      {j+1}. {text}")
                print(f"   🕒 包含时间: {has_time}")
                print(f"   📺 包含重播标识: {has_replay}")
                
            except Exception as e:
                print(f"   ❌ 分析失败: {e}")
                analysis_results.append({
                    "frame_file": str(filepath),
                    "error": str(e)
                })
        
        return {
            "session_dir": str(self.session_dir),
            "frames": [str(f) for f in saved_files],
            "analysis": analysis_results
        }


def test_local_images():
    """测试本地图片功能"""
    print("🧪 测试模式 - 使用本地图片")
    print("=" * 50)
    
    # 创建测试会话
    capturer = StreamFrameCapture("test://localhost", "app/output")
    
    # 测试图片列表
    test_images = ["3.png", "1.png"]
    
    # 初始化OCR引擎
    print("🔤 初始化OCR引擎...")
    try:
        ocr_engine = PaddleOCREngine()
        print("✅ OCR引擎初始化完成")
    except Exception as e:
        print(f"❌ OCR引擎初始化失败: {e}")
        return
    
    analysis_results = []
    
    for i, img_path in enumerate(test_images):
        if not Path(img_path).exists():
            print(f"⚠️  图片不存在: {img_path}")
            continue
            
        print(f"\n🖼️  处理图片 {i+1}: {img_path}")
        
        try:
            # 加载并复制图片到输出目录
            image = Image.open(img_path)
            print(f"   原始尺寸: {image.size}, 模式: {image.mode}")
            
            # 确保图像是RGB模式（JPEG兼容）
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
                print(f"   🔄 转换为RGB模式")
            
            # 保存原图副本
            filename = f"test_frame_{i+1:02d}_{int(time.time())}.jpg"
            filepath = capturer.session_dir / filename
            image.save(filepath, 'JPEG', quality=95)
            print(f"   💾 已保存: {filename}")
            
            # 裁剪右上角
            width, height = image.size
            crop_ratio = 0.25
            crop_width = int(width * crop_ratio)
            crop_height = int(height * crop_ratio)
            
            left = width - crop_width
            top = 0
            right = width
            bottom = crop_height
            
            cropped = image.crop((left, top, right, bottom))
            print(f"   ✂️  裁剪尺寸: {cropped.size}")
            
            # 确保裁剪图片也是RGB模式
            if cropped.mode in ('RGBA', 'LA', 'P'):
                cropped = cropped.convert('RGB')
            
            # 保存裁剪图片
            cropped_filename = f"test_frame_{i+1:02d}_cropped_{int(time.time())}.jpg"
            cropped_filepath = capturer.session_dir / cropped_filename
            cropped.save(cropped_filepath, 'JPEG', quality=95)
            print(f"   💾 裁剪已保存: {cropped_filename}")
            
            # OCR识别
            import numpy as np
            cropped_array = np.array(cropped.convert('RGB'))
            result = ocr_engine.infer(cropped_array)
            
            # 分析结果
            texts = [line.text for line in result.texts]
            has_time = any(':' in text and any(c.isdigit() for c in text) for text in texts)
            has_replay = any('播' in text or '重播' in text or 'REPLAY' in text.upper() for text in texts)
            
            print(f"   📝 识别到 {len(texts)} 行文本")
            for j, text in enumerate(texts[:5]):  # 显示前5行
                print(f"      {j+1}. {text}")
            print(f"   🕒 包含时间: {has_time}")
            print(f"   📺 包含重播标识: {has_replay}")
            print(f"   ⏱️  OCR耗时: {result.time_ms}ms")
            
            analysis_results.append({
                "source": img_path,
                "frame_file": str(filepath),
                "cropped_file": str(cropped_filepath),
                "texts": texts,
                "has_time": has_time,
                "has_replay": has_replay,
                "ocr_time_ms": result.time_ms
            })
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    # 总结
    print(f"\n📋 测试总结:")
    print(f"   输出目录: {capturer.session_dir}")
    print(f"   处理图片: {len(analysis_results)}")
    
    time_count = sum(1 for a in analysis_results if a.get('has_time', False))
    replay_count = sum(1 for a in analysis_results if a.get('has_replay', False))
    
    if time_count > 0:
        print(f"   🕒 包含时间的图片: {time_count}")
    if replay_count > 0:
        print(f"   📺 包含重播标识的图片: {replay_count}")


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_local_images()
        return
        
    parser = argparse.ArgumentParser(description='快速获取直播流帧')
    parser.add_argument('stream_url', help='直播流URL (HLS/DASH)')
    parser.add_argument('-n', '--frames', type=int, default=2, help='获取帧数 (默认: 2)')
    parser.add_argument('-w', '--width', type=int, default=1280, help='缩放宽度 (默认: 1280)')
    parser.add_argument('--no-ocr', action='store_true', help='不进行OCR分析')
    parser.add_argument('-o', '--output', default='app/output', help='输出目录 (默认: app/output)')
    
    args = parser.parse_args()
    
    print("🚀 直播流帧捕获工具")
    print("=" * 50)
    
    # 创建捕获器
    capturer = StreamFrameCapture(args.stream_url, args.output)
    
    if args.no_ocr:
        # 只捕获帧，不进行OCR分析
        saved_files = capturer.capture_frames(args.frames, args.width)
        print(f"\n📋 总结:")
        print(f"   捕获帧数: {len(saved_files)}")
        print(f"   保存位置: {capturer.session_dir}")
    else:
        # 捕获帧并进行OCR分析
        result = capturer.capture_and_analyze(args.frames)
        
        print(f"\n📋 总结:")
        print(f"   会话目录: {result['session_dir']}")
        print(f"   捕获帧数: {len(result['frames'])}")
        print(f"   分析结果: {len(result['analysis'])} 个")
        
        # 显示分析汇总
        time_frames = sum(1 for a in result['analysis'] if a.get('has_time', False))
        replay_frames = sum(1 for a in result['analysis'] if a.get('has_replay', False))
        
        if time_frames > 0:
            print(f"   🕒 包含时间信息的帧: {time_frames}")
        if replay_frames > 0:
            print(f"   📺 包含重播标识的帧: {replay_frames}")


if __name__ == "__main__":
    main()
