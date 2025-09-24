#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time
import io
import sys
from pathlib import Path
from typing import Optional, Tuple, Generator, Callable
import numpy as np
from PIL import Image
import threading
import queue
import logging

# 添加上级目录到Python路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 导入上级目录的模块
from paddle_ocr import PaddleOCREngine
from ocr import OCRResult

import sys

# 配置更显眼的日志格式
logging.basicConfig(
    level=logging.INFO,  # 改为INFO级别，减少过多的DEBUG信息
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 强制输出到stdout
    ]
)

# 创建日志记录器
logger = logging.getLogger(__name__)

# 设置PaddleOCR相关日志级别为WARNING，减少干扰
logging.getLogger('paddleocr').setLevel(logging.WARNING)
logging.getLogger('paddlex').setLevel(logging.WARNING)
logging.getLogger('paddle').setLevel(logging.WARNING)


class ImageCropper:
    """图像裁剪工具类 - 专门用于裁剪右上角区域"""
    
    def __init__(self, crop_ratio: float = 0.25):
        """初始化裁剪器
        
        Args:
            crop_ratio: 裁剪比例，0.25表示裁剪右上角25%的区域
        """
        self.crop_ratio = crop_ratio
    
    def crop_top_right(self, image: Image.Image) -> Image.Image:
        """裁剪图像的右上角区域
        
        Args:
            image: PIL图像对象
            
        Returns:
            裁剪后的图像
        """
        width, height = image.size
        
        # 计算裁剪区域（右上角）
        crop_width = int(width * self.crop_ratio)
        crop_height = int(height * self.crop_ratio)
        
        # 裁剪坐标 (left, top, right, bottom)
        left = width - crop_width
        top = 0
        right = width
        bottom = crop_height
        
        cropped = image.crop((left, top, right, bottom))
        logger.debug(f"裁剪区域: ({left}, {top}, {right}, {bottom}), 原尺寸: {width}x{height}, 裁剪后: {crop_width}x{crop_height}")
        
        return cropped
    
    def get_crop_coordinates(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """获取裁剪坐标
        
        Args:
            width: 图像宽度
            height: 图像高度
            
        Returns:
            (left, top, right, bottom) 裁剪坐标
        """
        crop_width = int(width * self.crop_ratio)
        crop_height = int(height * self.crop_ratio)
        
        left = width - crop_width
        top = 0
        right = width
        bottom = crop_height
        
        return left, top, right, bottom


class FFmpegStreamOCR:
    """FFmpeg流媒体OCR处理器 - 持续识别右上角时间信息"""
    
    def __init__(self, 
                 stream_url: str,
                 ocr_engine: Optional[PaddleOCREngine] = None,
                 crop_ratio: float = 0.25,
                 fps: int = 1,
                 scale_width: int = 1280,
                 max_queue_size: int = 10):
        """初始化流处理器
        
        Args:
            stream_url: 流媒体URL (HLS/DASH)
            ocr_engine: OCR引擎实例
            crop_ratio: 右上角裁剪比例
            fps: 抽帧帧率
            scale_width: 缩放宽度
            max_queue_size: 帧队列最大长度
        """
        self.stream_url = stream_url
        self.ocr_engine = ocr_engine or PaddleOCREngine()
        self.cropper = ImageCropper(crop_ratio)
        self.content_analyzer = ContentAnalyzer()
        self.fps = fps
        self.scale_width = scale_width
        
        # 线程安全的帧队列
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        
        # 控制标志
        self.running = False
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        
        # 线程
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
    
    def _build_ffmpeg_command(self) -> list[str]:
        """构建FFmpeg命令"""
        return [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-fflags', 'nobuffer',
            '-reconnect', '1',
            '-i', self.stream_url,
            '-vf', f'fps={self.fps},scale={self.scale_width}:-2',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-'
        ]
    
    def _capture_frames(self):
        """帧捕获线程 - 从FFmpeg管道读取JPEG帧"""
        try:
            cmd = self._build_ffmpeg_command()
            logger.info(f"🎬 启动FFmpeg命令: {' '.join(cmd)}")
            
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info("✅ FFmpeg进程已启动")
            
            # JPEG文件头标识
            jpeg_start = b'\xff\xd8'
            jpeg_end = b'\xff\xd9'
            
            buffer = b''
            frame_count = 0
            
            logger.info("🔄 开始读取FFmpeg输出流...")
            
            while self.running and self.ffmpeg_process.poll() is None:
                # 读取数据块
                chunk = self.ffmpeg_process.stdout.read(4096)
                if not chunk:
                    logger.debug("📭 未读取到数据块")
                    continue
                    
                buffer += chunk
                logger.debug(f"📦 读取数据块: {len(chunk)} bytes, 缓冲区总大小: {len(buffer)} bytes")
                
                # 查找完整的JPEG帧
                while True:
                    start_idx = buffer.find(jpeg_start)
                    if start_idx == -1:
                        break
                    
                    end_idx = buffer.find(jpeg_end, start_idx + 2)
                    if end_idx == -1:
                        break
                    
                    # 提取完整JPEG帧
                    jpeg_data = buffer[start_idx:end_idx + 2]
                    buffer = buffer[end_idx + 2:]
                    
                    try:
                        # 转换为PIL图像
                        image = Image.open(io.BytesIO(jpeg_data))
                        frame_count += 1
                        logger.info(f"🖼️  成功解析第{frame_count}帧，尺寸: {image.size}")
                        
                        # 放入队列（非阻塞，队列满时丢弃旧帧）
                        try:
                            self.frame_queue.put(image, block=False)
                            logger.debug(f"✅ 帧已放入队列，当前队列大小: {self.frame_queue.qsize()}")
                        except queue.Full:
                            # 队列满时，移除最旧的帧
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put(image, block=False)
                                logger.warning("⚠️  队列已满，丢弃旧帧")
                            except queue.Empty:
                                pass
                                
                    except Exception as e:
                        logger.warning(f"解析JPEG帧失败: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"帧捕获线程异常: {e}")
        finally:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process = None
    
    def _process_frames(self):
        """帧处理线程 - OCR识别右上角区域"""
        logger.info("🔍 OCR处理线程已启动")
        processed_count = 0
        
        while self.running:
            try:
                # 获取帧（阻塞等待，超时1秒）
                logger.debug("⏳ 等待获取帧...")
                image = self.frame_queue.get(timeout=1.0)
                processed_count += 1
                logger.info(f"🎯 开始处理第{processed_count}帧，原始尺寸: {image.size}")
                
                # 裁剪右上角
                cropped_image = self.cropper.crop_top_right(image)
                logger.info(f"✂️  裁剪完成，裁剪后尺寸: {cropped_image.size}")
                
                # 将PIL图像转换为numpy数组并确保正确的颜色格式
                cropped_array = np.array(cropped_image)
                logger.debug(f"🔄 原始数组形状: {cropped_array.shape}")
                
                # 确保图像是RGB格式（3通道）
                if len(cropped_array.shape) == 2:
                    # 灰度图转RGB
                    cropped_array = np.stack([cropped_array] * 3, axis=-1)
                    logger.debug("🔄 灰度图转换为RGB")
                elif len(cropped_array.shape) == 3 and cropped_array.shape[2] == 4:
                    # RGBA转RGB（移除alpha通道）
                    cropped_array = cropped_array[:, :, :3]
                    logger.debug("🔄 RGBA转换为RGB")
                elif len(cropped_array.shape) == 3 and cropped_array.shape[2] != 3:
                    logger.warning(f"⚠️  不支持的图像格式，通道数: {cropped_array.shape[2]}")
                    continue
                    
                # 确保数据类型正确
                if cropped_array.dtype != np.uint8:
                    cropped_array = cropped_array.astype(np.uint8)
                    logger.debug("🔄 转换数据类型为uint8")
                
                logger.debug(f"🔄 处理后数组形状: {cropped_array.shape}")
                
                # OCR识别
                logger.info("🔤 开始OCR识别...")
                start_time = time.time()
                ocr_result = self.ocr_engine.infer(cropped_array)
                process_time = time.time() - start_time
                logger.info(f"✅ OCR识别完成，耗时: {process_time*1000:.1f}ms")

                # print(f"🔍 OCR结果: {ocr_result.texts}")
                logger.info(f"📝 识别到{len(ocr_result.texts)}行文本")
                for i, text_line in enumerate(ocr_result.texts):
                    logger.info(f"   第{i+1}行: '{text_line.text}' (置信度: {text_line.confidence:.3f})")
                
                # 内容分析
                logger.debug("🧠 开始内容分析...")
                content_analysis = self.content_analyzer.analyze_texts(ocr_result.texts)
                logger.info(f"🎯 分析结果 - 时间: {content_analysis['has_time']}, 重播: {content_analysis['is_replay']}")
                if (content_analysis['has_time']):
                    print(f"🕒 时间: {content_analysis['time_texts']}")
                    print(f"关键时刻：{content_analysis['ge_20_min']}")
                if (content_analysis['is_replay']):
                    print(f"📺 重播: {content_analysis['replay_indicators']}")
                
                # 构建结果
                result = {
                    'timestamp': time.time(),
                    'ocr_result': ocr_result,
                    'content_analysis': content_analysis,
                    'process_time_ms': int(process_time * 1000),
                    'original_size': image.size,
                    'cropped_size': cropped_image.size
                }
                
                # 放入结果队列
                try:
                    self.result_queue.put(result, block=False)
                except queue.Full:
                    # 结果队列满时移除最旧结果
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(result, block=False)
                    except queue.Empty:
                        pass
                
                logger.debug(f"OCR处理完成，识别到{len(ocr_result.texts)}行文本，耗时{process_time*1000:.1f}ms")
                
            except queue.Empty:
                logger.debug("⏰ 帧队列超时，继续等待...")
                continue
            except Exception as e:
                logger.error(f"❌ 帧处理线程异常: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
    
    def start(self):
        """启动流处理"""
        if self.running:
            logger.warning("⚠️  流处理已在运行")
            return
            
        logger.info("🚀 启动流处理系统...")
        self.running = True
        
        # 启动线程
        logger.info("📡 启动帧捕获线程...")
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        logger.info("🔍 启动OCR处理线程...")
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.process_thread.start()
        
        logger.info("✅ 流处理系统启动完成！")
        
        # 等待一下确保线程启动
        time.sleep(0.5)
        logger.info(f"📊 线程状态 - 捕获线程: {'运行' if self.capture_thread.is_alive() else '停止'}, 处理线程: {'运行' if self.process_thread.is_alive() else '停止'}")
    
    def stop(self):
        """停止流处理"""
        self.running = False
        
        # 终止FFmpeg进程
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
        
        # 等待线程结束
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.process_thread:
            self.process_thread.join(timeout=2)
            
        logger.info("流处理已停止")
    
    def get_latest_result(self) -> Optional[dict]:
        """获取最新的OCR识别结果"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_results(self) -> list[dict]:
        """获取所有待处理的OCR结果"""
        results = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results
    
    def is_currently_replay(self) -> Optional[bool]:
        """快速检查当前是否为重播状态"""
        result = self.get_latest_result()
        if result and 'content_analysis' in result:
            return result['content_analysis']['is_replay']
        return None
    
    def get_current_time_info(self) -> Optional[list]:
        """获取当前检测到的时间信息"""
        result = self.get_latest_result()
        if result and 'content_analysis' in result:
            return result['content_analysis']['time_texts']
        return None
    
    def get_replay_indicators(self) -> Optional[list]:
        """获取当前检测到的重播指示器"""
        result = self.get_latest_result()
        if result and 'content_analysis' in result:
            return result['content_analysis']['replay_indicators']
        return None


class ContentAnalyzer:
    """内容分析器 - 识别时间、重播等特定内容"""
    
    def __init__(self):
        import re
        self.re = re
        
        # 时间格式模式（更稳健）：匹配任意位置的 MM:SS / MM.SS / MM：SS / MM ． SS，避免被其他数字粘连
        # 使用负向/正向断言，防止被更长数字串吞并
        self.time_regex = self.re.compile(r"(?<!\d)(\d{1,2})[\:：\.  ][\s]?(\d{2})(?!\d)")
        
        # 兼容旧逻辑（保留占位，内部统一用 time_regex）
        self.time_patterns = [
            r'时间(\d{2}):(\d{2})',      # 时间MM:SS
            r'时间(\d{2})\.(\d{2})',     # 时间MM.SS
            r'(\d{2}):(\d{2})',          # MM:SS (单独的时间)
            r'(\d{2})\.(\d{2})',         # MM.SS (单独的时间)
        ]
        
        # 重播相关关键词
        self.replay_keywords = [
            '播',      # 重播、回播
            '重播',    # 重播
            '回播',    # 回播
            '录播',    # 录播
            '重放',    # 重放
            'REPLAY',  # 英文重播
            'RERUN',   # 英文重播
            '精选',    # 精选重播
            '回看',    # 回看
        ]
        
        # 直播相关关键词（用于排除误判）
        self.live_keywords = [
            '直播',
            '现场',
            'LIVE',
            '实况',
            '正在播出',
        ]
    
    def _normalize_for_time(self, text: str) -> str:
        """将文本规整为便于时间识别的形态。
        - 全角转半角（NFKC）
        - 常见混淆字符替换：'O'->'0', 'o'->'0', '．'->'.', '：'->':'
        - 去除多余空格
        """
        import unicodedata
        t = unicodedata.normalize('NFKC', text)
        repl = {
            'O': '0', 'o': '0', '〇': '0', '零': '0',
            '：': ':', '。': '.', '．': '.', '·': '.',
        }
        for k, v in repl.items():
            t = t.replace(k, v)
        return t
    
    def is_time_text(self, text: str) -> bool:
        """判断是否为时间格式（使用归一化 + 稳健正则）"""
        norm = self._normalize_for_time(text)
        return self.time_regex.search(norm) is not None
    
    def extract_time(self, text: str) -> Optional[str]:
        """从文本中提取时间并转换为标准格式（HH:MM:SS）。仅返回首个匹配。"""
        from datetime import time as _time
        norm = self._normalize_for_time(text)
        m = self.time_regex.search(norm)
        if not m:
            return None
        mm = int(m.group(1))
        ss = int(m.group(2))
        if not (0 <= mm <= 99 and 0 <= ss < 60):
            return None
        try:
            return _time(hour=0, minute=mm, second=ss).strftime("%H:%M:%S")
        except Exception:
            return f"00:{mm:02d}:{ss:02d}"
    
    def extract_time_sec(self, text: str) -> Optional[int]:
        """提取首个时间并返回秒数。"""
        norm = self._normalize_for_time(text)
        m = self.time_regex.search(norm)
        if not m:
            return None
        mm = int(m.group(1))
        ss = int(m.group(2))
        if 0 <= mm <= 99 and 0 <= ss < 60:
            return mm * 60 + ss
        return None
    
    def extract_all_times(self, ocr_lines) -> list:
        """从多行 OCR 文本中提取全部时间，返回带标准化字段。"""
        items = []
        for line in ocr_lines:
            text = getattr(line, 'text', '') or ''
            norm = self._normalize_for_time(text)
            for m in self.time_regex.finditer(norm):
                mm = int(m.group(1))
                ss = int(m.group(2))
                if 0 <= mm <= 99 and 0 <= ss < 60:
                    items.append({
                        'text': text,
                        'norm': f"{mm:02d}:{ss:02d}",
                        'mm': mm,
                        'ss': ss,
                        'sec': mm * 60 + ss,
                        'confidence': getattr(line, 'confidence', None),
                        'bbox': getattr(line, 'bbox', None),
                    })
        return items
    
    def has_reached_20_min(self, ocr_lines) -> bool:
        """判断 OCR 行文本中是否存在 >= 20:00 的时间。"""
        times = self.extract_all_times(ocr_lines)
        if not times:
            return False
        max_sec = max(t['sec'] for t in times)
        return max_sec >= 20 * 60
    
    def is_replay_indicator(self, text: str) -> bool:
        """判断是否为重播指示器"""
        # 先检查是否包含直播关键词（排除误判）
        for live_word in self.live_keywords:
            if live_word in text:
                return False
        
        # 检查重播关键词
        for replay_word in self.replay_keywords:
            if replay_word in text:
                return True
        return False
    
    def analyze_texts(self, ocr_lines) -> dict:
        """分析OCR结果中的时间和重播信息"""
        result = {
            'time_texts': [],
            'replay_indicators': [],
            'is_replay': False,
            'has_time': False,
            'ge_20_min': False,
            'max_time_sec': None,
        }
        
        # 提取时间
        all_times = self.extract_all_times(ocr_lines)
        if all_times:
            result['time_texts'] = all_times
            result['has_time'] = True
            result['max_time_sec'] = max(t['sec'] for t in all_times)
            result['ge_20_min'] = result['max_time_sec'] >= 20 * 60
        
        # 提取重播
        for line in ocr_lines:
            text = line.text.strip()
            if not text:
                continue
            if self.is_replay_indicator(text):
                result['replay_indicators'].append({
                    'text': text,
                    'confidence': line.confidence,
                    'bbox': line.bbox
                })
                result['is_replay'] = True
        
        return result


def time_text_filter(text: str) -> bool:
    """时间文本过滤器 - 判断是否为时间格式（保持向后兼容）"""
    analyzer = ContentAnalyzer()
    return analyzer.is_time_text(text)


# 使用示例
def main():
    """主函数示例"""
    # 流媒体URL（替换为实际的HLS或DASH地址）
    stream_url = "https://d1--cn-gotcha204b.bilivideo.com/live-bvc/211604/live_50329485_5259019_2500/index.m3u8?expires=1758716284&len=0&oi=1001173025&pt=html5&qn=250&trid=10071bb3e5ef9e8578f96449d5f44268d3d3&bmt=1&sigparams=cdn,expires,len,oi,pt,qn,trid,bmt&cdn=cn-gotcha204&sign=4821bebd022419c57f0d62ca3f6f5392&site=c8ad522035124d331d16f8b106170aef&free_type=0&mid=0&sche=ban&bvchls=1&trace=4&isp=fx&rg=Central&pv=Hubei&origin_bitrate=2242&media_type=0&deploy_env=prod&hdr_type=0&pp=rtmp&flvsk=e9111b082c7df4c25b0b5ceed178b4b3&source=puv3_onetier&score=1&info_source=cache&sk=f46bd97a996b066e2d719403bf9cdaca&p2p_type=-1&sl=2&hot_cdn=909709&suffix=2500&codec=0&vd=bc&src=puv3&order=2"
    
    # 创建OCR引擎
    ocr_engine = PaddleOCREngine()
    
    # 创建流处理器
    processor = FFmpegStreamOCR(
        stream_url=stream_url,
        ocr_engine=ocr_engine,
        crop_ratio=0.3,  # 裁剪右上角30%区域
        fps=1,           # 每秒2帧
        scale_width=1280 # 缩放到1280像素宽
    )
    
    try:
        # 启动处理
        processor.start()
        logger.info("开始监控右上角时间信息...")
        
        # 持续获取结果
        while True:
            time.sleep(1)  # 每秒检查一次
            
            # 获取最新结果
            result = processor.get_latest_result()
            if result:
                content_analysis = result['content_analysis']
                
                # 显示时间信息
                if content_analysis['has_time']:
                    time_texts = [item['text'] for item in content_analysis['time_texts']]
                    logger.info(f"🕒 检测到时间: {', '.join(time_texts)}")
                
                # 显示重播状态
                if content_analysis['is_replay']:
                    replay_indicators = [item['text'] for item in content_analysis['replay_indicators']]
                    logger.info(f"📺 检测到重播标识: {', '.join(replay_indicators)}")
                    logger.info("🔄 当前状态: 重播内容")
                else:
                    logger.info("🔴 当前状态: 直播内容")
                
                # 显示处理信息
                if content_analysis['has_time'] or content_analysis['is_replay']:
                    logger.info(f"⏱️  处理耗时: {result['process_time_ms']}ms")
                    logger.info("-" * 50)
                    
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        processor.stop()


def test_with_local_image():
    """使用本地图片测试OCR功能"""
    print("\n" + "="*60)
    print("🧪 开始本地图片测试...")
    print("="*60)
    logger.info("🧪 开始本地图片测试...")
    
    # 创建OCR引擎和分析器
    print("📦 正在初始化OCR引擎...")
    ocr_engine = PaddleOCREngine()
    cropper = ImageCropper(crop_ratio=0.25)
    analyzer = ContentAnalyzer()
    print("✅ OCR引擎初始化完成")
    
    # 测试图片路径
    test_images = ["3.png", "1.png"]  # 根据你的项目调整
    
    for img_path in test_images:
        try:
            print(f"\n📸 测试图片: {img_path}")
            logger.info(f"📸 测试图片: {img_path}")
            
            # 加载图片并确保RGB格式
            image = Image.open(img_path)
            print(f"   原始尺寸: {image.size}, 模式: {image.mode}")
            logger.info(f"原始尺寸: {image.size}, 模式: {image.mode}")
            
            # 确保图像是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"   🔄 转换图像模式 -> RGB")
                logger.info(f"🔄 转换图像模式 -> RGB")
            
            # 裁剪右上角
            cropped = cropper.crop_top_right(image)
            print(f"   裁剪后尺寸: {cropped.size}")
            logger.info(f"裁剪后尺寸: {cropped.size}")
            
            # 转换为numpy数组并确保正确的颜色格式
            cropped_array = np.array(cropped)
            logger.info(f"原始数组形状: {cropped_array.shape}")
            
            # 确保图像是RGB格式（3通道）
            if len(cropped_array.shape) == 2:
                # 灰度图转RGB
                cropped_array = np.stack([cropped_array] * 3, axis=-1)
                logger.info("🔄 灰度图转换为RGB")
            elif cropped_array.shape[2] == 4:
                # RGBA转RGB（移除alpha通道）
                cropped_array = cropped_array[:, :, :3]
                logger.info("🔄 RGBA转换为RGB")
            elif cropped_array.shape[2] != 3:
                logger.error(f"❌ 不支持的图像格式，通道数: {cropped_array.shape[2]}")
                continue
                
            logger.info(f"处理后数组形状: {cropped_array.shape}")
            
            # 确保数据类型正确
            if cropped_array.dtype != np.uint8:
                cropped_array = cropped_array.astype(np.uint8)
                logger.info("🔄 转换数据类型为uint8")
            
            # OCR识别
            print("   🔤 开始OCR识别...")
            logger.info("🔤 开始OCR识别...")
            result = ocr_engine.infer(cropped_array)
            print(f"   ✅ OCR识别完成，耗时: {result.time_ms}ms")
            logger.info(f"识别完成，耗时: {result.time_ms}ms")
            
            print(f"\n📝 识别到 {len(result.texts)} 行文本:")
            for i, line in enumerate(result.texts):
                print(f"   第{i+1}行: '{line.text}' (置信度: {line.confidence:.3f})")
            
            # 内容分析
            analysis = analyzer.analyze_texts(result.texts)
            print(f"\n🎯 内容分析结果:")
            print(f"   - 包含时间: {analysis['has_time']}")
            print(f"   - 是重播: {analysis['is_replay']}")
            if analysis['time_texts']:
                time_list = [t['text'] for t in analysis['time_texts']]
                print(f"   - 时间信息: {time_list}")
            if analysis['replay_indicators']:
                replay_list = [r['text'] for r in analysis['replay_indicators']]
                print(f"   - 重播标识: {replay_list}")
            
            print("-" * 60)
            
        except FileNotFoundError:
            logger.warning(f"⚠️  图片文件不存在: {img_path}")
        except Exception as e:
            logger.error(f"❌ 测试图片 {img_path} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())


def simple_demo():
    """简化演示 - 快速检测重播状态"""
    # 模拟流媒体URL
    stream_url = "https://example.com/live/stream.m3u8"
    
    # 创建处理器
    processor = FFmpegStreamOCR(stream_url, crop_ratio=0.3)
    
    try:
        processor.start()
        logger.info("🚀 开始监控直播状态...")
        
        while True:
            time.sleep(2)  # 每2秒检查一次
            
            # 快速检查重播状态
            is_replay = processor.is_currently_replay()
            time_info = processor.get_current_time_info()
            
            if is_replay is not None:
                status = "🔄 重播" if is_replay else "🔴 直播"
                logger.info(f"状态: {status}")
                
                if time_info:
                    times = [item['text'] for item in time_info]
                    logger.info(f"时间: {', '.join(times)}")
                    
    except KeyboardInterrupt:
        logger.info("停止监控")
    finally:
        processor.stop()


if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 本地图片测试模式
        test_with_local_image()
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        # 简化演示模式
        simple_demo()
    else:
        # 完整示例模式
        main()