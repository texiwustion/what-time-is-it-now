#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FFmpeg "玩法" 抽象系统
定义各种FFmpeg操作的最小抽象接口和实现
"""

from __future__ import annotations
from typing import Protocol, List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import subprocess
import io
from abc import ABC, abstractmethod


@dataclass
class FFmpegCommand:
    """FFmpeg命令配置"""
    executable: str = 'ffmpeg'                    # FFmpeg可执行文件名
    hide_banner: bool = True                      # 隐藏版权信息横幅
    log_level: str = 'error'                     # 日志级别 (quiet/error/warning/info/debug)
    input_flags: Dict[str, Any] = None           # 输入相关参数
    input_source: str = ''                       # 输入源
    video_filters: List[str] = None              # 视频滤镜列表
    output_flags: Dict[str, Any] = None          # 输出相关参数
    output_format: str = ''                      # 输出格式
    output_target: str = '-'                     # 输出目标
    
    def __post_init__(self):
        if self.input_flags is None:
            self.input_flags = {}
        if self.video_filters is None:
            self.video_filters = []
        if self.output_flags is None:
            self.output_flags = {}
    
    def build_command(self) -> List[str]:
        """构建完整的FFmpeg命令行"""
        cmd = [self.executable]
        
        # 基础参数
        if self.hide_banner:
            cmd.extend(['-hide_banner'])
        cmd.extend(['-loglevel', self.log_level])
        
        # 输入参数
        for key, value in self.input_flags.items():
            cmd.extend([f'-{key}', str(value)])
        
        # 输入源
        if self.input_source:
            cmd.extend(['-i', self.input_source])
        
        # 视频滤镜
        if self.video_filters:
            filter_chain = ','.join(self.video_filters)
            cmd.extend(['-vf', filter_chain])
        
        # 输出参数
        for key, value in self.output_flags.items():
            cmd.extend([f'-{key}', str(value)])
        
        # 输出格式
        if self.output_format:
            cmd.extend(['-f', self.output_format])
        
        # 输出目标
        cmd.append(self.output_target)
        
        return cmd


class FFmpeg玩法Protocol(Protocol):
    """FFmpeg玩法的抽象协议"""
    
    def build_command(self) -> List[str]:
        """构建FFmpeg命令"""
        ...
    
    def execute(self) -> Any:
        """执行FFmpeg命令并返回结果"""
        ...


class 流媒体玩法Protocol(Protocol):
    """流媒体相关玩法的协议"""
    
    def set_stream_url(self, url: str) -> None:
        """设置流媒体URL"""
        ...
    
    def set_reconnect(self, enable: bool) -> None:
        """设置是否自动重连"""
        ...
    
    def capture_frames(self, count: int) -> Iterator[bytes]:
        """捕获指定数量的帧"""
        ...


class 图像处理玩法Protocol(Protocol):
    """图像处理相关玩法的协议"""
    
    def set_scale(self, width: int, height: int = -2) -> None:
        """设置缩放尺寸"""
        ...
    
    def set_fps(self, fps: float) -> None:
        """设置帧率"""
        ...
    
    def add_filter(self, filter_name: str, params: Dict[str, Any] = None) -> None:
        """添加视频滤镜"""
        ...


class 输出玩法Protocol(Protocol):
    """输出相关玩法的协议"""
    
    def set_output_format(self, format_name: str) -> None:
        """设置输出格式"""
        ...
    
    def set_codec(self, codec_name: str) -> None:
        """设置编码器"""
        ...
    
    def pipe_output(self) -> bool:
        """是否管道输出"""
        ...


class BaseFFmpeg玩法(ABC):
    """FFmpeg玩法基础实现"""
    
    def __init__(self):
        self.command = FFmpegCommand()
        self._process: Optional[subprocess.Popen] = None
    
    @abstractmethod
    def build_command(self) -> List[str]:
        """构建FFmpeg命令（子类必须实现）"""
        pass
    
    def execute(self) -> subprocess.Popen:
        """执行FFmpeg命令"""
        cmd = self.build_command()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        return self._process
    
    def terminate(self):
        """终止FFmpeg进程"""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


class 直播流帧捕获玩法(BaseFFmpeg玩法):
    """直播流帧捕获的具体实现"""
    
    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self.frame_count = 2
        self.scale_width = 1280
        self.fps = 1.0
        
        # 配置默认参数
        self.command.input_flags = {
            'fflags': 'nobuffer',
            'reconnect': '1'
        }
        self.command.output_format = 'image2pipe'
        self.command.output_flags = {'vcodec': 'mjpeg'}
    
    def set_stream_url(self, url: str) -> None:
        """设置流媒体URL"""
        self.stream_url = url
    
    def set_frame_count(self, count: int) -> None:
        """设置捕获帧数"""
        self.frame_count = count
    
    def set_scale(self, width: int, height: int = -2) -> None:
        """设置缩放尺寸"""
        self.scale_width = width
    
    def set_fps(self, fps: float) -> None:
        """设置帧率"""
        self.fps = fps
    
    def set_reconnect(self, enable: bool) -> None:
        """设置是否自动重连"""
        if enable:
            self.command.input_flags['reconnect'] = '1'
        else:
            self.command.input_flags.pop('reconnect', None)
    
    def build_command(self) -> List[str]:
        """构建FFmpeg命令"""
        self.command.input_source = self.stream_url
        self.command.video_filters = [
            f'fps={self.fps}',
            f'scale={self.scale_width}:-2'
        ]
        self.command.output_flags['vframes'] = str(self.frame_count)
        
        return self.command.build_command()
    
    def capture_frames(self, count: int = None) -> Iterator[bytes]:
        """捕获指定数量的帧"""
        if count:
            self.set_frame_count(count)
        
        process = self.execute()
        
        # JPEG帧分隔符
        jpeg_start = b'\xff\xd8'
        jpeg_end = b'\xff\xd9'
        
        buffer = b''
        frame_count = 0
        
        while frame_count < self.frame_count and process.poll() is None:
            chunk = process.stdout.read(4096)
            if not chunk:
                continue
                
            buffer += chunk
            
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
                
                frame_count += 1
                yield jpeg_data
                
                if frame_count >= self.frame_count:
                    break
        
        self.terminate()


class 视频转图片玩法(BaseFFmpeg玩法):
    """视频转图片的玩法"""
    
    def __init__(self, input_file: str, output_pattern: str):
        super().__init__()
        self.input_file = input_file
        self.output_pattern = output_pattern
        self.command.output_format = 'image2'
        self.command.output_flags = {'vcodec': 'mjpeg'}
    
    def set_time_range(self, start: str, duration: str = None):
        """设置时间范围"""
        self.command.input_flags['ss'] = start
        if duration:
            self.command.input_flags['t'] = duration
    
    def set_frame_rate(self, fps: float):
        """设置输出帧率"""
        self.command.video_filters.append(f'fps={fps}')
    
    def build_command(self) -> List[str]:
        """构建FFmpeg命令"""
        self.command.input_source = self.input_file
        self.command.output_target = self.output_pattern
        return self.command.build_command()


class 流媒体录制玩法(BaseFFmpeg玩法):
    """流媒体录制玩法"""
    
    def __init__(self, stream_url: str, output_file: str):
        super().__init__()
        self.stream_url = stream_url
        self.output_file = output_file
        self.duration = None
        
        # 配置录制参数
        self.command.input_flags = {
            'fflags': 'nobuffer',
            'reconnect': '1'
        }
        self.command.output_flags = {
            'c': 'copy'  # 复制流，不重新编码
        }
    
    def set_duration(self, seconds: int):
        """设置录制时长"""
        self.duration = seconds
        self.command.output_flags['t'] = str(seconds)
    
    def set_quality(self, crf: int = 23):
        """设置录制质量"""
        self.command.output_flags.pop('c', None)  # 移除copy选项
        self.command.output_flags['crf'] = str(crf)
    
    def build_command(self) -> List[str]:
        """构建FFmpeg命令"""
        self.command.input_source = self.stream_url
        self.command.output_target = self.output_file
        return self.command.build_command()


class FFmpeg玩法工厂:
    """FFmpeg玩法工厂类"""
    
    @staticmethod
    def create_直播流帧捕获(stream_url: str) -> 直播流帧捕获玩法:
        """创建直播流帧捕获玩法"""
        return 直播流帧捕获玩法(stream_url)
    
    @staticmethod
    def create_视频转图片(input_file: str, output_pattern: str) -> 视频转图片玩法:
        """创建视频转图片玩法"""
        return 视频转图片玩法(input_file, output_pattern)
    
    @staticmethod
    def create_流媒体录制(stream_url: str, output_file: str) -> 流媒体录制玩法:
        """创建流媒体录制玩法"""
        return 流媒体录制玩法(stream_url, output_file)
    
    @staticmethod
    def create_custom_玩法(command_config: FFmpegCommand) -> BaseFFmpeg玩法:
        """创建自定义玩法"""
        class Custom玩法(BaseFFmpeg玩法):
            def __init__(self, config: FFmpegCommand):
                super().__init__()
                self.command = config
            
            def build_command(self) -> List[str]:
                return self.command.build_command()
        
        return Custom玩法(command_config)


# 使用示例
def demo_usage():
    """使用示例"""
    print("FFmpeg玩法系统演示")
    print("=" * 50)
    
    # 示例1: 直播流帧捕获
    print("1. 直播流帧捕获玩法")
    stream_capture = FFmpeg玩法工厂.create_直播流帧捕获("https://example.com/live.m3u8")
    stream_capture.set_frame_count(3)
    stream_capture.set_fps(2.0)
    stream_capture.set_scale(1920)
    
    cmd = stream_capture.build_command()
    print(f"   命令: {' '.join(cmd)}")
    
    # 示例2: 视频转图片
    print("\n2. 视频转图片玩法")
    video_to_images = FFmpeg玩法工厂.create_视频转图片("input.mp4", "frame_%03d.jpg")
    video_to_images.set_time_range("00:01:00", "00:00:10")
    video_to_images.set_frame_rate(1.0)
    
    cmd = video_to_images.build_command()
    print(f"   命令: {' '.join(cmd)}")
    
    # 示例3: 流媒体录制
    print("\n3. 流媒体录制玩法")
    stream_record = FFmpeg玩法工厂.create_流媒体录制("https://example.com/live.m3u8", "recorded.mp4")
    stream_record.set_duration(300)  # 5分钟
    
    cmd = stream_record.build_command()
    print(f"   命令: {' '.join(cmd)}")
    
    # 示例4: 自定义玩法
    print("\n4. 自定义玩法")
    custom_config = FFmpegCommand(
        input_source="test.mp4",
        video_filters=["scale=640:480", "fps=15"],
        output_format="gif",
        output_target="output.gif"
    )
    custom_play = FFmpeg玩法工厂.create_custom_玩法(custom_config)
    
    cmd = custom_play.build_command()
    print(f"   命令: {' '.join(cmd)}")
    
    print("\n" + "=" * 50)
    print("抽象系统特点:")
    print("• Protocol定义接口规范")
    print("• 工厂模式创建不同玩法")
    print("• 链式调用配置参数") 
    print("• 统一的命令构建逻辑")
    print("• 可扩展的自定义玩法")


if __name__ == "__main__":
    demo_usage()
