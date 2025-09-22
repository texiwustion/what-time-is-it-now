#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Union
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from ocr import OCREngineProto, OCRResult, OCRLine, ImageLike, BBox


class PaddleOCREngine(OCREngineProto):
    """PaddleOCR 引擎实现，符合 OCREngineProto 协议"""
    
    def __init__(self, 
                 use_doc_orientation_classify: bool = False,
                 use_doc_unwarping: bool = False,
                 use_textline_orientation: bool = False,
                 **kwargs):
        """初始化 PaddleOCR 引擎
        
        Args:
            use_doc_orientation_classify: 是否使用文档方向分类
            use_doc_unwarping: 是否使用文档去扭曲
            use_textline_orientation: 是否使用文本行方向
            **kwargs: 其他 PaddleOCR 参数
        """
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            **kwargs
        )
    
    def infer(self, image: ImageLike) -> OCRResult:
        """执行 OCR 推理
        
        Args:
            image: 图像输入，可以是文件路径、numpy数组或PIL图像
            
        Returns:
            OCRResult: 包含识别结果的数据结构
        """
        start_time = time.time()
        
        # 执行 OCR 推理
        result = self.ocr.predict(input=image)
        
        end_time = time.time()
        time_ms = int((end_time - start_time) * 1000)
        
        # 解析结果
        texts = []
        total_confidence = 0.0
        line_count = 0
        
        for page_result in result:
            # 获取检测框、识别文本和置信度
            dt_polys = page_result.get('dt_polys')
            rec_texts = page_result.get('rec_texts', [])
            rec_scores = page_result.get('rec_scores', [])
            
            if dt_polys is not None and len(rec_texts) > 0:
                for line_id, (poly, text, confidence) in enumerate(zip(dt_polys, rec_texts, rec_scores)):
                    # 将多边形转换为边界框格式 (4个点的坐标)
                    bbox = [[float(x), float(y)] for x, y in poly]
                    
                    ocr_line = OCRLine(
                        line_id=line_id,
                        text=text,
                        confidence=float(confidence),
                        bbox=bbox,
                        page_id=1
                    )
                    texts.append(ocr_line)
                    
                    total_confidence += confidence
                    line_count += 1
        
        # 计算平均置信度
        avg_confidence = total_confidence / line_count if line_count > 0 else 0.0
        
        return OCRResult(
            texts=texts,
            avg_confidence=avg_confidence,
            time_ms=time_ms,
            raw=result
        )
    
    def save_results(self, result, output_dir: str = "output"):
        """保存 OCR 结果到文件（兼容原有的保存功能）
        
        Args:
            result: OCR结果
            output_dir: 输出目录
        """

        # 保存结果（使用 PaddleOCR 原生的保存方法）
        for res in result:
            res.print()
            res.save_to_img(output_dir)
            res.save_to_json(output_dir)


# 使用示例
if __name__ == "__main__":
    # 创建引擎实例
    engine = PaddleOCREngine(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    
    # 使用协议接口进行推理
    result = engine.infer("./3.png")
    
    print(f"识别到 {len(result.texts)} 行文本")
    print(f"平均置信度: {result.avg_confidence:.2f}")
    print(f"处理时间: {result.time_ms} ms")
    
    # 打印前几行识别结果
    for i, line in enumerate(result.texts[:5]):
        print(f"第{i+1}行: {line.text} (置信度: {line.confidence:.2f})")
    
    # 如果需要保存到 output 目录（兼容原有功能）
    engine.save_results(result.raw, "output")
