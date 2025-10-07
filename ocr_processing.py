"""
OCR機能を提供するモジュール

このモジュールは、画像化されたPDFからテキストを抽出し、
年収情報を検出・マスキングする機能を提供します。
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
import re
import os
from typing import List, Dict, Any, Tuple, Optional


# 日本語OCR用の設定
TESSERACT_CONFIG = r'--oem 3 --psm 6 -l jpn'

# 環境変数でTesseractのパスを設定可能
if os.getenv('TESSERACT_CMD'):
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')


def detect_text_regions(page: fitz.Page) -> bool:
    """
    ページにテキスト領域があるかどうかを検出
    
    Returns:
        True: 通常のテキストが存在（OCR不要）
        False: 画像のみでOCRが必要
    """
    try:
        # PyMuPDFでテキスト抽出を試行
        text = page.get_text()
        
        # 意味のあるテキストが十分あるかチェック
        # 日本語文字、英数字、一般的な記号の割合を計算
        meaningful_chars = len(re.findall(r'[あ-んア-ンー一-龠0-9a-zA-Z\s\.\,\(\)（）]', text))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return False
            
        meaningful_ratio = meaningful_chars / total_chars
        
        # 意味のあるテキストが70%以上 かつ 20文字以上の場合はOCR不要
        return meaningful_ratio >= 0.7 and total_chars >= 20
        
    except Exception as e:
        print(f"[DEBUG] Error detecting text regions: {e}")
        return False


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    OCR精度向上のための画像前処理
    """
    try:
        # PIL ImageをOpenCV形式に変換
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_array
            
        # グレースケール変換
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv
            
        # ノイズ除去
        denoised = cv2.medianBlur(gray, 3)
        
        # コントラスト強化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 二値化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # PIL Imageに戻す
        processed_image = Image.fromarray(binary)
        
        return processed_image
        
    except Exception as e:
        print(f"[DEBUG] Image preprocessing failed: {e}")
        return image


def extract_text_with_ocr(page: fitz.Page, dpi: int = 300) -> List[Dict[str, Any]]:
    """
    OCRを使用してページからテキストと位置情報を抽出
    
    Returns:
        List of dicts containing text and bbox information
    """
    try:
        # ページを画像として取得
        mat = fitz.Matrix(dpi/72, dpi/72)  # DPI変換行列
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # PIL Imageに変換
        image = Image.open(io.BytesIO(img_data))
        
        # 前処理
        processed_image = preprocess_image_for_ocr(image)
        
        # OCRでテキストと位置情報を取得
        ocr_data = pytesseract.image_to_data(
            processed_image, 
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT
        )
        
        # 結果を整理
        ocr_results = []
        page_width = page.rect.width
        page_height = page.rect.height
        img_width = processed_image.width
        img_height = processed_image.height
        
        # 座標変換係数
        x_scale = page_width / img_width
        y_scale = page_height / img_height
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            confidence = int(ocr_data['conf'][i])
            
            # 信頼度が低い、または空のテキストをスキップ
            if confidence < 30 or not text:
                continue
                
            # 座標をPDF座標系に変換
            x = ocr_data['left'][i] * x_scale
            y = ocr_data['top'][i] * y_scale
            w = ocr_data['width'][i] * x_scale
            h = ocr_data['height'][i] * y_scale
            
            ocr_results.append({
                'text': text,
                'bbox': [x, y, x + w, y + h],
                'confidence': confidence,
                'word_num': i
            })
            
        return ocr_results
        
    except Exception as e:
        print(f"[DEBUG] OCR extraction failed: {e}")
        return []


def group_ocr_words_into_lines(ocr_results: List[Dict[str, Any]], line_threshold: float = 10.0) -> List[Dict[str, Any]]:
    """
    OCR結果の単語を行にグループ化
    """
    if not ocr_results:
        return []
        
    # Y座標でソート
    sorted_words = sorted(ocr_results, key=lambda w: w['bbox'][1])
    
    lines = []
    current_line = None
    line_index = 0
    
    for word in sorted_words:
        word_y = word['bbox'][1]
        
        if current_line is None:
            # 最初の行
            current_line = {
                'line_index': line_index,
                'words': [word],
                'y_center': word_y,
                'text': word['text']
            }
        else:
            # 既存の行と同じ高さかチェック
            if abs(word_y - current_line['y_center']) <= line_threshold:
                # 同じ行に追加
                current_line['words'].append(word)
                # Y座標の平均を更新
                n = len(current_line['words'])
                current_line['y_center'] = (current_line['y_center'] * (n-1) + word_y) / n
            else:
                # 新しい行を開始
                lines.append(current_line)
                line_index += 1
                current_line = {
                    'line_index': line_index,
                    'words': [word],
                    'y_center': word_y,
                    'text': word['text']
                }
    
    # 最後の行を追加
    if current_line:
        lines.append(current_line)
    
    # 各行の単語をX座標でソートし、テキストを結合
    for line in lines:
        line['words'].sort(key=lambda w: w['bbox'][0])
        line['text'] = ' '.join(w['text'] for w in line['words'])
        
        # 行のバウンディングボックスを計算
        if line['words']:
            min_x = min(w['bbox'][0] for w in line['words'])
            min_y = min(w['bbox'][1] for w in line['words'])
            max_x = max(w['bbox'][2] for w in line['words'])
            max_y = max(w['bbox'][3] for w in line['words'])
            line['bbox'] = [min_x, min_y, max_x, max_y]
    
    return lines


def ocr_detect_salary_patterns(ocr_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OCR結果から年収パターンを検出
    """
    from app import MONEY_PATTERN, KEYWORDS_PATTERN, EXCLUDE_KEYWORDS_PATTERN
    
    salary_detections = []
    
    for line in ocr_lines:
        line_text = line['text']
        
        # 除外キーワードチェック
        if EXCLUDE_KEYWORDS_PATTERN.search(line_text):
            print(f"[DEBUG] OCR SKIP line due to exclude keywords: {line_text!r}")
            continue
            
        # 年収関連キーワードまたは金額パターンをチェック
        has_salary_keyword = KEYWORDS_PATTERN.search(line_text)
        money_matches = list(MONEY_PATTERN.finditer(line_text))
        
        if has_salary_keyword or money_matches:
            for match in money_matches:
                matched_text = match.group(0)
                
                # マッチした部分の位置を推定（簡易版）
                start_ratio = match.start() / len(line_text) if len(line_text) > 0 else 0
                end_ratio = match.end() / len(line_text) if len(line_text) > 0 else 1
                
                line_bbox = line['bbox']
                line_width = line_bbox[2] - line_bbox[0]
                
                match_x1 = line_bbox[0] + line_width * start_ratio
                match_x2 = line_bbox[0] + line_width * end_ratio
                match_bbox = [match_x1, line_bbox[1], match_x2, line_bbox[3]]
                
                salary_detections.append({
                    'text': matched_text,
                    'line_text': line_text,
                    'bbox': match_bbox,
                    'type': 'salary' if has_salary_keyword else 'money',
                    'confidence': sum(w['confidence'] for w in line['words']) / len(line['words']) if line['words'] else 0
                })
                
                print(f"[DEBUG] OCR SALARY detected: {matched_text!r} in line: {line_text!r}")
    
    return salary_detections


# 必要なimportが不足している場合の追加
try:
    import io
except ImportError:
    pass