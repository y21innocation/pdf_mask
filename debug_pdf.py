#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDFの内容を詳細に分析するためのデバッグスクリプト
"""
import re
import sys
import unicodedata
import fitz  # PyMuPDF
import pdfplumber

def normalize_and_remove_invisible_chars(text: str) -> str:
    """
    Unicode正規化と不可視文字の除去
    """
    normalized = unicodedata.normalize("NFKC", text)
    invisible_re = re.compile(r'[\u200B-\u200F\uFEFF\u202A-\u202E]')
    cleaned = invisible_re.sub('', normalized)
    return cleaned

def analyze_pdf(pdf_path):
    print(f"=== PDF分析: {pdf_path} ===\n")
    
    # 現在のパターン
    MONEY_PATTERN = re.compile(
        r"[0-9０-９][0-9０-９,\.]*\s*(万\s*円|万円|円|¥|￥|万)",
        re.IGNORECASE
    )
    
    KEYWORDS_PATTERN = re.compile(
        r"(年\s*収|給\s*与|月\s*給|賞\s*与|月\s*収|年\s*俸|手\s*当|給\s*料|日\s*給|時\s*給|インセンティブ|年間\s*賞与)",
        re.IGNORECASE
    )
    
    # PyMuPDFで分析
    print("--- PyMuPDF分析 ---")
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc):
        print(f"\nページ {page_idx + 1}:")
        
        # テキスト抽出
        text = page.get_text()
        print(f"全文テキスト:\n{repr(text)}\n")
        
        # 行ごとに分析
        lines = text.splitlines()
        for line_idx, line in enumerate(lines):
            line_clean = normalize_and_remove_invisible_chars(line.strip())
            if line_clean:
                print(f"行 {line_idx}: {repr(line_clean)}")
                
                # キーワードチェック
                if KEYWORDS_PATTERN.search(line_clean):
                    print(f"  → キーワード検出!")
                
                # 金額パターンチェック
                money_matches = MONEY_PATTERN.findall(line_clean)
                if money_matches:
                    print(f"  → 金額パターン検出: {money_matches}")
                
                # 数字を含む場合の詳細
                if re.search(r'[0-9０-９]', line_clean):
                    print(f"  → 数字を含むテキスト")
        
        # Wordsで分析
        words = page.get_text("words")
        print(f"\nWords抽出結果 ({len(words)}個):")
        for word in words[:20]:  # 最初の20個のみ表示
            x0, y0, x1, y1, txt, block_no, line_no, word_no = word
            txt_clean = normalize_and_remove_invisible_chars(txt)
            print(f"  {repr(txt_clean)} at ({x0:.1f}, {y0:.1f})")
    
    doc.close()
    
    # pdfplumberで分析
    print("\n--- pdfplumber分析 ---")
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            print(f"\nページ {page_idx + 1}:")
            
            # テキスト抽出
            text = page.extract_text()
            if text:
                print(f"pdfplumber全文:\n{repr(text)}\n")
                
                lines = text.splitlines()
                for line_idx, line in enumerate(lines):
                    line_clean = normalize_and_remove_invisible_chars(line.strip())
                    if line_clean:
                        print(f"行 {line_idx}: {repr(line_clean)}")
                        
                        # パターンマッチング
                        if KEYWORDS_PATTERN.search(line_clean):
                            print(f"  → キーワード検出!")
                        
                        money_matches = MONEY_PATTERN.findall(line_clean)
                        if money_matches:
                            print(f"  → 金額パターン検出: {money_matches}")
            
            # テーブル分析
            tables = page.extract_tables()
            if tables:
                print(f"\nテーブル検出: {len(tables)}個")
                for table_idx, table in enumerate(tables):
                    print(f"テーブル {table_idx}:")
                    for row_idx, row in enumerate(table[:5]):  # 最初の5行のみ
                        if row:
                            row_text = " ".join(str(cell).strip() if cell else "" for cell in row)
                            print(f"  行 {row_idx}: {repr(row_text)}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "BizResume_Ja_BU6033624.pdf"
    analyze_pdf(target)