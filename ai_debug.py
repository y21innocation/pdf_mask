#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI/ルールの判定状況を可視化するデバッグスクリプト。

使い方:
  python3 ai_debug.py /path/to/input.pdf [--provider mock|gemini]

出力:
  - 各ページのテキスト品質, 行数
  - 年収系キーワード行の表示
  - moneyパターンに一致した行の表示
  - AIのスパン検出とラベル
"""
import sys
import os
import json
import fitz
import pdfplumber

from app import (
    group_words_into_lines,
    KEYWORDS_PATTERN,
    MONEY_PATTERN,
    normalize_and_remove_invisible_chars,
    _text_quality_ratio,
)
from ai_mask import AiMasker


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ai_debug.py <pdf_path> [--provider mock|gemini]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    provider = "mock"
    if len(sys.argv) >= 4 and sys.argv[2] == "--provider":
        provider = sys.argv[3]

    if not os.path.exists(pdf_path):
        print("No such file:", pdf_path)
        sys.exit(2)

    doc = fitz.open(pdf_path)
    plumber = pdfplumber.open(pdf_path)

    print(f"PDF: {pdf_path}")
    print(f"Pages: {len(doc)}  Provider: {provider}")

    for pidx in range(len(doc)):
        page = doc[pidx]
        try:
            page_text = plumber.pages[pidx].extract_text() or ""
        except Exception:
            page_text = ""
        quality = _text_quality_ratio(page_text)
        lines = group_words_into_lines(page)

        print("\n=== Page", pidx+1, "===")
        print(f"text_quality={quality:.2f}  lines={len(lines)}")

        # 年収系キーワード行
        for ln in lines:
            t = ln.get('text','')
            if KEYWORDS_PATTERN.search(t):
                print("[KW]", t)

        # moneyパターン一致
        for ln in lines:
            t = ln.get('text','')
            if MONEY_PATTERN.search(t):
                m = MONEY_PATTERN.search(t)
                print("[RULE money]", t, "| matched=", m.group(0))

        # AIスパン
        ai = AiMasker(provider=provider, prompt=os.getenv("AI_PROMPT"), api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), model=os.getenv("AI_MODEL","gemini-1.5-flash"), mask_labels=None)
        rects, logs = ai.mask_page(lines, pidx)
        for _, log in logs:
            print(log)

    plumber.close()
    doc.close()


if __name__ == '__main__':
    main()
