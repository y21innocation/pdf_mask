#!/usr/bin/env python3
import fitz
import os

# テスト用PDFを作成
doc = fitz.open()

# ページ1: 通常のテキスト（マスキング対象あり）
page1 = doc.new_page()
text1 = """候補者推薦状

氏名: 田中太郎
年齢: 30歳
現年収: 500万円
希望年収: 600万円

経験概要:
システム開発の経験が豊富で、
プロジェクトマネジメントも可能です。
"""
page1.insert_text((50, 50), text1, fontsize=12)

# ページ2: 画像的なコンテンツ（文字化けテキスト + 金額）
page2 = doc.new_page()
garbled_text = """(cid:4)(cid:5)(cid:6)(cid:7)(cid:8)(cid:9)(cid:10)(cid:11)
年収: 480万円
(cid:31) (cid:31)!" #$%(cid:25)
月給: 35万円
(cid:140)(cid:141)(cid:142)(cid:143)(cid:144)
賞与: 80万円
"""
page2.insert_text((50, 50), garbled_text, fontsize=12)

doc.save('test_pdf.pdf')
doc.close()
print('test_pdf.pdf created successfully')