#!/usr/bin/env python3
"""
実際の画像ベースPDF（文字化けテキスト）を作成してGemini AIテストを実行
"""

import fitz
import os

def create_realistic_image_pdf():
    """実際の画像ベースPDFに近い文字化けテキストでテストファイルを作成"""
    doc = fitz.open()
    
    # ページ1: 比較的正常なテキスト
    page1 = doc.new_page()
    text1 = """候補者推薦状
    
氏名: 白井 雅紀(シライ マサキ)
性別: 男性  
年齢: 42歳

■待遇面
・現年収:480万円
・希望年収:480万円~500万円

■最短ご入社可能時期
ご内定後2ヶ月ほど
    """
    page1.insert_text((50, 50), text1, fontsize=12)
    
    # ページ2: 文字化けの激しいテキスト（画像OCRの失敗例）
    page2 = doc.new_page()
    text2 = """(cid:4)(cid:5)(cid:6)(cid:7)(cid:8)(cid:9)(cid:10)(cid:11)
(cid:12)(cid:13)(cid:12)(cid:14)(cid:15)(cid:13)(cid:16)(cid:17)(cid:18)(cid:11) @(cid:25)A+B(cid:8)CDE.F'GH(cid:21)#I&1J-KLMNO(cid:11)

年収: 480万円
(cid:31) (cid:31)!" #$%(cid:25) (cid:157)0&y-./02345678(cid:139)(cid:25)
月給: 35万円
(cid:140)(cid:141)(cid:142)(cid:143)(cid:144)(cid:6)(cid:145)*(cid:30)+(cid:146)hiO(cid:11)

賞与: 80万円
(cid:253)(cid:254)(cid:30)(cid:8)!(cid:255)˛(cid:0)*lJ-KL(cid:1)(cid:2)O~3.(cid:3)&Ø-(cid:210)(cid:6)

基本給: 400,000円
(cid:31)EF!(cid:8)$GF(cid:31)HIJK:LMNOPQRSTFUV(cid:6)WX(cid:31)YZ
    """
    page2.insert_text((50, 50), text2, fontsize=10)
    
    # ページ3: 極度に文字化けしたテキスト
    page3 = doc.new_page()  
    text3 = """(cid:148)(cid:149)(cid:150)(cid:151)>?@M&(cid:152)(cid:153)(cid:6)(cid:154)(cid:134)vw (cid:155)(cid:12)(cid:13)1(cid:10)(cid:15)(cid:156)Dw31(cid:157)(cid:16)N(cid:127)(cid:128)D(cid:158)(cid:159)(cid:160)¡¢£⁄¥ƒ§¤

年俸制: 600万円

(cid:31)5‚„"(cid:127)(cid:128)$>?@5‚(cid:6)»…(cid:155)(cid:15) 1(cid:13)p(cid:156):‰(cid:190)(cid:155)(cid:15) 1p(cid:156)(cid:6)¿(cid:192)D`‚(cid:20)(cid:138)(cid:139):´ˆ˜^(cid:143)

総支給額: 650万円

fl(cid:20)(cid:138)(cid:139):¯˘acd:⁄¥˙_¨cd:´_¡(cid:31)(cid:201)+\'?cd:˚(cid:160)"(cid:6)„
""#:´ˆ˜^¸(cid:204)(cid:6)8˝:(cid:128)(cid:20)(cid:6)˛ˇ—(cid:209)(cid:210)(cid:211):¥(cid:212)?‹{G(cid:213)(cid:214)JK:
(cid:215)(ˆ?^(cid:6)(cid:216)(cid:217)(cid:218)(cid:219)(cid:23)(cid:24)(cid:31)8˝c!D(cid:158)(cid:220)cd(cid:31)(cid:221)(cid:222)c!D(cid:11)
    """
    page3.insert_text((50, 50), text3, fontsize=9)
    
    filename = "image_based_test.pdf"
    doc.save(filename)
    doc.close()
    print(f"実画像ベースPDF作成完了: {filename}")
    return filename

def test_gemini_image_processing():
    """Gemini AIで画像ベースPDFをテスト"""
    
    # 1. テストPDF作成
    test_file = create_realistic_image_pdf()
    
    # 2. Gemini処理
    print("\n🚀 Gemini AI 画像マスキングテスト開始...")
    
    cmd = f"""
curl -X POST -F "file=@{test_file}" \
http://127.0.0.1:5006/upload \
-o gemini_image_result.zip \
-H "User-Agent: PDF-Mask-Gemini-Test"
    """.strip()
    
    print(f"実行コマンド:\n{cmd}")
    
    # 3. 結果表示
    print(f"\nテストファイル: {test_file}")
    print("Gemini AIでの画像ベースPDFマスキングテストが準備完了しました。")
    print("\n次のコマンドでテストを実行してください:")
    print(f"curl -X POST -F 'file=@{test_file}' http://127.0.0.1:5006/upload -o gemini_image_result.zip")

if __name__ == "__main__":
    test_gemini_image_processing()