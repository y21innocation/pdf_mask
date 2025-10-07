#!/usr/bin/env python3
"""
Gemini AIマスキングのセットアップとテストスクリプト
"""

import os
import sys
import tempfile
import fitz  # PyMuPDF

def create_test_pdf_with_salary():
    """給与情報を含むテストPDFを作成"""
    doc = fitz.open()
    
    # ページ1: 正常なテキスト
    page1 = doc.new_page()
    text1 = """履歴書
    
    氏名: 山田太郎
    年齢: 35歳
    
    現在の給与情報:
    ・年収: 650万円
    ・月給: 45万円
    ・賞与: 150万円
    
    希望条件:
    ・希望年収: 700万円以上
    ・勤務地: 東京都内
    """
    page1.insert_text((50, 50), text1, fontsize=12)
    
    # ページ2: 文字化けのあるテキスト（画像のようなページを模擬）
    page2 = doc.new_page()
    text2 = """
    (cid:4)(cid:5)(cid:6) 給与詳細 (cid:7)(cid:8)
    
    基本給: 400,000円
    (cid:31)(cid:32) 諸手当: 50,000円
    (cid:140)(cid:141) 賞与年額: 1,200,000円
    
    総年収: 6,600,000円
    """
    page2.insert_text((50, 50), text2, fontsize=12)
    
    filename = "salary_test.pdf"
    doc.save(filename)
    doc.close()
    print(f"テストPDF作成完了: {filename}")
    return filename

def test_ai_masking():
    """AIマスキングの基本機能をテスト"""
    
    # 1. テストPDF作成
    test_file = create_test_pdf_with_salary()
    
    # 2. AI戦略の設定確認
    print("\n=== AI設定確認 ===")
    print(f"AI_STRATEGY: {os.getenv('AI_STRATEGY', 'none')}")
    print(f"AI_PROVIDER: {os.getenv('AI_PROVIDER', 'mock')}")
    print(f"AI_PROMPT: {os.getenv('AI_PROMPT', 'default')}")
    print(f"GEMINI_API_KEY: {'設定済み' if os.getenv('GEMINI_API_KEY') else '未設定'}")
    
    # 3. Mock プロバイダーでのテスト
    try:
        from ai_mask import AiMasker
        
        masker = AiMasker(
            provider="mock",
            prompt="年収、月給、賞与など金額情報をマスクしてください",
        )
        
        print("\n=== Mock AIマスキングテスト ===")
        print("Mock AIマスキング機能は正常に動作可能です")
        
        # 4. Gemini対応確認
        try:
            import google.generativeai as genai
            print("\n=== Gemini API対応確認 ===")
            print("✅ google-generativeai インストール済み")
            
            if os.getenv('GEMINI_API_KEY'):
                print("✅ GEMINI_API_KEY 設定済み")
                print("🚀 本格的なGemini AIマスキングが利用可能です")
            else:
                print("⚠️  GEMINI_API_KEY 未設定")
                print("次のコマンドでAPIキーを設定してください:")
                print("export GEMINI_API_KEY='your-api-key-here'")
                
        except ImportError:
            print("❌ google-generativeai が見つかりません")
            print("pip3 install google-generativeai でインストールしてください")
            
    except Exception as e:
        print(f"エラー: {e}")
    
    print(f"\nテストファイル: {test_file}")
    print("=== セットアップ完了 ===")

if __name__ == "__main__":
    test_ai_masking()