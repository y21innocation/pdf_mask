#!/usr/bin/env python3
"""
AIマスキングシステムの動作フローを詳細に解説・デバッグ
"""

def explain_ai_masking_flow():
    """現在のAIマスキングシステムの動作を詳しく解説"""
    
    print("🤖 PDF年収マスキングシステム - AI処理フロー詳細解説")
    print("=" * 60)
    
    print("\n1️⃣ **AI戦略判定**")
    print("   環境変数 AI_STRATEGY の値による分岐:")
    print("   • 'ai' → AI単体でマスキング")
    print("   • 'ai+rules' → AI + ルールベース併用")
    print("   • 'none' → 従来のルールベースのみ")
    
    print("\n2️⃣ **AI処理の実行順序**")
    print("   ①テキスト抽出: PyMuPDF page.get_text('words')")
    print("   ②行グループ化: group_words_into_lines()")
    print("   ③文字化け検出: _is_corrupted_text()")
    print("   ④低品質判定: _text_quality_ratio() < 0.35")
    print("   ⑤AI呼び出し: AiMasker.mask_page()")
    
    print("\n3️⃣ **AIプロバイダーの種類**")
    print("   • 'mock' → ローカル処理、ネットワーク不要")
    print("   • 'gemini' → Google Gemini API使用")
    
    print("\n4️⃣ **Gemini AIの処理内容**")
    print("   ①テキスト行を Gemini に送信")
    print("   ②プロンプト: '年収、月給、賞与などの金額をマスク'")
    print("   ③Gemini が金額箇所を特定")
    print("   ④座標を返してマスキング四角形作成")
    
    print("\n5️⃣ **現在発生している問題**")
    print("   ❌ AIマスキングが実行されていない")
    print("   ❌ 代わりにアグレッシブマスキングが動作")
    print("   ❌ removed_items への記録が空")
    
    print("\n6️⃣ **問題の原因候補**")
    print("   • AI_STRATEGY が正しく設定されていない")
    print("   • group_words_into_lines() でエラー発生")
    print("   • AiMasker の初期化に失敗")
    print("   • Gemini API通信エラー")
    
    print("\n🔧 **デバッグ方法**")
    print("   1. 環境変数の確認")
    print("   2. AI処理パスの詳細ログ出力")
    print("   3. Gemini APIの直接テスト")

def debug_current_settings():
    """現在の設定を詳細に確認"""
    import os
    
    print("\n📋 **現在の設定確認**")
    print("-" * 40)
    
    settings = [
        ("AI_STRATEGY", os.getenv("AI_STRATEGY", "none")),
        ("AI_PROVIDER", os.getenv("AI_PROVIDER", "mock")),
        ("AI_PROMPT", os.getenv("AI_PROMPT", "default")),
        ("GEMINI_API_KEY", "設定済み" if os.getenv("GEMINI_API_KEY") else "未設定"),
        ("OCR_IF_GARBLED", os.getenv("OCR_IF_GARBLED", "1")),
        ("OCR_GARBLED_MIN_RATIO", os.getenv("OCR_GARBLED_MIN_RATIO", "0.35")),
    ]
    
    for key, value in settings:
        print(f"   {key}: {value}")
    
    print(f"\n💡 **判定結果**")
    ai_strategy = os.getenv("AI_STRATEGY", "none")
    if ai_strategy in ("ai", "ai+rules"):
        print(f"   ✅ AI処理が有効 (戦略: {ai_strategy})")
        
        provider = os.getenv("AI_PROVIDER", "mock")
        if provider == "gemini":
            if os.getenv("GEMINI_API_KEY"):
                print(f"   ✅ Gemini API使用準備完了")
            else:
                print(f"   ❌ GEMINI_API_KEY が未設定")
        else:
            print(f"   ⚠️ Mock プロバイダー使用中")
    else:
        print(f"   ❌ AI処理無効 (現在: {ai_strategy})")

if __name__ == "__main__":
    explain_ai_masking_flow()
    debug_current_settings()