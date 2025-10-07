import os
import re
import zipfile
import tempfile
import unicodedata
import gc  # メモリ管理用
import time  # パフォーマンス測定用
from flask import Flask, render_template, request, send_from_directory
import fitz  # PyMuPDF
import pdfplumber
from ai_mask import AiMasker
try:
    import pytesseract
except Exception:
    pytesseract = None

# OCR機能をインポート（オプション）
try:
    from ocr_processing import (
        detect_text_regions, 
        extract_text_with_ocr, 
        group_ocr_words_into_lines,
        ocr_detect_salary_patterns
    )
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] OCR functionality not available: {e}")
    OCR_AVAILABLE = False

app = Flask(__name__)

def _is_already_masked_filename(name: str) -> bool:
    base = os.path.basename(name or "")
    return base.startswith("maskedfix_") or base.startswith("masked_")

# ---- 「金額っぽい文字列」を検出する正規表現パターン ----
#  「万」単体でもマッチするように (...|万) を追加
MONEY_PATTERN = re.compile(
        r"""
        (
            # Range: 400〜600万円 / 400-600 万円 / 500万〜600万円 (allow inner spaces/commas)
            (?:[0-9０-９](?:[\s,，\.．]*[0-9０-９])*)\s*[~〜\-–—]\s*(?:[0-9０-９](?:[\s,，\.．]*[0-9０-９])*)\s*(?:万\s*円|万円|円|¥|￥|万)
            |
            # Currency-first: ¥8,000,000 / ￥８００００００ (with optional spaces)
            (?:[¥￥]\s*[0-9０-９](?:[\s,，\.．]*[0-9０-９])*)
            |
            # Digit + unit: 800万円 / 800,000 円 / 800万 / 2億円 / 2億 / 500万 / 600万円
            (?:[0-9０-９](?:[\s,，\.．]*[0-9０-９])*\s*(?:億\s*円|億円|億|万\s*円|万円|円|¥|￥|万))
            |
            # Kanji numerals + unit: 八百万円 / 五十万 / 八千円 / 一億円
            (?:[一二三四五六七八九十百千万億兆〇零]+\s*(?:億\s*円|億円|億|万\s*円|万円|円|万))
            |
            # Mixed: 1億5000万円 / 1億500万円 / 2億円 / 二億三千万円
            (?:
                (?:[0-9０-９](?:[\s,，\.．]*[0-9０-９])*|[一二三四五六七八九十百千万億兆〇零]+)\s*億\s*
                (?:
                    (?:[0-9０-９](?:[\s,，\.．]*[0-9０-９])*|[一二三四五六七八九十百千万〇零]+)\s*万?
                )?
                \s*(?:円)?
            )
            |
            # 追加: 単体数字 + 万 (500万、600万など)
            (?:[0-9０-９]{2,4}\s*万)
            |
            # 追加: 年収範囲 (500万〜、〜600万円など)
            (?:[0-9０-９]{2,4}\s*万\s*[〜~])
            |
            (?:[〜~]\s*[0-9０-９]{2,4}\s*万\s*円?)
        )
        """,
        re.IGNORECASE | re.VERBOSE,
)

# ---- 「年収に関わる記載」かどうかを判定するキーワード ----
KEYWORDS_PATTERN = re.compile(
    r"(年\s*収|希望\s*年\s*収|想定\s*年\s*収|年\s*収\s*目\s*安|直\s*近\s*の\s*年\s*収|現\s*年\s*収|前\s*年\s*収|給\s*与|月\s*給|賞\s*与|月\s*収|年\s*俸|手\s*当|給\s*料|日\s*給|時\s*給|報\s*酬|単\s*価|月\s*額|年\s*額|インセンティブ|年間\s*賞与|年\s*棒|基\s*本\s*給|総\s*額|支\s*給|待\s*遇|年\s*間\s*収\s*入)",
    re.IGNORECASE
)

# ---- 企業情報など年収と関係ない金額を除外するキーワード ----
EXCLUDE_KEYWORDS_PATTERN = re.compile(
    r"(資\s*本\s*金|売\s*上\s*高|売\s*上|資\s*産|負\s*債|純\s*資\s*産|時\s*価\s*総\s*額|企\s*業\s*価\s*値|株\s*式\s*数|株\s*価|取\s*引\s*額|契\s*約\s*金\s*額|投\s*資\s*額|融\s*資\s*額|借\s*入\s*金|預\s*金|残\s*高|口\s*座|従\s*業\s*員\s*数|設\s*立|創\s*業|上\s*場)",
    re.IGNORECASE
)

# 前後何行をマスク対象に含めるか
NEAR_RANGE = 2

# OCR設定
OCR_ENABLED = os.environ.get("OCR_ENABLED", "false").lower() == "true"  # デフォルトで無効
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))
OCR_MIN_CONFIDENCE = int(os.environ.get("OCR_MIN_CONFIDENCE", "30"))
OCR_GARBLED_MIN_RATIO = float(os.environ.get("OCR_GARBLED_MIN_RATIO", "0.7"))

# マスキング戦略
# MASK_STRATEGY: "near"(従来の近傍マスク) / "all"(ページ全体の金額をマスク)
# FALLBACK_MASK_PAGE_IF_NONE: 近傍マスクでヒットがゼロだったページに対して金額の全体マスクをフォールバック実行
MASK_STRATEGY = os.getenv("MASK_STRATEGY", "all").lower()  # デフォルトを "all" に変更
FALLBACK_MASK_PAGE_IF_NONE = os.getenv("FALLBACK_MASK_PAGE_IF_NONE", "1") == "1"
# AI戦略: none / ai / ai+rules
AI_STRATEGY = os.getenv("AI_STRATEGY", "none").lower()
AI_PROVIDER = os.getenv("AI_PROVIDER", "mock").lower()
AI_PROMPT = os.getenv("AI_PROMPT", "Mask monetary amounts (金額, 年収, 月給, 円/万円/¥/￥).")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")
# Prefer GEMINI_API_KEY, fallback to GOOGLE_API_KEY
AI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
AI_MASK_LABELS = os.getenv("AI_MASK_LABELS")  # e.g., "annual_income,monthly_income,bonus"
AI_FALLBACK_TO_RULES_ON_ZERO = os.getenv("AI_FALLBACK_TO_RULES_ON_ZERO", "1") == "1"
ADJACENT_MASK_ALL_DIGITS = os.getenv("ADJACENT_MASK_ALL_DIGITS", "1") == "1"
FALLBACK_STRONG_NUMBER_MASK = os.getenv("FALLBACK_STRONG_NUMBER_MASK", "1") == "1"
FALLBACK_PAIR_NUM_UNIT = os.getenv("FALLBACK_PAIR_NUM_UNIT", "1") == "1"
ESCALATE_MASK_ALL_DIGITS_ON_ZERO = os.getenv("ESCALATE_MASK_ALL_DIGITS_ON_ZERO", "1") == "1"  # デフォルトを有効化
OCR_IF_GARBLED = os.getenv("OCR_IF_GARBLED", "1") == "1"
OCR_GARBLED_MIN_RATIO = float(os.getenv("OCR_GARBLED_MIN_RATIO", "0.35"))
# 文字化けテキストの無差別マスキングを制御
DISABLE_AGGRESSIVE_MASKING = os.getenv("DISABLE_AGGRESSIVE_MASKING", "1") == "1"  # デフォルトで無効化
# OCR設定
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") == "1"
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_MIN_CONFIDENCE = int(os.getenv("OCR_MIN_CONFIDENCE", "30"))

# 強い数値トークン判定（単位がなくても「金額っぽい」）
STRONG_NUMBER_TOKEN_RE = re.compile(r"^(?:[0-9０-９]{1,3}(?:[,，][0-9０-９]{3})+|[0-9０-９]{5,})$")

def normalize_and_remove_invisible_chars(text: str) -> str:
    """
    1) Unicode正規化(NFKC)で形態をそろえる
    2) 不可視文字(ゼロ幅スペースなど)や制御文字を取り除く
    """
    normalized = unicodedata.normalize("NFKC", text)
    invisible_re = re.compile(r'[\u200B-\u200F\uFEFF\u202A-\u202E]')
    cleaned = invisible_re.sub('', normalized)
    return cleaned

def _is_corrupted_text(text: str) -> bool:
    """
    Check if text contains corrupted encoding patterns (like cid:xxx)
    """
    if not text:
        return False
    
    # Check for cid encoding patterns
    cid_pattern = re.compile(r'\(cid:\d+\)')
    if cid_pattern.search(text):
        return True
    
    # Check for excessive non-printable characters
    printable_chars = sum(1 for c in text if c.isprintable())
    total_chars = len(text)
    if total_chars > 0 and printable_chars / total_chars < 0.5:
        return True
    
    # Check for high ratio of control characters  
    control_chars = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
    if total_chars > 0 and control_chars / total_chars > 0.3:
        return True
    
    return False

def force_reconnect_money_string(line: str) -> str:
    """
    例: "年間賞与 110 万円 0.8万円" => "年間賞与 110万円 0.8万円"
         "・月収  420 万 円" => "・月収 420万 円"
    """
    # 数字→単位の余分な空白を除去
    pattern1 = re.compile(r'([0-9０-９][0-9０-９,，\.]*)\s+((万|万円|円|¥|￥))')
    line = pattern1.sub(r'\1\2', line)
    # 単位(通貨)→数字の余分な空白を除去 (¥ 8000000 -> ¥8000000)
    pattern2 = re.compile(r'((¥|￥))\s+([0-9０-９][0-9０-９,，\.]*)')
    line = pattern2.sub(r'\1\3', line)
    return line

def is_number_like(txt: str) -> bool:
    """トークンが「数字主体」ならTrue"""
    return bool(re.match(r'^[0-9０-９,\.]+$', txt))

def is_unit_like(txt: str) -> bool:
    """トークンが「万/億/円/¥/￥」などで始まるならTrue"""
    return bool(re.match(r'^(兆|億円|億|万|万円|円|¥|￥)', txt))

def unify_adjacent_tokens(tokens, base_gap=1.5, extended_gap=10.0):
    """
    tokens: [(x0,y0,x1,y1,text), ...] (x0昇順想定)
      数字と単位がやや離れていても連結するロジックを簡易的に入れる。
    """
    if not tokens:
        return tokens
    
    merged = []
    current = list(tokens[0])  # [x0,y0,x1,y1,text]

    for i in range(1, len(tokens)):
        nxt = tokens[i]
        gap = nxt[0] - current[2]  # x0(次) - x1(現)
        # 数字→単位、または 単位→数字 の並びのときは gap許容をextended_gapへ
        if (is_number_like(current[4]) and is_unit_like(nxt[4])) or \
           (is_unit_like(current[4]) and is_number_like(nxt[4])):
            allow_gap = extended_gap
        else:
            allow_gap = base_gap

        if gap <= allow_gap:
            # 連結
            current[4] = current[4] + nxt[4]
            current[2] = nxt[2]
            current[1] = min(current[1], nxt[1])
            current[3] = max(current[3], nxt[3])
        else:
            merged.append(tuple(current))
            current = list(nxt)
    merged.append(tuple(current))
    return merged

def group_words_into_lines(page, threshold=2.0):
    """
    PyMuPDFの page.get_text("words") から行単位にまとめる。
    """
    words = page.get_text("words")
    if not words:
        return []

    # (y0, x0) 昇順に
    words.sort(key=lambda w: (w[1], w[0]))

    lines = []
    current_line = None
    line_count = 0

    for w in words:
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        txt_clean = normalize_and_remove_invisible_chars(txt)
        if not txt_clean:
            continue
        mid_y = (y0 + y1)/2.0

        if current_line is None:
            current_line = {
                'line_index': line_count,
                'tokens': [(x0,y0,x1,y1,txt_clean)],
                'y': mid_y
            }
        else:
            prev_y = current_line['y']
            if abs(mid_y - prev_y) <= threshold:
                current_line['tokens'].append((x0,y0,x1,y1,txt_clean))
                size_now = len(current_line['tokens'])
                # 行中心yを平均化
                total = prev_y*(size_now-1) + mid_y
                current_line['y'] = total / size_now
            else:
                lines.append(current_line)
                line_count += 1
                current_line = {
                    'line_index': line_count,
                    'tokens': [(x0,y0,x1,y1,txt_clean)],
                    'y': mid_y
                }

    if current_line:
        lines.append(current_line)

    # tokenをx0順に並べ替えて unify
    for ln in lines:
        ln['tokens'].sort(key=lambda i: i[0])
        unified = unify_adjacent_tokens(ln['tokens'])
        ln['tokens'] = unified
        # 行テキスト再構築
        text_parts = []
        for i, item in enumerate(ln['tokens']):
            if i>0: 
                text_parts.append(" ")
            text_parts.append(item[4])
        ln['text'] = "".join(text_parts)

    return lines

def ocr_extract_lines(page) -> list:
    """画像ベースのページ向けにOCRして簡易的な行配列を返す。
    戻り値は group_words_into_lines と同等の [{line_index, tokens:[(x0,y0,x1,y1,text)], text, y}] 形式。
    """
    if pytesseract is None:
        return []
    try:
        dpi = 200
        pix = page.get_pixmap(dpi=dpi)
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        raw = pytesseract.image_to_data(img, lang='jpn', output_type='dict')
    except Exception:
        return []

    n = len(raw.get('text', []))
    # Tesseractの出力(bboxは左上x,y,幅,高さ)
    words = []
    # ページ座標へのスケール変換
    pw = float(pix.width); ph = float(pix.height)
    rw = float(page.rect.width); rh = float(page.rect.height)
    sx = rw / pw if pw else 1.0
    sy = rh / ph if ph else 1.0

    for i in range(n):
        txt = raw['text'][i]
        if not txt or not txt.strip():
            continue
        try:
            x = raw['left'][i]; y = raw['top'][i]; w = raw['width'][i]; h = raw['height'][i]
            # 画像ピクセル -> PDFページ座標
            x0 = float(x) * sx; y0 = float(y) * sy; x1 = (float(x) + float(w)) * sx; y1 = (float(y) + float(h)) * sy
        except Exception:
            continue
        txt = normalize_and_remove_invisible_chars(txt)
        words.append((x0,y0,x1,y1,txt))

    # 簡易に y 中心でクラスタリング
    words.sort(key=lambda w: (w[1], w[0]))
    lines = []
    current = None
    li = 0
    for w in words:
        x0,y0,x1,y1,txt = w
        midy = (y0+y1)/2.0
        if current is None:
            current = {'line_index': li, 'tokens': [w], 'y': midy}
        else:
            if abs(midy - current['y']) <= 6.0:
                current['tokens'].append(w)
                n = len(current['tokens'])
                current['y'] = (current['y']*(n-1) + midy)/n
            else:
                lines.append(current)
                li += 1
                current = {'line_index': li, 'tokens': [w], 'y': midy}
    if current:
        lines.append(current)
    for ln in lines:
        ln['tokens'].sort(key=lambda i: i[0])
        # ここでは結合なし。スペース結合テキストを生成
        text = " ".join(t[4] for t in ln['tokens'])
        ln['text'] = text
    return lines

def _text_quality_ratio(text: str) -> float:
    """ページテキストの中で『意味のある日本語/数字/通貨/単位/記号』が占める割合を返す。"""
    if not text:
        return 0.0
    t = normalize_and_remove_invisible_chars(text)
    total = len(t)
    if total == 0:
        return 0.0
    # 許容文字: 和字、カナ、漢字、数字、通貨記号、単位、句読点/記号など
    ok_re = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF0-9０-９¥￥円万億兆\-~〜,，\.．:：\s\(\)\[\]\{\}/\\・・\+\-]")
    ok = sum(1 for ch in t if ok_re.match(ch))
    return ok / total

# ----------------------------------------------------
#  行ごとのマスク
# ----------------------------------------------------
def mask_line_money_pattern(page, ln, removed_items, page_idx):
    """
    この行(ln)のテキストに金額正規表現を照合し、
    被るトークンのbboxをレダクト。
    ただし、企業情報（資本金、売上高など）は除外する。
    """
    original_line_text = ln['text']
    tokens = ln['tokens']

    # 除外キーワードチェック: 資本金、売上高などが含まれている行はスキップ
    if EXCLUDE_KEYWORDS_PATTERN.search(original_line_text):
        print(f"[DEBUG] SKIP masking line due to exclude keywords: {original_line_text!r}")
        return

    # token => 連結文字列の位置map
    combined_text_parts = []
    index_map = []
    cur_offset = 0
    for i, tk in enumerate(tokens):
        if i>0:
            combined_text_parts.append(" ")
            cur_offset += 1
        start_pos = cur_offset
        combined_text_parts.append(tk[4])
        cur_offset += len(tk[4])
        end_pos = cur_offset
        index_map.append((start_pos, end_pos, tk))
    # 検索対象文字列（トークンを単一空白で連結）
    combined_text = "".join(combined_text_parts)
    search_text = combined_text

    # MONEY_PATTERNで探す
    for match in MONEY_PATTERN.finditer(search_text):
        m_start = match.start()
        m_end   = match.end()
        matched_str = match.group(0)

        # token範囲とマッチ範囲が一部でも被ればOK
        hit_tokens = []
        for (tk_start, tk_end, token_item) in index_map:
            if not (tk_end <= m_start or tk_start >= m_end):
                hit_tokens.append(token_item)

        if hit_tokens:
            min_x = min(t[0] for t in hit_tokens)
            min_y = min(t[1] for t in hit_tokens)
            max_x = max(t[2] for t in hit_tokens)
            max_y = max(t[3] for t in hit_tokens)
            page.add_redact_annot(fitz.Rect(min_x, min_y, max_x, max_y), fill=(1,1,1))

            removed_items.append(
                (page_idx+1,
                 f"[mask money pattern] line=({original_line_text}), matched=({matched_str})")
            )
            print(f"[DEBUG] SALARY line mask => line={original_line_text!r} matched={matched_str!r}")


def mask_line_all_numbers(page, ln, removed_items, page_idx):
    """
    この行(ln)に含まれる「数字トークン」はすべてマスクする。
    """
    line_text = ln['text']
    # 該当の数字トークン
    numeric_tokens = [tk for tk in ln['tokens'] if is_number_like(tk[4])]

    if numeric_tokens:
        for tk in numeric_tokens:
            # token item => (x0,y0,x1,y1, text)
            rect = fitz.Rect(tk[0], tk[1], tk[2], tk[3])
            page.add_redact_annot(rect, fill=(1,1,1))

        removed_items.append(
            (page_idx+1,
             f"[mask all digits in KW-line] line=({line_text}) => masked {len(numeric_tokens)} numeric tokens")
        )
        print(f"[DEBUG] KW line => masked all digits. line=({line_text!r})")


def mask_line_strong_numbers(page, ln, removed_items, page_idx):
    """
    単位なしでも金額っぽい強い数値(5桁以上 or カンマ区切り)のみをマスク。
    フォールバック用。
    """
    line_text = ln['text']
    strong_tokens = [tk for tk in ln['tokens'] if STRONG_NUMBER_TOKEN_RE.match(tk[4])]

    if strong_tokens:
        for tk in strong_tokens:
            rect = fitz.Rect(tk[0], tk[1], tk[2], tk[3])
            page.add_redact_annot(rect, fill=(1,1,1))

        removed_items.append(
            (page_idx+1,
             f"[fallback strong-numbers] line=({line_text}) => masked {len(strong_tokens)} tokens")
        )
        print(f"[DEBUG] Strong numbers fallback => masked {len(strong_tokens)} tokens. line=({line_text!r})")


def fallback_pair_numbers_and_units(page, removed_items, page_idx, y_thresh=3.5, x_thresh=160.0):
    """
    ページ全体で、数値トークンと単位トークン(万/円/¥/￥/億/兆)が近接するペアを見つけてマスクする。
    行分割が崩れた表でも、同一行(±y_thresh)かつx方向で近いものを結合して捕捉する。
    """
    words = page.get_text("words") or []
    if not words:
        return
    # 正規化済みテキストを持つアイテムに変換
    toks = []
    for w in words:
        x0,y0,x1,y1,txt = w[0],w[1],w[2],w[3],normalize_and_remove_invisible_chars(w[4])
        if not txt:
            continue
        toks.append((x0,y0,x1,y1,txt))

    num_toks = [t for t in toks if is_number_like(t[4])]
    unit_re = re.compile(r"^(兆|億|万|万円|円|¥|￥)$")
    unit_toks = [t for t in toks if unit_re.match(t[4])]

    before = len(removed_items)
    for nt in num_toks:
        nx0,ny0,nx1,ny1,ntxt = nt
        nym = (ny0+ny1)/2.0
        # 候補: 同一行近傍
        best = None
        best_dx = 1e9
        for ut in unit_toks:
            ux0,uy0,ux1,uy1,utxt = ut
            uym = (uy0+uy1)/2.0
            if abs(uym - nym) <= y_thresh:
                # 同一行とみなし、x距離で最短を採用
                dx = min(abs(ux0-nx1), abs(nx0-ux1))
                if dx < best_dx and dx <= x_thresh:
                    best_dx = dx
                    best = ut
        if best is None:
            continue
        bx0,by0,bx1,by1,btxt = best
        min_x = min(nx0,bx0); min_y = min(ny0,by0)
        max_x = max(nx1,bx1); max_y = max(ny1,by1)
        page.add_redact_annot(fitz.Rect(min_x,min_y,max_x,max_y), fill=(1,1,1))
        removed_items.append(
            (page_idx+1, f"[fallback pair num-unit] num=({ntxt}) unit=({btxt})")
        )
    if len(removed_items) > before:
        print(f"[DEBUG] Pair num-unit fallback => {len(removed_items)-before} pairs masked on page {page_idx+1}")


def mask_money_in_nearby_salary_lines(page, removed_items, page_idx, page_text: str = None):
    """
    1) 行分割
    2) キーワード行(kw_line)は「すべての数字トークン」をマスク + money_patternマスク
    3) キーワード行の前後NEAR_RANGE行 => money_patternだけマスク
    4) 他はマスクなし
    """
    lines = group_words_into_lines(page)
    if not lines and pytesseract is not None:
        ocr_lines = ocr_extract_lines(page)
        if ocr_lines:
            print(f"[DEBUG] Using OCR lines in rule-based path on page {page_idx+1}")
            lines = ocr_lines
    if not lines:
        return

    # グローバルマスクモード: ページ内の全行に金額パターンを適用
    if MASK_STRATEGY == "all":
        for ln in lines:
            mask_line_money_pattern(page, ln, removed_items, page_idx)
        return

    # キーワード含む行indexを抽出
    kw_line_idxs = set()
    for ln in lines:
        if KEYWORDS_PATTERN.search(ln['text']):
            kw_line_idxs.add(ln['line_index'])

    # 前後N行 => money_pattern
    target_line_idxs = set(kw_line_idxs)
    for idx in kw_line_idxs:
        for off in range(1, NEAR_RANGE+1):
            target_line_idxs.add(idx - off)
            target_line_idxs.add(idx + off)

    # 行ごとにマスク適用
    before_count = len(removed_items)
    for ln in lines:
        original_line_idx = ln['line_index']

        if original_line_idx in kw_line_idxs:
            # (A) キーワード行 => 1)数字全部マスク、2)money_patternマスク
            mask_line_all_numbers(page, ln, removed_items, page_idx)
            mask_line_money_pattern(page, ln, removed_items, page_idx)

        elif original_line_idx in target_line_idxs:
            # (B) 前後N行 => money_patternのみ
            mask_line_money_pattern(page, ln, removed_items, page_idx)
            if ADJACENT_MASK_ALL_DIGITS:
                # 単位が別セルなどで欠落しているケースに対応するため、近傍行の数字トークンもマスク
                mask_line_all_numbers(page, ln, removed_items, page_idx)
        else:
            # (C) それ以外の行 => 何もしない
            pass

    # フォールバック: 近傍マスクでヒットがゼロの場合、ページ全体の金額をマスク
    if FALLBACK_MASK_PAGE_IF_NONE and len(removed_items) == before_count:
        print(f"[DEBUG] No masks from NEAR strategy on page {page_idx+1}. Fallback: mask all money patterns on page.")
        before_page_all = len(removed_items)
        for ln in lines:
            mask_line_money_pattern(page, ln, removed_items, page_idx)
        # それでもゼロなら、数値と単位の近接ペア/強い数値のフォールバックを実行
        if FALLBACK_STRONG_NUMBER_MASK and len(removed_items) == before_page_all:
            if FALLBACK_PAIR_NUM_UNIT:
                print(f"[DEBUG] No money-pattern hits on page {page_idx+1}. Fallback: pair numbers with nearby units.")
                fallback_pair_numbers_and_units(page, removed_items, page_idx)
            if len(removed_items) == before_page_all:
                print(f"[DEBUG] No pair hits on page {page_idx+1}. Fallback: mask strong numeric tokens.")
                for ln in lines:
                    mask_line_strong_numbers(page, ln, removed_items, page_idx)
        # それでも尚ゼロなら、ページ内の全数字トークンをマスク（安全ではないため手動オプトイン）
        if ESCALATE_MASK_ALL_DIGITS_ON_ZERO and len(removed_items) == before_page_all:
            # ページ本文に給与系キーワードがある場合のみ全数字マスクを実施
            allow_escalate = True
            if page_text is not None:
                allow_escalate = bool(KEYWORDS_PATTERN.search(normalize_and_remove_invisible_chars(page_text)))
            if allow_escalate:
                print(f"[DEBUG] Still zero on page {page_idx+1}. Escalate: mask ALL digit tokens on page (keyword present).")
                for ln in lines:
                    mask_line_all_numbers(page, ln, removed_items, page_idx)
            else:
                print(f"[DEBUG] Escalation skipped on page {page_idx+1} (no salary keyword found).")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return "ok", 200

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file')
    if not files:
        return "No files uploaded", 400

    # 既にマスク済みと思われるファイル名は拒否（テキストが除去済みのため再マスク不可）
    bad = [f.filename for f in files if _is_already_masked_filename(f.filename)]
    if bad:
        return (
            "以下のファイルは既に処理済みの可能性があるため、元のPDFをアップロードしてください:\n"+
            "\n".join(bad),
            400,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        if not os.path.exists('output'):
            os.makedirs('output')

        output_pdfs = []
        log_files = []

        for file_ in files:
            in_pdf_path = os.path.join(tmpdir, file_.filename)
            file_.save(in_pdf_path)

            # PDFオープン
            doc = fitz.open(in_pdf_path)
            plumber_pdf = pdfplumber.open(in_pdf_path)

            removed_items = []
            for pindex in range(len(doc)):
                page = doc[pindex]
                pdfp_page = plumber_pdf.pages[pindex]

                # --- デバッグ用ログ（テーブル行 / 非テーブル行） ---
                print("[DEBUG] extract_table_rows_as_strings() called")
                table_settings = {"vertical_strategy":"lines","horizontal_strategy":"lines"}
                raw_table = pdfp_page.extract_table(table_settings=table_settings)
                if raw_table:
                    for i, row in enumerate(raw_table):
                        if row:
                            row_text = " ".join(cell.strip() for cell in row if cell)
                            print(f"[DEBUG]  Table row {i}: {repr(row_text)}")

                print("[DEBUG] extract_non_table_text_as_lines() called")
                full_text = pdfp_page.extract_text()
                if full_text:
                    lines_ = full_text.splitlines()
                    for i, line_ in enumerate(lines_):
                        line_ = normalize_and_remove_invisible_chars(line_.strip())
                        if line_:
                            print(f"[DEBUG]  Non-table line {i}: {repr(line_)}")

                # --- マスキング ---
                # まずOCR必要性を判定
                use_ocr = False
                if OCR_ENABLED and OCR_AVAILABLE:
                    # pytesseractが利用可能かチェック
                    try:
                        import pytesseract
                        # 簡単なテストを実行
                        pytesseract.get_tesseract_version()
                        tesseract_available = True
                    except Exception as e:
                        print(f"[DEBUG] Tesseract not available: {e}")
                        tesseract_available = False
                    
                    if tesseract_available:
                        # テキスト抽出可能性をチェック
                        has_extractable_text = detect_text_regions(page)
                        if not has_extractable_text:
                            print(f"[DEBUG] Page {pindex+1} appears to be image-only, using OCR")
                            use_ocr = True
                        else:
                            # テキスト品質をチェック
                            try:
                                page_text = pdfp_page.extract_text() or ""
                            except Exception:
                                page_text = ""
                            if _text_quality_ratio(page_text) < OCR_GARBLED_MIN_RATIO:
                                print(f"[DEBUG] Page {pindex+1} has poor text quality, using OCR")
                                use_ocr = True
                    else:
                        print(f"[DEBUG] Tesseract not available, skipping OCR for page {pindex+1}")

                # OCR処理
                if use_ocr:
                    print(f"[DEBUG] Processing page {pindex+1} with OCR")
                    
                    try:
                        # OCRでテキスト抽出
                        ocr_results = extract_text_with_ocr(page, dpi=OCR_DPI)
                        print(f"[DEBUG] OCR extracted {len(ocr_results)} text regions")
                        
                        # OCR結果を行にグループ化
                        ocr_lines = group_ocr_words_into_lines(ocr_results)
                        print(f"[DEBUG] OCR grouped into {len(ocr_lines)} lines")
                        
                        # OCRから年収パターンを検出
                        salary_detections = ocr_detect_salary_patterns(ocr_lines)
                        print(f"[DEBUG] OCR detected {len(salary_detections)} salary patterns")
                        
                        # OCR検出結果をマスキング
                        for detection in salary_detections:
                            bbox = detection['bbox']
                            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                            page.add_redact_annot(rect, fill=(1,1,1))
                            
                            removed_items.append(
                                (pindex+1, f"[OCR salary] {detection['text']} in line: {detection['line_text']}")
                            )
                            print(f"[DEBUG] OCR MASKED: {detection['text']} at {bbox}")
                        
                        # OCR処理が完了したら通常処理をスキップ
                        page.apply_redactions()
                        continue
                    except Exception as e:
                        print(f"[DEBUG] OCR processing failed for page {pindex+1}: {e}")
                        print(f"[DEBUG] Falling back to regular text processing")
                        # OCR失敗時は通常処理に続行

                # AI優先（通常のテキスト処理）
                print(f"[DEBUG] Checking AI_STRATEGY: {AI_STRATEGY}")
                if AI_STRATEGY in ("ai", "ai+rules"):
                    print(f"[DEBUG] AI masking enabled on page {pindex+1}")
                    lines = group_words_into_lines(page)
                    print(f"[DEBUG] Extracted {len(lines)} lines from page {pindex+1}")
                    
                    # Check for corrupted text and force OCR if detected
                    force_ocr = False
                    if lines:
                        for line in lines:
                            line_text = line.get('text', '')
                            if _is_corrupted_text(line_text):
                                print(f"[DEBUG] Corrupted text detected on page {pindex+1}: {repr(line_text[:50])}")
                                force_ocr = True
                                break
                    
                    # 文字化けや未抽出時はOCRへ
                    if OCR_IF_GARBLED or force_ocr:
                        try:
                            page_text = pdfp_page.extract_text() or ""
                        except Exception:
                            page_text = ""
                        if _text_quality_ratio(page_text) < OCR_GARBLED_MIN_RATIO or force_ocr:
                            ocr_lines = ocr_extract_lines(page)
                            if ocr_lines:
                                print(f"[DEBUG] Low quality text on page {pindex+1} -> Using OCR lines (AI path)")
                                # Convert OCR lines to the format expected by AI masking
                                lines = []
                                for ocr_line in ocr_lines:
                                    # Create a line structure similar to group_words_into_lines output
                                    lines.append({
                                        'text': ocr_line,
                                        'tokens': [],  # OCR doesn't provide token positions
                                        'bbox': [0, 0, 0, 0]  # Default bbox
                                    })
                    if not lines:  # 画像のみ等でwordsが拾えない場合はOCRにフォールバック
                        ocr_lines = ocr_extract_lines(page)
                        if ocr_lines:
                            print(f"[DEBUG] Using OCR lines on page {pindex+1}")
                            lines = []
                            for ocr_line in ocr_lines:
                                lines.append({
                                    'text': ocr_line,
                                    'tokens': [],
                                    'bbox': [0, 0, 0, 0]
                                })
                    mask_labels = None
                    if AI_MASK_LABELS:
                        mask_labels = {s.strip() for s in AI_MASK_LABELS.split(',') if s.strip()}
                        print(f"[DEBUG] AI mask labels: {mask_labels}")
                    else:
                        print(f"[DEBUG] No AI mask labels configured")
                    print(f"[DEBUG] Creating AiMasker with provider={AI_PROVIDER}")
                    ai = AiMasker(provider=AI_PROVIDER, prompt=AI_PROMPT, api_key=AI_API_KEY, model=AI_MODEL, mask_labels=mask_labels)
                    print(f"[DEBUG] Running AI masking on page {pindex+1} with {len(lines)} lines")
                    rects, ai_logs = ai.mask_page(lines, pindex)
                    print(f"[DEBUG] AI masking found {len(rects)} rectangles and {len(ai_logs)} log entries")
                    for r in rects:
                        page.add_redact_annot(r, fill=(1,1,1))
                    removed_items.extend(ai_logs)
                    if AI_STRATEGY == "ai":
                        # AIのみモード: ヒット0ならルールベースへ自動フォールバック（環境変数で無効化可）
                        if len(rects) == 0 and AI_FALLBACK_TO_RULES_ON_ZERO:
                            print(f"[DEBUG] AI had 0 hits on page {pindex+1}. Fallback to rule-based masking.")
                        else:
                            page.apply_redactions()
                            continue

                # 通常のテキスト処理のためのテキスト取得
                try:
                    page_text = pdfp_page.extract_text() or ""
                except Exception:
                    page_text = ""

                # 文字化けページ向けの積極的マスキング（AI処理後のフォールバック）
                if _text_quality_ratio(page_text) < 0.7:  # 閾値を0.3から0.7に上げて、より多くのページで積極的マスキング
                    print(f"[DEBUG] Low quality text on page {pindex+1} (quality: {_text_quality_ratio(page_text):.2f}) -> Using aggressive masking")
                    
                    # まず通常のlines処理を実行してlinesを定義
                    try:
                        # ページの全テキストをトークンで行にグループ化（正しい引数で呼び出し）
                        lines = group_words_into_lines(page, NEAR_RANGE)
                    except Exception as e:
                        print(f"[DEBUG] Error in line processing: {e}")
                        lines = []
                        # エラーの場合は別の方法で処理
                        try:
                            # 代替方法: pdfplumberからテキストを取得
                            page_text = pdfp_page.extract_text() or ""
                            # 簡単な行分割でダミーのlinesを作成（group_words_into_lines形式に合わせる）
                            text_lines = page_text.split('\n')
                            lines = []
                            for i, line_text in enumerate(text_lines):
                                if line_text.strip():
                                    # group_words_into_lines()の戻り値形式に合わせる
                                    lines.append({
                                        'line_index': i,
                                        'tokens': [(0, i*20, 595, (i+1)*20, line_text.strip())],  # (x0,y0,x1,y1,text)形式
                                        'y': i*20,
                                        'text': line_text.strip()
                                    })
                        except Exception as e2:
                            print(f"[DEBUG] Alternative line processing also failed: {e2}")
                            lines = []
                    
                    # 文字化けテキストからも金額パターンを検出
                    garbled_money_patterns = [
                        r'[0-9０-９][0-9０-９,\.\s]*\s*(?:万\s*円|万円|円|¥|￥|万)',
                        r'[0-9０-９]+\s*(?:万|円)',
                        r'¥\s*[0-9０-９]+',
                        r'[0-9０-９]+\s*¥',
                        r'年収.*?[0-9０-９]+',
                        r'給与.*?[0-9０-９]+',
                        r'月給.*?[0-9０-９]+',
                        r'[0-9０-９]+.*?万',
                        # より積極的な文字化けパターン
                        r'[0-9０-９]{2,3}(?:万|円)',  # 2-3桁数字+万/円
                        r'[0-9０-９]{3,4}(?:万|円)',  # 3-4桁数字+万/円
                        r'[4-8][0-9０-９]{2}(?:万|円)',  # 400-899万円パターン
                        r'[1-9][0-9０-９]{2,3}(?:万|円)',  # 100-9999万円パターン
                        # 記号が混在している場合
                        r'[0-9０-９]+[^\w\s]*(?:万|円)',
                        r'¥[^\w\s]*[0-9０-９]+',
                        # 部分的に文字化けした場合
                        r'[0-9０-９]+.*?[万円]',
                        r'[年収給与月額].*?[0-9０-９]+',
                    ]
                    
                    for line_obj in lines:
                        try:
                            # linesの形式を確認
                            if isinstance(line_obj, dict):
                                line_text = line_obj.get('text', '')
                            else:
                                print(f"[DEBUG] Unexpected line_obj type: {type(line_obj)}")
                                continue
                                
                            if not line_text:
                                continue
                                
                            for pattern in garbled_money_patterns:
                                matches = re.finditer(pattern, line_text, re.IGNORECASE)
                                for match in matches:
                                    for token in line_obj['tokens']:
                                        # token は (x0, y0, x1, y1, text) のタプル形式
                                        if len(token) >= 5:
                                            token_text = token[4]
                                            # マッチ範囲内のトークンをマスキング
                                            if re.search(r'[0-9０-９]+', token_text):
                                                rect = fitz.Rect(token[0], token[1], token[2], token[3])
                                                page.add_redact_annot(rect, fill=(1,1,1))
                                                removed_items.append(
                                                    (pindex+1, f"[GARBLED] {token_text} in line: {line_text[:50]}...")
                                                )
                                                print(f"[DEBUG] GARBLED MASKED: {token_text} at {rect}")
                        except Exception as e3:
                            print(f"[DEBUG] Error processing line in aggressive masking: {e3}")
                            continue
                    
                    page.apply_redactions()
                    continue

                # 全ページ対象のフォールバック: すべての数字を含む文字をマスキング
                # ただし、文字化けの無差別マスキングは制御可能
                if not DISABLE_AGGRESSIVE_MASKING:
                    print(f"[DEBUG] Applying universal fallback masking on page {pindex+1}")
                    try:
                        lines = group_words_into_lines(page, NEAR_RANGE)
                    except Exception as e:
                        print(f"[DEBUG] Error in fallback line processing: {e}")
                        # 代替方法: ページ全体をマスクする
                        try:
                            page_text = pdfp_page.extract_text() or ""
                            if re.search(r'[0-9０-９]', page_text):
                                # ページ全体をマスク
                                page_rect = page.rect
                                page.add_redact_annot(page_rect, fill=(1,1,1))
                                print(f"[DEBUG] Emergency full-page mask applied to page {pindex+1}")
                                removed_items.append((pindex+1, f"[FULL-PAGE] Emergency mask - contains numbers"))
                        except Exception as e2:
                            print(f"[DEBUG] Emergency mask also failed: {e2}")
                        page.apply_redactions()
                        continue
                
                    fallback_masks = 0
                    for line_obj in lines:
                        # Handle different token formats
                        line_texts = []
                        for token in line_obj['tokens']:
                            if isinstance(token, dict):
                                line_texts.append(token.get('text', ''))
                            elif isinstance(token, (list, tuple)) and len(token) >= 5:
                                line_texts.append(token[4])  # text is at index 4
                            else:
                                line_texts.append(str(token))
                        line_text = ' '.join(line_texts)
                        
                        # 任意の数字を含む行をマスキング
                        if re.search(r'[0-9０-９]', line_text):
                            print(f"[DEBUG] Fallback mask line with numbers: '{line_text[:50]}...'")
                            for token in line_obj['tokens']:
                                token_text = ""
                                bbox = [0, 0, 0, 0]
                                
                                if isinstance(token, dict):
                                    token_text = token.get('text', '')
                                    bbox = token.get('bbox', [0, 0, 0, 0])
                                elif isinstance(token, (list, tuple)) and len(token) >= 5:
                                    token_text = token[4]
                                    bbox = [token[0], token[1], token[2], token[3]]
                                
                                if re.search(r'[0-9０-９]', token_text):
                                    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                    page.add_redact_annot(rect, fill=(1,1,1))
                                    removed_items.append(
                                        (pindex+1, f"[FALLBACK] {token_text} in line: {line_text[:50]}...")
                                    )
                                    fallback_masks += 1
                                    print(f"[DEBUG] FALLBACK MASKED: {token_text} at {rect}")
                    
                    print(f"[DEBUG] Applied {fallback_masks} fallback masks on page {pindex+1}")
                    page.apply_redactions()
                    continue
                else:
                    # アグレッシブマスキングが無効の場合、ルールベース処理へ進む
                    print(f"[DEBUG] Aggressive masking disabled, proceeding to rule-based processing on page {pindex+1}")

                # ルールベース
                # ルールでもテキスト品質が悪ければOCR行を利用
                page_text = full_text or ""
                if OCR_IF_GARBLED and _text_quality_ratio(page_text) < OCR_GARBLED_MIN_RATIO:
                    print(f"[DEBUG] Low quality text on page {pindex+1} -> Using OCR lines (rules path)")
                    # mask_money_in_nearby_salary_lines は内部で group_words_into_lines を呼ぶため
                    # シンプルにOCR行を使う別ルートは用意せず、ページ全体fallbackを期待する。
                    # ただし、最低限のmoneyパターンをOCR行にも適用する軽処理を追加。
                    ocr_lines = ocr_extract_lines(page)
                    before = len(removed_items)
                    for ln in ocr_lines:
                        mask_line_money_pattern(page, ln, removed_items, pindex)
                    if len(removed_items) == before:
                        # 何も取れなければ通常ロジックへ（内部fallbackがさらに走る）
                        mask_money_in_nearby_salary_lines(page, removed_items, pindex, page_text)
                else:
                    mask_money_in_nearby_salary_lines(page, removed_items, pindex, page_text)
                page.apply_redactions()

            # 保存
            out_pdf_name = f"maskedfix_{file_.filename}"
            out_pdf_path = os.path.join('output', out_pdf_name)
            doc.save(out_pdf_path, deflate=True, clean=True)
            doc.close()
            plumber_pdf.close()
            
            # メモリクリーンアップ
            gc.collect()

            # ログ出力
            out_log_name = f"masked_{os.path.splitext(file_.filename)[0]}.txt"
            out_log_path = os.path.join('output', out_log_name)
            if removed_items:
                with open(out_log_path, "w", encoding="utf-8") as f:
                    for (pg, txt) in removed_items:
                        f.write(f"Page:{pg}, RedactedText:{txt}\n")
            else:
                open(out_log_path, "w").close()

            output_pdfs.append(out_pdf_path)
            log_files.append(out_log_path)

        # ZIP圧縮
        zip_name = "maskedfix_and_texts.zip"
        zip_path = os.path.join('output', zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for path_ in output_pdfs + log_files:
                zf.write(path_, os.path.basename(path_))

        return send_from_directory('output', zip_name, as_attachment=True)

if __name__=="__main__":
    # AI設定をデバッグ出力
    print(f"Server AI_STRATEGY: {os.getenv('AI_STRATEGY', 'none')}")
    print(f"Server AI_MASK_LABELS: {os.getenv('AI_MASK_LABELS', '')}")
    print(f"Server AI_PROVIDER: {os.getenv('AI_PROVIDER', 'mock')}")
    
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    # リローダは外して安定起動
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)