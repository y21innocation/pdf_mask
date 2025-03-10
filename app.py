# -*- coding: utf-8 -*-
import os
import re
import zipfile
import tempfile
import unicodedata
from flask import Flask, render_template, request, send_from_directory
import fitz  # PyMuPDF
import pdfplumber

app = Flask(__name__)

# ---- 「金額っぽい文字列」を検出する正規表現パターン ----
#  「万」単体でもマッチするように (...|万) を追加
MONEY_PATTERN = re.compile(
    r"[0-9０-９][0-9０-９,\.]*\s*(万\s*円|万円|円|¥|￥|万)",
    re.IGNORECASE
)

# ---- 「年収に関わる記載」かどうかを判定するキーワード ----
KEYWORDS_PATTERN = re.compile(
    r"(年\s*収|給\s*与|月\s*給|賞\s*与|月\s*収|年\s*俸|手\s*当|給\s*料|日\s*給|時\s*給|インセンティブ|年間\s*賞与)",
    re.IGNORECASE
)

# 前後何行をマスク対象に含めるか
NEAR_RANGE = 2

def normalize_and_remove_invisible_chars(text: str) -> str:
    """
    1) Unicode正規化(NFKC)で形態をそろえる
    2) 不可視文字(ゼロ幅スペースなど)や制御文字を取り除く
    """
    normalized = unicodedata.normalize("NFKC", text)
    invisible_re = re.compile(r'[\u200B-\u200F\uFEFF\u202A-\u202E]')
    cleaned = invisible_re.sub('', normalized)
    return cleaned

def force_reconnect_money_string(line: str) -> str:
    """
    例: "年間賞与 110 万円 0.8万円" => "年間賞与 110万円 0.8万円"
         "・月収  420 万 円" => "・月収 420万 円"
    """
    pattern = re.compile(r'([0-9０-９][0-9０-９,\.]*)\s+((万|万円|円|¥|￥))')
    replaced = pattern.sub(r'\1\2', line)
    return replaced

def is_number_like(txt: str) -> bool:
    """トークンが「数字主体」ならTrue"""
    return bool(re.match(r'^[0-9０-９,\.]+$', txt))

def is_unit_like(txt: str) -> bool:
    """トークンが「万」「万円」「円」「¥」「￥」などで始まるならTrue"""
    return bool(re.match(r'^(万|万円|円|¥|￥)', txt))

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
        # 数字→単位の並びのときは gap許容をextended_gapへ
        if is_number_like(current[4]) and is_unit_like(nxt[4]):
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

# ----------------------------------------------------
#  行ごとのマスク
# ----------------------------------------------------
def mask_line_money_pattern(page, ln, removed_items, page_idx):
    """
    この行(ln)のテキストに金額正規表現を照合し、
    被るトークンのbboxをレダクト。
    """
    original_line_text = ln['text']
    tokens = ln['tokens']

    # 数字と単位を再連結
    reconnected_line_text = force_reconnect_money_string(original_line_text)

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

    # MONEY_PATTERNで探す
    for match in MONEY_PATTERN.finditer(reconnected_line_text):
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


def mask_money_in_nearby_salary_lines(page, removed_items, page_idx):
    """
    1) 行分割
    2) キーワード行(kw_line)は「すべての数字トークン」をマスク + money_patternマスク
    3) キーワード行の前後NEAR_RANGE行 => money_patternだけマスク
    4) 他はマスクなし
    """
    lines = group_words_into_lines(page)
    if not lines:
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
    for ln in lines:
        original_line_idx = ln['line_index']

        if original_line_idx in kw_line_idxs:
            # (A) キーワード行 => 1)数字全部マスク、2)money_patternマスク
            mask_line_all_numbers(page, ln, removed_items, page_idx)
            mask_line_money_pattern(page, ln, removed_items, page_idx)

        elif original_line_idx in target_line_idxs:
            # (B) 前後N行 => money_patternのみ
            mask_line_money_pattern(page, ln, removed_items, page_idx)
        else:
            # (C) それ以外の行 => 何もしない
            pass


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file')
    if not files:
        return "No files uploaded", 400

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
                mask_money_in_nearby_salary_lines(page, removed_items, pindex)
                page.apply_redactions()

            # 保存
            out_pdf_name = f"maskedfix_{file_.filename}"
            out_pdf_path = os.path.join('output', out_pdf_name)
            doc.save(out_pdf_path, deflate=True, clean=True)
            doc.close()
            plumber_pdf.close()

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
    app.run(host='0.0.0.0', port=5000, debug=True)