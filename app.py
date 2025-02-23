# -*- coding: utf-8 -*-
import os
import re
import zipfile
import tempfile
from flask import Flask, render_template, request, send_from_directory
import fitz  # PyMuPDF
import pdfplumber

app = Flask(__name__)

# ---- 「金額っぽい文字列」を検出する正規表現パターン ----
MONEY_PATTERN = re.compile(
    r"[0-9０-９][0-9０-９,\.]*\s*(万|万円|円|¥|￥)",
    re.IGNORECASE
)

# ---- 「年収に関わる記載」かどうかを判定するキーワード ----
KEYWORDS_PATTERN = re.compile(r"(年収|給与|月給|賞与|月収|年俸|手当|給料|日給|時給)", re.IGNORECASE)

# ---- 前後何行を「関連行」とみなすか ----
NEAR_RANGE = 2  # 前後2行

def extract_table_rows_as_strings(pdfp_page):
    """
    pdfplumberのextract_table()で抽出したテーブルをログ出力用に取得
    """
    print("[DEBUG] extract_table_rows_as_strings() called")
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }
    raw_table = pdfp_page.extract_table(table_settings=table_settings)
    results = []
    if raw_table:
        for i, row in enumerate(raw_table):
            if not row:
                continue
            row_text = " ".join(cell.strip() for cell in row if cell)
            if row_text.strip():
                results.append(row_text)
                print(f"[DEBUG]  Table row {i}: {repr(row_text)}")
    return results

def extract_non_table_text_as_lines(pdfp_page):
    """
    pdfplumberのextract_text() => 行(改行)ごとログ出力用
    """
    print("[DEBUG] extract_non_table_text_as_lines() called")
    full_text = pdfp_page.extract_text()
    results = []
    if full_text:
        lines = full_text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                results.append(line)
                print(f"[DEBUG]  Non-table line {i}: {repr(line)}")
    return results

def group_words_into_lines(page, threshold=2.0):
    """
    PyMuPDFの page.get_text("words") ですべてのワードを取得し、
    y座標(ベースライン)が近いものを1行としてまとめる。
    
    Returns:
        lines: [
            {
              'line_index': int, 
              'text': "...",
              'tokens': [(x0,y0,x1,y1, token), ...],
              'y': (行の基準y座標)
            },
            ...
        ]
    """
    words = page.get_text("words")
    if not words:
        return []

    # sort by (y0, x0)
    words.sort(key=lambda w: (w[1], w[0]))

    lines = []
    current_line = None
    line_count = 0

    for w in words:
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        mid_y = (y0 + y1) / 2.0

        if current_line is None:
            current_line = {
                'line_index': line_count,
                'tokens': [(x0, y0, x1, y1, txt)],
                'y': mid_y
            }
        else:
            prev_y = current_line['y']
            if abs(mid_y - prev_y) <= threshold:
                # 同じ行とみなす
                current_line['tokens'].append((x0, y0, x1, y1, txt))
                # 行の中心yを平均で更新
                size_now = len(current_line['tokens'])
                total = prev_y * (size_now - 1) + mid_y
                current_line['y'] = total / size_now
            else:
                # 別行とみなす
                lines.append(current_line)
                line_count += 1
                current_line = {
                    'line_index': line_count,
                    'tokens': [(x0, y0, x1, y1, txt)],
                    'y': mid_y
                }

    if current_line is not None:
        lines.append(current_line)

    # lineごとに tokens を x0 でソートし、text を連結
    for ln in lines:
        ln['tokens'].sort(key=lambda i: i[0])  # x0順
        text_parts = []
        for i, item in enumerate(ln['tokens']):
            if i > 0:
                text_parts.append(" ")
            text_parts.append(item[4])
        ln['text'] = "".join(text_parts)

    return lines


def mask_money_in_nearby_salary_lines(page, removed_items, page_idx):
    """
    1) PyMuPDFのwords => 行にまとめる
    2) 年収キーワードが含まれる行 => line_indexを記録
    3) 前後 NEAR_RANGE 行も含めて => 金額マスク対象
    4) 金額(MONEY_PATTERN)にヒットした単語を塗りつぶし
    """
    lines = group_words_into_lines(page)
    if not lines:
        return

    # 1) キーワード行を探す
    keyword_line_idxs = set()
    for ln in lines:
        line_text = ln['text']
        if KEYWORDS_PATTERN.search(line_text):
            keyword_line_idxs.add(ln['line_index'])

    # 2) 前後 NEAR_RANGE 行も対象化
    all_target_line_idxs = set(keyword_line_idxs)
    for idx in keyword_line_idxs:
        for offset in range(1, NEAR_RANGE+1):
            all_target_line_idxs.add(idx - offset)
            all_target_line_idxs.add(idx + offset)

    # 3) 行ごとに金額マスク
    for ln in lines:
        if ln['line_index'] not in all_target_line_idxs:
            continue
        line_text = ln['text']
        # MONEY_PATTERN でヒットした区間をmask
        # => トークン（単語）の範囲を確認して bounding box union
        # 単語オフセット管理
        tokens = ln['tokens']
        # tokens はリスト[(x0,y0,x1,y1, txt), ...]

        # 行テキストを連結しながら start,end を対応付ける
        combined_text = []
        index_map = []
        cur_offset = 0
        for i, tk in enumerate(tokens):
            if i > 0:
                combined_text.append(" ")
                cur_offset += 1
            start = cur_offset
            combined_text.append(tk[4])
            cur_offset += len(tk[4])
            end = cur_offset
            index_map.append((start, end, tk))
        combined_line_text = "".join(combined_text)

        for match in MONEY_PATTERN.finditer(combined_line_text):
            m_start = match.start()
            m_end = match.end()
            matched_str = match.group(0)

            # 重複するトークンを探す
            hit_tokens = []
            for (tk_start, tk_end, item) in index_map:
                if not (tk_end <= m_start or tk_start >= m_end):
                    hit_tokens.append(item)

            if hit_tokens:
                # bounding box union
                min_x = min(i[0] for i in hit_tokens)
                min_y = min(i[1] for i in hit_tokens)
                max_x = max(i[2] for i in hit_tokens)
                max_y = max(i[3] for i in hit_tokens)
                page.add_redact_annot(fitz.Rect(min_x, min_y, max_x, max_y), fill=(1,1,1))

                print(f"[DEBUG] SALARY line mask => line={line_text!r}  matched={matched_str!r}")
                removed_items.append(
                    (page_idx+1,
                     f"[mask near SALARY-lines] line_index={ln['line_index']}, match=({matched_str}), text=({line_text})")
                )

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

            # PyMuPDF (fitz) で PDF を開く
            doc = fitz.open(in_pdf_path)
            # pdfplumber でも開く => テキスト抽出やテーブル抽出ログ出力
            plumber_pdf = pdfplumber.open(in_pdf_path)

            removed_items = []

            for pindex in range(len(doc)):
                page = doc[pindex]
                pdfp_page = plumber_pdf.pages[pindex]

                # (debug) テーブル行/非テーブル行のログ出し
                table_rows = extract_table_rows_as_strings(pdfp_page)
                non_table_lines = extract_non_table_text_as_lines(pdfp_page)

                # ★ 前後2行にキーがあっても金額をマスクするロジック
                mask_money_in_nearby_salary_lines(page, removed_items, pindex)

                page.apply_redactions()

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

        # すべてを ZIP にまとめて返す
        zip_name = "maskedfix_and_texts.zip"
        zip_path = os.path.join('output', zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for path_ in output_pdfs + log_files:
                zf.write(path_, os.path.basename(path_))

        return send_from_directory('output', zip_name, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)