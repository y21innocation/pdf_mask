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
#   - 半角数字 [0-9] / 全角数字 [０-９]
#   - 途中のカンマ,ドットも許容
#   - 「万」「万円」「円」「¥」などを含む
MONEY_PATTERN = re.compile(
    r"[0-9０-９][0-9０-９,\.]*\s*(万|万円|円|¥)",
    re.IGNORECASE
)

# ---- 「年収に関わる記載」かどうかを判定するキーワード ----
#     例: 年収 / 給与 / 月給 / 賞与 など
#     必要に応じて増減してください
KEYWORDS_PATTERN = re.compile(r"(年収|給与|月給|賞与|月収|年俸|手当|給料|日給|時給)", re.IGNORECASE)


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


def mask_money_in_salary_lines_by_words(page, removed_items, page_idx):
    """
    1) PyMuPDFの page.get_text("words") ですべてのワードを取得
    2) y座標（ベースライン）が近いワード同士を同じ「行」としてグループ化
    3) 各「行テキスト」を連結し、その行が「年収/給与等のキーワード」を含むかをチェック
       => 含む行だけ、金額の正規表現(MONEY_PATTERN)にマッチする単語をマスキング
    4) 該当マッチに被る単語（複数の場合もある）をまとめて bounding box で塗りつぶし
    5) ログには「該当行(line_text)」＋「実際にマッチした金額文字列」を記録
    """

    words = page.get_text("words")  # [x0, y0, x1, y1, "text", block_no, line_no, word_no]
    if not words:
        return

    # 1) ソート: y0(上端)が小さい順でソート
    words.sort(key=lambda w: (w[1], w[0]))  # (y0, x0) の順
    
    lines = []
    threshold = 2.0  # 同じ行とみなす y 座標差
    current_line = None

    for w in words:
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        mid_y = (y0 + y1) / 2.0

        if current_line is None:
            # 最初の行を新規作成
            current_line = {
                'y': mid_y,
                'items': [(x0, y0, x1, y1, txt)]
            }
        else:
            prev_y = current_line['y']
            if abs(mid_y - prev_y) <= threshold:
                # 同じ行とみなす
                current_line['items'].append((x0, y0, x1, y1, txt))
                # 行の中心yを平均で更新
                total = prev_y * (len(current_line['items']) - 1) + mid_y
                current_line['y'] = total / len(current_line['items'])
            else:
                # 新しい行
                lines.append(current_line)
                current_line = {
                    'y': mid_y,
                    'items': [(x0, y0, x1, y1, txt)]
                }

    # 最後の行を追加
    if current_line is not None:
        lines.append(current_line)

    # 2) 行ごとにワードを x 座標順にソート => line_text を作る
    for line_obj in lines:
        line_obj['items'].sort(key=lambda i: i[0])  # x0順でソート

        # 行テキストおよびトークンの文字オフセット管理
        line_text_parts = []
        index_map = []  # [(start, end, (x0,y0,x1,y1, txt)), ... ]
        current_offset = 0

        for item in line_obj['items']:
            token = item[4]
            # 単語間にスペースを1つ入れる
            if line_text_parts:
                line_text_parts.append(" ")
                current_offset += 1

            start_idx = current_offset
            line_text_parts.append(token)
            current_offset += len(token)
            end_idx = current_offset

            index_map.append((start_idx, end_idx, item))

        line_text = "".join(line_text_parts)

        # 3) 「年収/給与」などキーワードを含む行だけ処理
        if not KEYWORDS_PATTERN.search(line_text):
            # 年収に関わるキーワードが無い行 => スキップ
            continue

        # (行をログなどで参照したい場合はここに出力可)
        # print(f"[DEBUG] line has '年収/給与' => {line_text}")

        # 4) line_text から正規表現で金額部分だけ探し、マッチ範囲に被る単語の bbox を union
        for match in MONEY_PATTERN.finditer(line_text):
            match_start = match.start()
            match_end = match.end()
            matched_str = match.group(0)

            # このマッチ範囲に一部でも被るトークンを取得
            hit_tokens = []
            for (tk_start, tk_end, tk_item) in index_map:
                # トークンが [tk_start, tk_end) と [match_start, match_end) で1文字以上重なる？
                if not (tk_end <= match_start or tk_start >= match_end):
                    hit_tokens.append(tk_item)

            if hit_tokens:
                # bbox の union
                min_x = min(t[0] for t in hit_tokens)
                min_y = min(t[1] for t in hit_tokens)
                max_x = max(t[2] for t in hit_tokens)
                max_y = max(t[3] for t in hit_tokens)

                # PDF 上に redact 注釈を追加
                page.add_redact_annot(fitz.Rect(min_x, min_y, max_x, max_y), fill=(1,1,1))

                # ログ出力用に
                removed_items.append(
                    (
                        page_idx + 1,
                        f"[mask in SALARY-line] line=({line_text}), matched=({matched_str.strip()})"
                    )
                )
                print(f"[DEBUG] SALARY line mask => line={line_text!r}  matched={matched_str!r}")


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
            # pdfplumber でも開く => テキスト抽出やテーブル抽出
            plumber_pdf = pdfplumber.open(in_pdf_path)

            removed_items = []  # (page_num, log_message) を append していく

            for pindex in range(len(doc)):
                page = doc[pindex]
                pdfp_page = plumber_pdf.pages[pindex]

                # ★ デバッグやログ用: テーブル行/非テーブル行を出力する
                table_rows = extract_table_rows_as_strings(pdfp_page)
                non_table_lines = extract_non_table_text_as_lines(pdfp_page)

                # ★「行テキストに年収/給与キーワードがあれば、金額だけマスク」処理
                mask_money_in_salary_lines_by_words(page, removed_items, pindex)

                # レダクション適用
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

        # ZIP圧縮して返す
        zip_name = "maskedfix_and_texts.zip"
        zip_path = os.path.join('output', zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for path_ in output_pdfs + log_files:
                zf.write(path_, os.path.basename(path_))

        return send_from_directory('output', zip_name, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)