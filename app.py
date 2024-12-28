# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_from_directory
import os
import re
import pdfplumber
from io import BytesIO
import zipfile
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import tempfile
from reportlab.pdfgen import canvas

app = Flask(__name__)

# ---------------------------------------------------------
# 1) 微小スペースマージ (隣接単語をくっつける)
# ---------------------------------------------------------
def merge_adjacent_words(words, x_threshold=5.0, y_threshold=5.0):
    """
    同じ行近辺かつ x座標の隙間が小さい単語を結合する。
    """
    merged_words = []
    i = 0
    while i < len(words):
        current = words[i]
        text = current['text']
        x0 = current['x0']
        x1 = current['x1']
        top = current['top']
        bottom = current['bottom']

        j = i + 1
        while j < len(words):
            nxt = words[j]
            if (abs(nxt['top'] - top) <= y_threshold) and (abs(nxt['bottom'] - bottom) <= y_threshold):
                gap = nxt['x0'] - x1
                if 0 <= gap <= x_threshold:
                    text += nxt['text']  # 単語を連結
                    x1 = nxt['x1']
                    top = min(top, nxt['top'])
                    bottom = max(bottom, nxt['bottom'])
                    j += 1
                else:
                    break
            else:
                break

        merged_words.append({
            'text': text,
            'x0': x0,
            'x1': x1,
            'top': top,
            'bottom': bottom
        })
        i = j

    return merged_words


# ---------------------------------------------------------
# 2) 行単位にグルーピング
# ---------------------------------------------------------
def group_words_by_line(words, line_threshold=5.0):
    """
    top座標が近い(±line_threshold px以内)の単語を同じ行とみなす。
    戻り値: [ (line_words, line_top, line_bottom), ... ]
    """
    if not words:
        return []

    # top座標でソート
    words_sorted = sorted(words, key=lambda w: w['top'])
    lines = []

    current_line = [words_sorted[0]]
    current_top = words_sorted[0]['top']
    current_bottom = words_sorted[0]['bottom']

    for w in words_sorted[1:]:
        if abs(w['top'] - current_top) <= line_threshold:
            current_line.append(w)
            current_top = min(current_top, w['top'])
            current_bottom = max(current_bottom, w['bottom'])
        else:
            lines.append((current_line, current_top, current_bottom))
            current_line = [w]
            current_top = w['top']
            current_bottom = w['bottom']

    if current_line:
        lines.append((current_line, current_top, current_bottom))

    # 各行の単語を x0で再ソート
    grouped_lines = []
    for (line_words, ltop, lbottom) in lines:
        lw_sorted = sorted(line_words, key=lambda x: x['x0'])
        grouped_lines.append((lw_sorted, ltop, lbottom))

    return grouped_lines


# ---------------------------------------------------------
# 3) キーワードと金額の正規表現
# ---------------------------------------------------------
KEYWORD_PATTERN = re.compile(
    r"(年収|現年収|希望年収|月収|手当|賞与|残業代|最低希望年収|月給|年棒|月額|現在年収)",
    re.IGNORECASE
)

MONEY_PATTERN = re.compile(
    r"\d[\d,\.]*\s*(万|万円|円)",  # 例: 400万, 400万円, 400,000円
    re.IGNORECASE
)


# ---------------------------------------------------------
# 4) マスク抽出
#    → 「キーワードがある行」だけを対象に、行内の「金額トークン」をマスク
# ---------------------------------------------------------
def extract_mask_positions(input_pdf_path):
    mask_positions = []

    with pdfplumber.open(input_pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_width = page.width
            page_height = page.height

            # A) 単語抽出
            raw_words = page.extract_words()
            raw_words = sorted(raw_words, key=lambda w: (round(w['top']), w['x0']))

            # B) 微小スペースマージ
            merged_words = merge_adjacent_words(raw_words, x_threshold=5.0, y_threshold=5.0)

            # C) 行単位にまとめる
            lines = group_words_by_line(merged_words, line_threshold=5.0)

            for (line_words, ltop, lbottom) in lines:
                # 行のテキストを取得
                line_text = " ".join([w['text'] for w in line_words])
                line_text_lower = line_text.lower()

                # 1) 行にキーワードが含まれているか?
                if KEYWORD_PATTERN.search(line_text_lower):
                    # 2) 含まれていれば、この行の中にある金額トークンだけマスク
                    for w in line_words:
                        token_text = w['text']
                        if MONEY_PATTERN.search(token_text):
                            mask_positions.append({
                                "page": page_idx,
                                "x0": w['x0'],
                                "y0": w['top'],
                                "x1": w['x1'],
                                "y1": w['bottom'],
                                "keyword": w['text'],  # デバッグ用
                                "page_width": page_width,
                                "page_height": page_height
                            })

    return mask_positions


# ---------------------------------------------------------
# 5) 画像マスキング
# ---------------------------------------------------------
def mask_images(image_paths, mask_positions):
    masks_by_page = {}
    for pos in mask_positions:
        p = pos['page']
        if p not in masks_by_page:
            masks_by_page[p] = []
        masks_by_page[p].append(pos)

    for page_idx, img_path in enumerate(image_paths):
        with Image.open(img_path) as im:
            draw = ImageDraw.Draw(im)
            if page_idx in masks_by_page:
                for m in masks_by_page[page_idx]:
                    pdf_w = m['page_width']
                    pdf_h = m['page_height']
                    img_w, img_h = im.size

                    scale_x = img_w / pdf_w
                    scale_y = img_h / pdf_h

                    ix0 = m['x0'] * scale_x
                    iy0 = m['y0'] * scale_y
                    ix1 = m['x1'] * scale_x
                    iy1 = m['y1'] * scale_y

                    draw.rectangle([ix0, iy0, ix1, iy1], fill="white")
            im.save(img_path)


# ---------------------------------------------------------
# 6) PDF再生成 (ページサイズを保持)
# ---------------------------------------------------------
def images_to_pdf(image_paths, page_sizes, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=(1, 1))
    for i, image_path in enumerate(image_paths):
        w, h = page_sizes[i]
        c.setPageSize((w, h))
        c.drawImage(image_path, 0, 0, width=w, height=h)
        c.showPage()
    c.save()


# ---------------------------------------------------------
# Flaskアプリ
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file')
    if not files:
        return 'No files uploaded', 400

    with tempfile.TemporaryDirectory() as tmpdir:
        if not os.path.exists('output'):
            os.makedirs('output')

        output_files = []
        text_files = []

        for file in files:
            input_path = os.path.join(tmpdir, file.filename)
            file.save(input_path)

            # 1) マスク対象抽出
            mask_positions = extract_mask_positions(input_path)

            # 2) テキストファイル出力（デバッグ用）
            output_txt_path = os.path.join('output', 'masked_' + file.filename.replace('.pdf', '.txt'))
            with open(output_txt_path, "w", encoding="utf-8") as txtf:
                for m in mask_positions:
                    txtf.write(f"Page: {m['page']+1}, Keyword: {m['keyword']}\n")

            # 3) PDF → PNG
            images = convert_from_path(input_path, dpi=150, fmt='png')
            image_paths = []
            page_sizes = []
            for idx, img in enumerate(images):
                img_path = os.path.join(tmpdir, f"page_{idx}.png")
                img.save(img_path, "PNG")
                image_paths.append(img_path)
                page_sizes.append(img.size)

            # 4) マスク描画
            mask_images(image_paths, mask_positions)

            # 5) PNG → PDF
            output_pdf_path = os.path.join('output', 'maskedfix_' + file.filename)
            images_to_pdf(image_paths, page_sizes, output_pdf_path)

            output_files.append(output_pdf_path)
            text_files.append(output_txt_path)

        # 6) ZIPにまとめて返す
        zip_filename = "maskedfix_and_texts.zip"
        zip_filepath = os.path.join("output", zip_filename)

        try:
            with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                for f in output_files + text_files:
                    zipf.write(f, os.path.basename(f))
        except Exception as e:
            return f"An error occurred while creating the ZIP file: {str(e)}", 500

        return send_from_directory(directory='output', path=zip_filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

