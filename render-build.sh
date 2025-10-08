#!/usr/bin/env bash
# 必要なパッケージをインストール
apt-get update && apt-get install -y \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-jpn-vert

# インストールが成功したか確認
echo "Checking if pdfinfo is installed..."
which pdfinfo || { echo "pdfinfo not found. Installation failed."; exit 1; }

echo "Checking if tesseract is installed..."
which tesseract || { echo "tesseract not found. Installation failed."; exit 1; }

# アプリケーションを起動（RenderのPORT変数を使用）
gunicorn app:app --workers=2 --timeout=300 --worker-class=sync --threads=1 --bind 0.0.0.0:$PORT
