#!/usr/bin/env bash
# 必要なパッケージをインストール
apt-get update && apt-get install -y \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    poppler-utils

# インストールが成功したか確認
echo "Checking if pdfinfo is installed..."
which pdfinfo || { echo "pdfinfo not found. Installation failed."; exit 1; }
