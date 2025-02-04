# ベースイメージとして Python 3.11-slim を使用
FROM python:3.11-slim

# システム依存パッケージをインストール
# poppler-utils: PDF→画像変換に必要
# fonts-ipafont-gothic, fonts-ipafont-mincho, fonts-noto-cjk: 日本語フォント
# locales: ロケール生成用
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    fonts-ipafont-gothic \
    fonts-ipafont-mincho \
    fonts-noto-cjk \
    locales && \
    rm -rf /var/lib/apt/lists/*

# ja_JP.UTF-8 ロケールを生成する
RUN sed -i 's/# ja_JP.UTF-8 UTF-8/ja_JP.UTF-8 UTF-8/' /etc/locale.gen && locale-gen

# 環境変数としてロケールを設定
ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8

# 作業ディレクトリを /app に設定
WORKDIR /app

# requirements.txt をコンテナにコピーし、Pythonパッケージをインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクト内の全ファイルをコンテナにコピー
COPY . /app

# コンテナが利用するポート 5000 を公開
EXPOSE 5000

# コンテナ起動時に Gunicorn を使用してアプリケーションを起動
# ※ app.py 内の Flask アプリが app という変数に格納されている前提です
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
