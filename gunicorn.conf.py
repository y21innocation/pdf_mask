import multiprocessing
import os

# Renderのメモリ制限に配慮した設定
workers = 1  # ワーカー数を1に制限（メモリ節約）
worker_class = "sync"
timeout = 120  # タイムアウトを短縮
keepalive = 2
max_requests = 100  # リクエスト数制限でメモリリーク防止
max_requests_jitter = 10
preload_app = True  # アプリ事前読み込み
worker_tmp_dir = "/dev/shm"  # メモリファイルシステム使用

# Renderの環境変数からポート取得
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# ログ設定
loglevel = "info"
accesslog = "-"
errorlog = "-"

# メモリ使用量を抑制
worker_connections = 50