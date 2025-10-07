import multiprocessing
import os

# メモリ制限に応じた動的設定
memory_tier = os.getenv("MEMORY_TIER", "free")  # free, starter, pro

if memory_tier == "free":
    # 無料プラン: 512MB制限
    workers = 1
    worker_class = "sync"
    timeout = 120
    max_requests = 100
elif memory_tier == "starter":
    # Starter: 2GB
    workers = 2
    worker_class = "sync"
    timeout = 180
    max_requests = 200
else:
    # Pro以上: 4GB+
    workers = min(4, multiprocessing.cpu_count())
    worker_class = "sync"
    timeout = 300
    max_requests = 500

keepalive = 2
max_requests_jitter = 10
preload_app = True
worker_tmp_dir = "/dev/shm"

# Renderの環境変数からポート取得
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# ログ設定
loglevel = "info"
accesslog = "-"
errorlog = "-"

# メモリ使用量を抑制
worker_connections = 50