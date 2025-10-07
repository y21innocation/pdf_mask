import os
import multiprocessing

# メモリ制約環境での動的設定
def get_memory_tier():
    """環境変数または Render プランからメモリティアを判定"""
    memory_tier = os.getenv("MEMORY_TIER", "").lower()
    if memory_tier:
        return memory_tier
    
    # Render環境の場合、RAMからティアを推定
    render_service = os.getenv("RENDER_SERVICE_NAME")
    if render_service:
        return "starter"  # デフォルトでStarterプランとして扱う
    
    return "local"

memory_tier = get_memory_tier()
print(f"[INFO] Detected memory tier: {memory_tier}")

# メモリティア別設定
if memory_tier == "starter":
    # Starter Plan: 2GB RAM - 超保守的設定
    workers = 1
    worker_class = "sync" 
    worker_connections = 100
    max_requests = 50  # 頻繁にワーカー再起動
    max_requests_jitter = 10
    timeout = 240
    keepalive = 2
    preload_app = False  # メモリ共有を避ける
elif memory_tier == "standard":
    # Standard Plan: 4GB RAM
    workers = min(2, multiprocessing.cpu_count())
    worker_class = "sync"
    worker_connections = 250
    max_requests = 200
    max_requests_jitter = 50
    timeout = 300
    keepalive = 2
    preload_app = True
else:
    # Local/Premium
    workers = min(4, multiprocessing.cpu_count())
    worker_class = "sync"
    worker_connections = 500
    max_requests = 1000
    max_requests_jitter = 200
    timeout = 300
    keepalive = 2
    preload_app = True

# 共通設定
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

# ログ設定
accesslog = "-"
errorlog = "-"
loglevel = "info"

print(f"[INFO] Gunicorn config - Workers: {workers}, Max requests: {max_requests}, Timeout: {timeout}s")
