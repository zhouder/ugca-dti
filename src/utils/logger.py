import sys, time, json

def log_kv(**kwargs):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{ts}] {json.dumps(kwargs, ensure_ascii=False)}\n")
    sys.stdout.flush()
