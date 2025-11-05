# src/offline/chemberta_extract.py
from __future__ import annotations
import argparse, glob, hashlib, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = AutoTokenizer = AutoModel = None

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:24]

def _dtype_of(s: str):
    if torch is None:
        return None
    s = s.lower()
    if s in ("fp16", "float16", "half"): return torch.float16
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float32

def _save_npz(out_dir: Path, key: str, vec: np.ndarray):
    out = out_dir / f"{key}.npz"
    np.savez_compressed(out, chemberta=vec.astype(np.float16))

def _pool_hidden(last_hidden, attn_mask, mode: str):
    """
    last_hidden: (1, T, H); attn_mask: (1, T)
    RoBERTa 类模型：CLS 在索引 0（<s>），序列末尾是 </s>
    """
    if mode == "cls":
        return last_hidden[:, 0, :]  # (1, H)
    # mean pooling（去掉 special token）
    mask = attn_mask.clone()
    mask[:, 0] = 0       # <s>
    mask[:, -1] = 0      # </s>
    m = mask.unsqueeze(-1)                 # (1, T, 1)
    summed = (last_hidden * m).sum(dim=1)  # (1, H)
    denom = m.sum(dim=1).clamp(min=1)      # (1, 1)
    return summed / denom                  # (1, H)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", type=str, required=True,
                    help='例如: "data/BindingDB/fold*_*.csv"')
    ap.add_argument("--out", type=str, required=True,
                    help="输出根目录（脚本会在其下创建数据集子目录）")
    ap.add_argument("--dataset", type=str, default=None,
                    help="可选：BindingDB / DAVIS / BioSNAP；若提供则写到 out/<dataset>/")
    ap.add_argument("--model_dir", type=str, default="/root/lanyun-tmp/hf/ChemBERTa-77M-MLM",
                    help="本地模型目录（推荐）或 HF 仓库名（需联网）")
    ap.add_argument("--offline", action="store_true",
                    help="仅使用本地文件（local_files_only=True）")
    ap.add_argument("--device", type=str,
                    default=("cuda" if (torch and torch.cuda.is_available()) else "cpu"))
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    args = ap.parse_args()

    # —— 先准备输出目录 —— #
    out = Path(args.out)
    if args.dataset:
        out = out / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    # —— 先定义 device / dtype（确保后面加载模型时可用）—— #
    dtype = _dtype_of(args.dtype) if torch else None
    device = args.device
    if torch and device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # —— 收集 CSV —— #
    files = sorted(glob.glob(args.csv_glob))
    if not files:
        print(f"[WARN] No CSV matched: {args.csv_glob}", file=sys.stderr)

    # —— 加载 tokenizer / model —— #
    tok = mdl = None
    if AutoTokenizer is None or AutoModel is None:
        print("[INFO] transformers not available; writing zeros as placeholder")
    else:
        local_only = args.offline or os.path.isabs(args.model_dir)
        try:
            tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=local_only)
            mdl = AutoModel.from_pretrained(args.model_dir, local_files_only=local_only)
            mdl.eval()
            mdl = mdl.to(device)  # ★ 关键：模型放到与输入相同的设备
            print(f"[OK] Loaded model from {args.model_dir} (offline={local_only})", file=sys.stderr)
        except Exception as e:
            # 常见原因：torch<2.6 且模型是 pytorch_model.bin（安全限制）；或本地缺少权重/分词文件
            print(f"[WARN] Failed to load model locally: {e}\n→ Will write zero placeholders.", file=sys.stderr)
            tok = mdl = None

    # —— 逐唯一 SMILES 编码 —— #
    seen = set()
    n_ok = n_zero = 0
    for f in files:
        df = pd.read_csv(f)
        for smi in df["smiles"].astype(str).unique():
            key = _hash(smi)
            if key in seen:
                continue
            seen.add(key)

            if tok is None or mdl is None or torch is None:
                H = 768  # ChemBERTa-77M 默认隐藏维（占位）
                vec = np.zeros((H,), dtype=np.float32)
                n_zero += 1
                _save_npz(out, key, vec)
                continue

            try:
                t = tok(smi, return_tensors="pt", truncation=True, max_length=args.max_len)
                t = {k: v.to(device) for k, v in t.items()}  # ★ 输入张量移到同一设备
                use_amp = (dtype is not None and dtype != torch.float32 and device.startswith("cuda"))
                with torch.no_grad():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            outobj = mdl(**t)
                    else:
                        outobj = mdl(**t)
                    last_hidden = outobj.last_hidden_state         # (1, T, H)
                    pooled = _pool_hidden(last_hidden, t["attention_mask"], args.pool)  # (1, H)
                    if use_amp:
                        pooled = pooled.to(dtype)
                    vec = pooled.squeeze(0).detach().cpu().numpy()  # (H,)
                n_ok += 1
            except Exception as e:
                H = getattr(mdl.config, "hidden_size", 768) if mdl is not None else 768
                vec = np.zeros((H,), dtype=np.float32)
                n_zero += 1
                print(f"[ERR] key={key}: {e} → zero vec", file=sys.stderr)

            _save_npz(out, key, vec)

    print(f"[DONE] saved: {n_ok} real, {n_zero} zeros, out={out}")

if __name__ == "__main__":
    main()
