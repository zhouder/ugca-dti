# UGCA-DTI（Binary DTI, 非 DTA）

本仓库严格按实现文档组织：三数据集（DAVIS/BindingDB/BioSNAP）、五折评测（fold{1..5}_train/test.csv）、UGCA+MUTAN、两阶段训练与指标汇总。

## 快速开始
1. 安装依赖
   ```bash
   conda create -n ugca-dti python=3.10 
   conda activate ugca-dti
   pip install -r requirements.txt
   ```
2. 准备数据
   - 将 `fold{1..5}_train.csv` 与 `fold{1..5}_test.csv` 放入 `data/{DAVIS|BindingDB|BioSNAP}/`。
   - CSV 必含列：`smiles,protein,label`（0/1）。
3. 可选：一次性离线抽取（建议首次运行）
   - ems-2
   ```bash
   python src/offline/esm2_extract.py \
   --csv_glob "data/DAVIS/fold*_*.csv" \
   --out /root/lanyun-tmp/cache/esm2 \
   --dataset DAVIS \
   --model_dir /root/lanyun-tmp/hf/esm2_t33_650M_UR50D \
   --offline \
   --device cuda \
   --dtype fp16 \
   --chunk_len 1000
   ```
   - chemberta（目前是384维，后面可以用768维）
   ```bash
   python src/offline/chemberta_extract.py \
   --csv_glob "data/DAVIS/fold*_*.csv" \
   --out /root/lanyun-tmp/cache/chemberta \
   --dataset DAVIS \
   --model_dir /root/lanyun-tmp/hf/ChemBERTa-77M-MLM \
   --offline \
   --device cuda \
   --dtype fp16 \
   --pool cls
   ```
   - molclr
   ```bash
   export MOLCLR_HOME=/root/ugca-dti
   export CKPT=/root/lanyun-tmp/hf/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth
   export OUT=/root/lanyun-tmp/cache/molclr

   python "$MOLCLR_HOME/src/offline/molclr_extract.py" \
   --csv_glob "/root/lanyun-tmp/davis_k5_seed42/fold*_*.csv" \
   --out "$OUT" --dataset DAVIS \
   --mode molclr --molclr_repo "$MOLCLR_HOME" --ckpt "$CKPT" \
   --device cuda --dtype fp16 --hidden 300 --layers 5
   ``` 
4. 训练与评测（示例：DAVIS）
   ```bash
   python -m src/train.py \
   --dataset DAVIS \
   --dataset-dirname davis_k5 \
   --data-root /root/lanyun-tmp \
   --out /root/lanyun-tmp/ugca-runs/davis \
   --epochs 60 --batch-size 256 --workers 6 --lr 2e-4
   ```
    当中断训练，利用添加--resume继续训练（采用`last.pt`）