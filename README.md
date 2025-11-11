# UGCA-DTI（Binary DTI, 非 DTA）

本仓库基于 ESM2 / MolCLR / ChemBERTa 的三路特征、UGCA 交互 + MUTAN 融合的二分类 DTI 模型。  
**重要更新：** 训练评测切换为**冷启动场景（Group K-fold）**，仅使用 `all.csv`，不再使用 `fold{1..5}_train/test.csv`。

## 快速开始
1. 安装依赖
   ```bash
   conda create -n ugca-dti python=3.10 
   conda activate ugca-dti
   pip install -r requirements.txt
   ```
2. 准备数据
   - 将原始数据集`all.csv`放入`./data/DAVIS`下。
   - CSV 必含列：`smiles,protein,label`（0/1）。
3. 可选：一次性离线抽取（建议首次运行）
   - ems-2
   ```bash
   python src/offline/esm2_extract.py \
   --csv_glob "/data/DAVIS/all.csv" \
   --out /root/lanyun-tmp/cache/esm2 \
   --dataset DAVIS \
   --model_dir /root/lanyun-tmp/hf/esm2_t33_650M_UR50D \
   --offline --device cuda --dtype fp16 --chunk_len 1000
   ```
   - chemberta（目前是384维，后面可以用768维）
   ```bash
   python src/offline/chemberta_extract.py \
   --csv_glob "/data/DAVIS/all.csv" \
   --out /root/lanyun-tmp/cache/chemberta \
   --dataset DAVIS \
   --model_dir /root/lanyun-tmp/hf/ChemBERTa-77M-MLM \
   --offline --device cuda --dtype fp16 --pool cls
   ```
   - molclr
   ```bash
   export MOLCLR_HOME=/root/ugca-dti
   export CKPT=/root/lanyun-tmp/hf/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth
   
   python "$MOLCLR_HOME/src/offline/molclr_extract.py" \
   --csv_glob "/data/DAVIS/all.csv" \
   --out "/root/lanyun-tmp/cache/molclr" --dataset DAVIS \
   --mode molclr --molclr_repo "$MOLCLR_HOME" --ckpt "$CKPT" \
   --device cuda --dtype fp16 --hidden 300 --layers 5
   ``` 
4. 训练与评测（示例：DAVIS）
   ```bash
   python -m src.train \
   --dataset DAVIS \
   --sequence \
   --epochs 100 --batch-size 64 --workers 6 --lr 5e-5 \
   --split-mode cold-protein \        # 或 cold-drug
   --cv-folds 5 \                     # 冷启动K折（默认5）
   --thr auto \                       # 自适应阈值（验证集上选 F1 最优阈值）
   --gate-budget 0.001 --gate-rho 0.6
   ```
    当中断训练，利用添加--resume继续训练（采用`last.pt`）