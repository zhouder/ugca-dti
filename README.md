# UGCA-DTI（Binary DTI, 非 DTA）

本仓库严格按实现文档组织：三数据集（DAVIS/BindingDB/BioSNAP）、五折评测（fold{1..5}_train/test.csv）、UGCA+MUTAN、两阶段训练与指标汇总。

## 快速开始
1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 准备数据
   - 将 `fold{1..5}_train.csv` 与 `fold{1..5}_test.csv` 放入 `data/{DAVIS|BindingDB|BioSNAP}/`。
   - CSV 必含列：`smiles,protein,label`（0/1）。
3. 可选：一次性离线抽取（建议首次运行）
   ```bash
   python src/offline/esm2_extract.py       --csv_glob "data/DAVIS/fold*_*.csv"       --out cache/esm2
   python src/offline/molclr_extract.py     --csv_glob "data/DAVIS/fold*_*.csv"       --out cache/molclr
   python src/offline/chemberta_extract.py  --csv_glob "data/DAVIS/fold*_*.csv"       --out cache/chemberta
   ```
4. 训练与评测（示例：DAVIS）
   ```bash
   python src/train.py --config configs/ugca_dti.yaml --dataset DAVIS
   ```
5. 汇总报告（均值±标准差）
   ```bash
   python src/eval.py --root runs/DAVIS --out runs/DAVIS_summary.json
   ```

**注意**：离线脚本在缺少预训练权重时会自动采用安全回退（零向量或简化特征），不影响端到端跑通。接入真实权重后，即可无缝替换缓存内容复现论文结果。
