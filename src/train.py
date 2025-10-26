from __future__ import annotations
import argparse, json, math
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.seed import set_seed
from .utils.paths import ensure_dir
from .utils.logger import log_kv
from .utils.timer import timer
from .datamodule import DataModule, DMConfig
from .metrics import compute_all_metrics
from .losses import BCEWithLogitsLossWrapper, FocalLoss
from .model.ugca import UGCA, Projector
from .model.mutan import MUTAN
from .model.pooling import TokenPooler

class Model(nn.Module):
    def __init__(self, d: int, cfg_model: dict):
        super().__init__()
        self.d = d
        self.proj_D = Projector(in_dim=None, d=d, p=cfg_model['ugca']['dropout'])
        self.proj_P = Projector(in_dim=None, d=d, p=cfg_model['ugca']['dropout'])
        self.proj_C = Projector(in_dim=None, d=d, p=cfg_model['ugca']['dropout'])
        self.ugca = UGCA(d=d,
                         heads=cfg_model['ugca']['heads'],
                         dropout=cfg_model['ugca']['dropout'],
                         k_target=2.0,
                         g_min=cfg_model['ugca']['g_min'],
                         rho=cfg_model['ugca']['rho'],
                         budget_lambda=cfg_model['ugca']['budget_lambda'],
                         topk_enable=cfg_model['ugca']['topk_enable'],
                         topk_ratio=cfg_model['ugca']['topk_ratio'],
                         layers=cfg_model['ugca']['layers'])
        self.poolD = TokenPooler(d, mode=cfg_model['pooling']['type'])
        self.poolP = TokenPooler(d, mode=cfg_model['pooling']['type'])
        self.mutan = MUTAN(d_in1=d*2, d_in2=d,
                           d_out=cfg_model['mutan']['z_dim'],
                           rank=cfg_model['mutan']['rank_R'],
                           dropout=cfg_model['mutan']['dropout'])
        hid = cfg_model['classifier']['hidden'][0]
        self.cls = nn.Sequential(
            nn.Linear(cfg_model['mutan']['z_dim'], hid), nn.ReLU(), nn.Dropout(cfg_model['classifier']['dropout']),
            nn.Linear(hid, 1)
        )

    def reset_proj(self, dD: int, dP: int, dC: int):
        self.proj_D = Projector(dD, self.d, p=0.0)
        self.proj_P = Projector(dP, self.d, p=0.0)
        self.proj_C = Projector(dC, self.d, p=0.0)

    def forward(self, batch, stage='B', k_now=None):
        HD = self.proj_D(batch['H_D'])
        HP = self.proj_P(batch['H_P'])
        hC = self.proj_C(batch['h_C'])
        XD, XP, gD, gP, loss_reg = self.ugca(HD, batch['mask_D'], HP, batch['mask_P'], stage=stage, k_now=k_now)
        hD = self.poolD(XD, batch['mask_D'])
        hP = self.poolP(XP, batch['mask_P'])
        z = self.mutan(torch.cat([hD, hC], dim=-1), hP)
        logit = self.cls(z).squeeze(-1)
        return logit, loss_reg

def make_loss(main: str):
    if main == 'bce':
        return BCEWithLogitsLossWrapper()
    elif main == 'bce_weighted':
        return BCEWithLogitsLossWrapper(pos_weight=2.0)
    else:
        return FocalLoss()

def train_one_dataset(cfg: dict, dataset: str):
    set_seed(cfg['seed'])
    device = 'cuda' if (cfg['device']=='cuda' and torch.cuda.is_available()) else 'cpu'
    folds = cfg['data']['folds']

    def path_of(tmpl: str, fold: int):
        return tmpl.replace('${dataset}', dataset).replace('${fold}', str(fold))

    results = []

    for fold in folds:
        train_csv = path_of(cfg['data']['train_csv_tmpl'], fold)
        test_csv  = path_of(cfg['data']['test_csv_tmpl'],  fold)
        out_dir = ensure_dir(Path('runs')/dataset/f"fold{fold}")

        dm = DataModule(DMConfig(
            train_csv=train_csv,
            test_csv=test_csv,
            batch_size=cfg['data']['batch_size'],
            num_workers=cfg['data']['num_workers'],
            cache_dirs={'esm2': cfg['cache']['esm2_dir'], 'molclr': cfg['cache']['molclr_dir'], 'chemberta': cfg['cache']['chemberta_dir']}
        ))
        train_loader, test_loader = dm.loaders()

        first_batch = next(iter(train_loader))
        dD, dP, dC = first_batch['H_D'].shape[-1], first_batch['H_P'].shape[-1], first_batch['h_C'].shape[-1]
        model = Model(d=cfg['model']['d'], cfg_model=cfg['model'])
        model.reset_proj(dD, dP, dC)
        model.to(device)

        loss_main = make_loss(cfg['train']['loss']['main'])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
        total_steps = cfg['train']['total_epochs'] * max(1, len(train_loader))
        warmup_steps = int(total_steps * cfg['train']['warmup_ratio'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,total_steps-warmup_steps))
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg['precision']=='amp' and device=='cuda'))

        best = {'auprc': -1.0, 'metrics': None, 'epoch': -1}
        stageA_epochs = cfg['train']['stageA_epochs']
        k_warm = cfg['model']['ugca']['k_warmup_epochs']
        global_step = 0

        for epoch in range(1, cfg['train']['total_epochs']+1):
            model.train()
            stage = 'A' if epoch <= stageA_epochs else 'B'
            ep_losses = []
            with timer() as t_ep:
                pbar = tqdm(train_loader, desc=f"train e{epoch} ({stage})")
                for i,batch in enumerate(pbar):
                    batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=(cfg['precision']=='amp' and device=='cuda')):
                        if stage=='B' and k_warm>0:
                            e_rel = min(1.0, (epoch-stageA_epochs)/max(1,k_warm))
                            k_now = 2.0 * e_rel
                        else:
                            k_now = None
                        logits, loss_reg = model(batch, stage=stage, k_now=k_now)
                        loss = loss_main(logits, batch['labels']) + loss_reg
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
                    scaler.step(opt); scaler.update()
                    if global_step < warmup_steps:
                        for pg in opt.param_groups:
                            pg['lr'] = cfg['train']['lr'] * (global_step+1) / max(1,warmup_steps)
                    else:
                        sched.step()
                    global_step += 1
                    ep_losses.append(loss.item())
                    pbar.set_postfix({'loss': f"{np.mean(ep_losses):.4f}"})
            ys, ps = [], []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='eval'):
                    batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
                    logits, _ = model(batch, stage='B')
                    prob = torch.sigmoid(logits)
                    ys.extend(batch['labels'].cpu().numpy().tolist())
                    ps.extend(prob.cpu().numpy().tolist())
            metrics = compute_all_metrics(ys, ps, threshold=cfg['eval']['threshold'])
            log_kv(epoch=epoch, fold=fold, dataset=dataset, **metrics)
            if metrics['auprc'] > best['auprc']:
                best.update({'auprc': metrics['auprc'], 'metrics': metrics, 'epoch': epoch})
                torch.save({'model': model.state_dict(), 'cfg': cfg}, out_dir/'best.pt')
            with open(out_dir/'last_metrics.json','w') as f: json.dump(metrics, f)
        with open(out_dir/'best_metrics.json','w') as f: json.dump(best, f)
        results.append(best['metrics'])
    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--dataset', type=str, required=True, choices=['DAVIS','BindingDB','BioSNAP'])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,'r'))
    res = train_one_dataset(cfg, args.dataset)
