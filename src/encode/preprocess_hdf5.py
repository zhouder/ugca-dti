import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.splits import generate_ids

ROOT = '/root/lanyun-fs/'

DATASETS = ['davis', 'kiba', 'drugbank']

def save_graph_to_group(group, npz_data):
    group.create_dataset('node_s', data=npz_data['node_s'].astype(np.float32))
    group.create_dataset('node_v', data=npz_data['node_v'].astype(np.float32))
    group.create_dataset('edge_index', data=npz_data['edge_index'].astype(np.int64))
    group.create_dataset('edge_s', data=npz_data['edge_s'].astype(np.float32))
    group.create_dataset('edge_v', data=npz_data['edge_v'].astype(np.float32))

def process_dataset(dataset_name):
    print(f"\n>>> Processing dataset {dataset_name} ...")
    base_dir = os.path.join(ROOT, dataset_name)
    csv_path = os.path.join(base_dir, f"{dataset_name}.csv")
    h5_path = os.path.join(base_dir, f"{dataset_name}_data.h5")

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    df = generate_ids(df)

    unique_dids = df['did'].unique()
    unique_pids = df['pid'].unique()

    with h5py.File(h5_path, 'w') as f:
        print(f"Creating HDF5 file: {h5_path}")

        drug_grp = f.create_group('drugs')
        print(f"Packing {len(unique_dids)} drugs ...")

        for did in tqdm(unique_dids, desc="Drugs"):
            d_sub = drug_grp.create_group(did)

            try:
                m = np.load(os.path.join(base_dir, 'molclr', f"{did}.npy"))
                if m.ndim == 2:
                    m = np.mean(m, axis=0)
                d_sub.create_dataset('molclr', data=m.astype(np.float32))
            except:
                d_sub.create_dataset('molclr', data=np.zeros(300, dtype=np.float32))

            try:
                c = np.load(os.path.join(base_dir, 'chemberta', f"{did}.npy"))
                if c.ndim == 2:
                    c = np.mean(c, axis=0)
                d_sub.create_dataset('chemberta', data=c.astype(np.float32))
            except:
                d_sub.create_dataset('chemberta', data=np.zeros(384, dtype=np.float32))

        prot_grp = f.create_group('proteins')
        print(f"Packing {len(unique_pids)} proteins (ESM2 + PocketGraph) ...")

        for pid in tqdm(unique_pids, desc="Proteins"):
            p_sub = prot_grp.create_group(pid)

            try:
                e_data = np.load(os.path.join(base_dir, 'esm2', f"{pid}.npz"))
                k = next((k for k in e_data.files if 'mean' in k), e_data.files[0])
                e = e_data[k]
                if e.ndim == 2:
                    e = np.mean(e, axis=0)
                p_sub.create_dataset('esm2', data=e.astype(np.float32))
            except:
                p_sub.create_dataset('esm2', data=np.zeros(1280, dtype=np.float32))

            try:
                pg_data = np.load(os.path.join(base_dir, 'pocket_graph', f"{pid}.npz"))
                g_sub = p_sub.create_group('pocket')
                save_graph_to_group(g_sub, pg_data)
            except:
                g_sub = p_sub.create_group('pocket')
                g_sub.create_dataset('node_s', data=np.zeros((1, 29), dtype=np.float32))
                g_sub.create_dataset('node_v', data=np.zeros((1, 3, 3), dtype=np.float32))
                g_sub.create_dataset('edge_index', data=np.zeros((2, 0), dtype=np.int64))
                g_sub.create_dataset('edge_s', data=np.zeros((0, 5), dtype=np.float32))
                g_sub.create_dataset('edge_v', data=np.zeros((0, 1, 3), dtype=np.float32))

    print(f"Done. File size: {os.path.getsize(h5_path) / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    process_dataset('kiba')
