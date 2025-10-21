import os
import shutil
import pandas as pd
from pathlib import Path
import deepchem as dc
from deepchem.utils import data_utils
import requests
import tarfile
import zipfile
import gzip
from tqdm import tqdm

# =========================
# Définir les dossiers
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
sider_dir = os.path.join(DATA_DIR, "sider")
pdbbind_dir = os.path.join(DATA_DIR, "pdbbind")

os.makedirs(sider_dir, exist_ok=True)
os.makedirs(pdbbind_dir, exist_ok=True)

# =========================
# 1️⃣ Télécharger SIDER et exporter CSV
# =========================


print("Téléchargement du dataset SIDER...")
sider_tasks, (train, valid, test), transformer = dc.molnet.load_sider(reload=True)

def dataset_to_df(dataset):
    X = [d[0] for d in dataset.X]  # SMILES
    y = dataset.y
    df = pd.DataFrame(y, columns=sider_tasks)
    df['smiles'] = X
    return df

train_df = dataset_to_df(train)
train_df.to_csv(os.path.join(sider_dir, "train.csv"), index=False)

valid_df = dataset_to_df(valid)
valid_df.to_csv(os.path.join(sider_dir, "valid.csv"), index=False)

test_df = dataset_to_df(test)
test_df.to_csv(os.path.join(sider_dir, "test.csv"), index=False)

print("SIDER CSVs sauvegardés dans :", sider_dir)




print(f"Data directory: {DATA_DIR}")

datasets = {
    "pdbbind_v2015.tar.gz": "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz",
}

def download_datasets(datasets:dict=datasets):
    for name, url in datasets.items():
        filepath = os.path.join(pdbbind_dir, name)
        if not os.path.exists(filepath):
            print(f"Téléchargement de {name}...")
            
            # Stream download with progress bar
            response = requests.get(url, stream=True, verify=False)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, "wb") as f, tqdm(
                desc=name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"{name} téléchargé et enregistré sous {filepath}")
        else:
            print(f"{name} est déjà présent.")

def extract_files(datasets:dict=datasets):
    """Extract downloaded compressed files"""
    for name in datasets.keys():
        filepath = os.path.join(pdbbind_dir, name)
        if os.path.exists(filepath):
            if name.endswith('.tar.gz'):
                extract_dir = os.path.join(pdbbind_dir, name.replace('.tar.gz', ''))
                if not os.path.exists(extract_dir):
                    print(f"Extraction de {name}...")
                    with tarfile.open(filepath, 'r:gz') as tar:
                        members = tar.getmembers()
                        with tqdm(total=len(members), desc=f"Extracting {name}") as pbar:
                            for member in members:
                                tar.extract(member, pdbbind_dir)
                                pbar.update(1)
                    print(f"{name} extrait dans {extract_dir}")
                else:
                    print(f"{name} est déjà extrait.")
            
            elif name.endswith('.csv.gz'):
                extract_path = os.path.join(pdbbind_dir, name.replace('.gz', ''))
                if not os.path.exists(extract_path):
                    print(f"Extraction de {name}...")
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(extract_path, 'wb') as f_out:
                            # Get compressed file size for progress bar
                            compressed_size = os.path.getsize(filepath)
                            
                            with tqdm(total=compressed_size, desc=f"Extracting {name}", unit='B', unit_scale=True) as pbar:
                                while True:
                                    chunk = f_in.read(8192)
                                    if not chunk:
                                        break
                                    f_out.write(chunk)
                                    # Update progress based on compressed bytes read
                                    pbar.update(len(chunk))
                    print(f"{name} extrait vers {extract_path}")
                else:
                    print(f"{name} est déjà extrait.")

            elif name.endswith('.zip'):
                extract_dir = os.path.join(DATA_DIR, name.replace('.zip', ''))
                if not os.path.exists(extract_dir):
                    print(f"Extraction de {name}...")
                    with zipfile.ZipFile(filepath, 'r') as z:
                        members = z.namelist()
                        with tqdm(total=len(members), desc=f"Extracting {name}") as pbar:
                            for member in members:
                                z.extract(member, DATA_DIR)
                                pbar.update(1)
                    print(f"{name} extrait dans {extract_dir}")
                else:
                    print(f"{name} est déjà extrait.")


def delete_files():
    for name in datasets.keys():
        filepath = os.path.join(DATA_DIR, name)
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    download_datasets()
    extract_files()
    delete_files()