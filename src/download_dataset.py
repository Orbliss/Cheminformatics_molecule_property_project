import os
import requests

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

datasets = {
    "PDBbind": "https://www.pdbbind-plus.org.cn/download.php",
    "SIDER": "https://sideeffects.embl.de/download/"
}

for name, url in datasets.items():
    filepath = os.path.join(DATA_DIR, f"{name}.html")
    if not os.path.exists(filepath):
        print(f"Téléchargement de {name}...")
        r = requests.get(url, verify=False)  # <- ignore SSL
        with open(filepath, "wb") as f:
            f.write(r.content)
        print(f"{name} téléchargé et enregistré sous {filepath}")
    else:
        print(f"{name} est déjà présent.")
