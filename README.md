# Cheminformatics_molecule_property_project

This repository contains a cheminformatics and drug design project for the course *Cheminformatics and Drug Design*.  
The goal is to predict molecular properties using datasets from [MoleculeNet](https://moleculenet.org), applying both a **regression task** and a **classification task** with machine learning methods.

---

## 📂 Repository Structure
Cheminformatics_molecule_property_projet/

- `README.md` — Project overview  
- `environment.yaml` — Python dependencies  
- `data/` — Dataset storage (not tracked in git)  
  - `README.md` — Explanation for dataset usage and choices 
- `notebooks/` — Jupyter notebooks for analysis  
  - `01_data_exploration.ipynb`  
  - `02_classification.ipynb`  
  - `03_regression.ipynb`  
- `src/` — Python modules  
  - `preprocessing.py`  
  - `models.py`  
  - `evaluation.py`
---
## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<user>/Cheminformatics_molecule_property_project.git
   cd Cheminformatics_molecule_property_project
2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
3. **Activate the envrionment**
   ```bash
   conda activate cheminformatics-project

4. **Run the notebooks**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb

---
# 📊 Project Tasks

- Select one regression dataset and one classification dataset from MoleculeNet.
- Apply different molecular featurization strategies (e.g., Morgan fingerprints, descriptors, graph-based).
- Train and compare at least two machine learning models per task (Random Forest, SVM, Neural Networks, etc.).
- Perform hyperparameter tuning and evaluate models with appropriate metrics.
- Provide visualizations and critical evaluation of the results.
- Ensure reproducibility and clear documentation of all steps.
---
# 👥 Contributors
- [Marcos Bolanos](https://github.com/marcosbolanos)
- [Robin Blouin](https://github.com/Orbliss)
