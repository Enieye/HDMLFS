## About

This repository contains the implementation of **Hybrid Deep Machine Learning Feature Selection (HDMLFS)**, a novel framework for Intrusion Detection Systems (IDS).  
HDMLFS addresses challenges of high-dimensional feature spaces, limited interpretability, and high computational cost in deep learning-based IDS models.  

The framework works in two stages:
1. **Correlation-based filtering** – removes redundant features by analyzing correlations.  
2. **Voting-based selection** – combines Integrated Gradients (IG) and SHAP rankings to retain the most informative features.  

---

## Key Results

- Evaluated on **NSL-KDD** and **CSE-CIC IDS2018** datasets.  
- Reduced feature space by **48%** (NSL-KDD) and **65%** (CSE-CIC IDS2018).  
- Improved detection of rare attacks:
  - **86.96% recall** for Web attacks (CSE-CIC IDS2018).  
  - **80.77% recall** for U2R attacks (NSL-KDD).  
- Achieved **98.23% weighted accuracy** (CSE-CIC IDS2018, ResNet-SF) and **99.77% accuracy** (NSL-KDD).  

---

## Highlights

- Improves IDS performance while reducing complexity.  
- Provides a more interpretable and efficient feature selection pipeline.  
- Supports deep learning models with stronger generalization to rare attack types.  

---

## Inputs and Outputs

Create a root folder called **`HDMLFS`** to contain all project files.  
There are three main files for each dataset (**NSL-KDD** and **CSE-CIC IDS2018**).  

### NSL-KDD
1. **`HDMLFSNSLKDD_01Preprocessing.py`**  
   - **Input**: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd) and store it in:  
     `path/to/HDMLFS/data_folder`  
   - **Output**:  
     - `path/to/HDMLFS/data_folder/01fulltraintest.csv`  
     - `path/to/HDMLFS/output_results_nslkdd01.txt`  

2. **`HDMLFSNSLKDD_02ML.py`**  
   - **Input**: `path/to/HDMLFS/data_folder/01fulltraintest.csv`  
   - **Output**:  
     - `path/to/HDMLFS/output_results_nslkdd02.txt`  

3. **`HDMLFSNSLKDD_03DL.py`**  
   - **Input**: `path/to/HDMLFS/data_folder/01fulltraintest.csv`  
   - **Output**:  
     - `path/to/HDMLFS/output_results_nslkdd03.txt`  

*(The CSE-CIC IDS2018 files follow the same structure and naming convention.)*  

---

## Setup

### 1. Create Virtual Environment

python3 -m venv .venv

### Activate it
#### Linux / macOS
source .venv/bin/activate

#### Windows (PowerShell)
.venv\Scripts\Activate.ps1

#### Install dependencies
pip install -r requirements.txt

#### Running each of the files with (for example)
./HDMLFSNSLKDD_01Preprocessing.py

## Note: 
- To see the plots and other nedia, run the corresponging .ipynb files.
- Ensure datasets are downloaded and placed in the data_folder before running scripts.


## Repository Structures 
```
HDMLFS/
│── data_folder/ # Raw and processed datasets
│ ├── NSL-KDD/ # Downloaded NSL-KDD dataset
│ └── CSE-CIC-IDS2018/ # Downloaded CSE-CIC IDS2018 dataset
│
│── notebooks/ # Jupyter notebooks for visualization & plots
│ ├── NSLKDD_Analysis.ipynb
│ └── CSECIC_Analysis.ipynb
│
│── scripts/ # Python scripts for preprocessing, ML, and DL
│ ├── HDMLFSNSLKDD_01Preprocessing.py
│ ├── HDMLFSNSLKDD_02ML.py
│ ├── HDMLFSNSLKDD_03DL.py
│ ├── HDMLFSCSECIC_01Preprocessing.py
│ ├── HDMLFSCSECIC_02ML.py
│ └── HDMLFSCSECIC_03DL.py
│
│── results/ # Output results and logs
│ ├── output_results_nslkdd01.txt
│ ├── output_results_nslkdd02.txt
│ ├── output_results_nslkdd03.txt
│ └── ...
│
│── requirements.txt # Python dependencies
│── README.md # Project documentation (this file)
```
