# Reference Difference Transformer (RDTransformer)
**RDTransformer** is a Transformer-based model designed to predict the functional effects of mutations in biological sequences (RNA, DNA, or proteins). Built upon a Transformer backbone, it introduces a **dynamic difference-based embedding mechanism** relative to the wild-type sequence. This mechanism filters out noise from non-mutational sites while emphasizing the effects of introduced or natural mutations.

---

## 🧠 Architecture
![Architecture of the RDTransformer](https://raw.githubusercontent.com/davidwongmedinfo/RDTransformer/main/architecture.png)

---

## 📚 Data
The pretraining data was sourced from the [**RNAcentral** database](https://rnacentral.org/).  
**Citation:**
> RNAcentral Consortium.  
> RNAcentral 2021: secondary structure integration, improved sequence search and new member databases.  
> Nucleic Acids Research, 2021; 49(D1):D212-D220.  
> [https://doi.org/10.1093/nar/gkaa921](https://doi.org/10.1093/nar/gkaa921)

Both raw and preprocessed pretraining datasets are available [here](https://github.com/davidwongmedinfo/RDTransformer/tree/main/data).

---

## 💻 Source Code
The full implementation is available [here](https://github.com/davidwongmedinfo/RDTransformer/tree/main/src).

---

## ⚙️ Environment Setup
**1. Create environment:**
```bash
conda env create -f environment.yml
```
> **Note**: This is a CPU-only environment to maximize compatibility. The original experiments in the paper were conducted with the following configuration:
> - PyTorch 2.6.0 + CUDA 12.4  
> - NVIDIA driver 555.42.06
> - cuDNN: as bundled with the PyTorch 2.6.0+cu124 distribution
>
> To reproduce GPU-accelerated results, install the matching CUDA-enabled PyTorch wheel: 
> ```bash
> pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
> ```

**2. Activate environment:**
```bash
conda activate rdt_env
```

---

## 🚀 Run Scripts:
**Configure finetuning using the YAML files in `src/finetune_configs/`, then run scripts from `src/`**

**1. Run pretraining:**
```bash
python pretrain.py
```
**2. Run cross-validation for fine-tuning:**
```bash
python finetune_cv.py --config finetune_configs/wb_cv_config.yaml
python finetune_cv.py --config finetune_configs/elisa_cv_config.yaml
```
**3. Run finetuning on full training set:**
```bash
python finetune_fulltrain.py --config finetune_configs/wb_fulltrain_config.yaml
python finetune_fulltrain.py --config finetune_configs/elisa_fulltrain_config.yaml
```
**4. Run evaluation on held-out test set:**
```bash
python finetune_test.py --config finetune_configs/wb_test_config.yaml
python finetune_test.py --config finetune_configs/elisa_test_config.yaml
```