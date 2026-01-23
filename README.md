# Where2Vec

**Official implementation of Where2Vec**

Where2Vec is a large-scale **multimodal geospatial representation learning framework** that jointly models vector maps, semantic knowledge, and satellite imagery to learn rich geographic embeddings.

---

## Abstract

Where2Vec is a large-scale multimodal framework for geospatial representation learning that integrates **vector maps**, **semantic knowledge**, and **satellite imagery**. Geographic entities are extracted from **OpenStreetMap**, enriched with **multilingual labels from Wikidata**, and paired with **Sentinel-2 satellite image chips**.

The model employs a **type-aware contrastive learning objective** that aligns both instance-level and semantic-level representations. Visual features are encoded using a **spectral attention CNN**, while textual embeddings are extracted using a lightweight transformer model (**MiniLM-L6-v2**) adapted via **LoRA**.

Experimental results demonstrate that Where2Vec effectively captures both geographic semantics and visual patterns, providing a strong foundation for **vision–language understanding**, **geospatial reasoning**, and **spatial analytics** tasks.

---

## Dataset

The preprocessed dataset used in this project can be downloaded from the following Google Drive link:

**Dataset:**  
https://drive.google.com/drive/folders/1SB0L1ym0MyMIQejVw-L6E9ZXN-by8Wsa?usp=sharing

The dataset includes:
- Text information matching the Sentinel-2 chips.
- Sentinel-2 satellite image chips
- Template to convert names to texts.

---

## Environment Setup

### 1. Create a Python virtual environment (Python 3.11)

```bash
python3.11 -m venv where2vec_env
source where2vec_env/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

Download checkpoints from the outputs folder.

Update the dataset path in config.py to point to the location where you downloaded the data from the Google Drive link.

## Training

After setting up the environment and updating the configuration, start training with:
```bash
python train.py
```
## Evaluation & Testing

To evaluate a trained model:

```bash
python evaluate.py
```
