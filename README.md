# VQ-VAE + Wave Function Collapse (WFC) — Map Synthesizer

Generate brand-new maps in the **style** of your examples by learning a discrete codebook with a **VQ-VAE**, arranging those codes with **Wave Function Collapse**, and decoding back to pixels.

> Works either from a **folder of tiles** or directly from a **single large image** (e.g., 4096×4096) that’s automatically split into 256×256 tiles.

---

## ✨ Highlights

- **Two training modes**
  - From a folder of tiles (`./tiles/*.png`, `./tiles/*.jpg`, …)
  - From a single big image (e.g. 4096×4096), auto-subdivided into tiles
- **Wave Function Collapse** works directly on the discrete VQ codes
- **Flexible**: configurable tile size, embedding size, latent dimension
- **Visualization-friendly**: decode generated maps back to images

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourname/vqvae-wfc.git
cd VQ-VAE-Wave-Function-Collapse-Map-Synthesizer
pip install -r requirements.txt
```

---

## Usage
**Train from a single image**
```bash
python vqvae_wfc_pipeline.py \
  --mode train \
  --big_image ./maps/test_map.png \
  --tile 256 \
  --epochs 100 \
  --batch 8 \
  --save vqvae_model.keras
```


**Train from a folder of tiles**
```bash
python vqvae_wfc_pipeline.py \
  --mode train \
  --data_dir ./tiles \
  --epochs 100 \
  --batch 8 \
  --save vqvae_model.keras
```

**Generate from a single image**
```bash
python vqvae_wfc_pipeline.py \
  --mode gen \
  --load vqvae_model.keras \
  --big_image ./maps/huge_map.png \
  --tile 256 \
  --out_codes 8 8 \
  --out_png generated_map.png
```

**Generate from a folder of tiles**
```bash
python vqvae_wfc_pipeline.py \
  --mode gen \
  --load vqvae_model.keras \
  --data_dir ./tiles \
  --out_codes 8 8 \
  --out_png generated_map.png
```

## Outputs
**vqvae_model.keras** : saved VQ-VAE model
**generated_map.png** : synthesized map image


## Parameters
`--tile` : size of tile when using a big image (default: 256)

`--latent_dim` : latent dimension of VQ-VAE (default: 512)

`--num_embeddings` : size of codebook (default: 16)

`--epochs` : number of training epochs (default: 100)

`--batch` : batch size (default: 8)

`--strict` : fail if big image not divisible by tile (default: False; will crop otherwise)






