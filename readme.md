# Dual-Stream Diffusion for Reflection Removal

This project implements a **Dual-Stream Diffusion Framework** based on IR-SDE for removing reflections from images captured through glass. The framework decomposes a mixed image $M = T + R$ into its transmission ($T$) and reflection ($R$) components.

---

## 🔧 Project Structure

```
dual_diffusion_rr/
├── models/
│   ├── dual_unet.py               # Dual-head UNet for T/R prediction
│   └── diffusion_utils.py         # Diffusion process helpers
├── losses/
│   ├── matching_loss.py           # IR-SDE matching loss
│   └── reflection_losses.py       # RR-specific loss (gradient, exclusion, etc.)
├── dataset/
│   ├── synthetic_generator.py     # Synthetic reflection dataset
│   └── real_dataset.py            # Real-world reflection dataset
├── train.py                       # Training pipeline
├── inference.py                   # Inference from M → (T, R)
├── config.yaml                    # Configurable hyperparameters
└── readme.md                      # Project documentation
```

---

## Key Idea

* Treat mixed image $M$ as a degraded version of both T and R
* Extend **IR-SDE** (Image Restoration via SDEs) with a **dual-stream architecture**
* Use shared encoder and split decoders (dual-head UNet) to recover both streams

---

## Dataset

* **Synthetic**: Generated with paired T and R (from VOC, COCO, etc.)
* **Nature**: Real-world scenes with natural reflections
* **Real**: Collected real photos with manually cleaned ground truth

> Dataset format: under each root, have subfolders: `T/`, `R/`, `M/`

---

## Training

```
python train.py --config config.yaml
```

* Diffusion is trained with IR-SDE matching loss + RR-specific loss
* Checkpoints saved every N iterations

---

## Inference

```
python inference.py --img ./test.jpg --checkpoint ./checkpoints/best.pth
```

* Starts from t=T and performs reverse steps to obtain $\hat{T}, \hat{R}$

---

## Notes

* Designed without pretrained weights (fully trainable)
* Fully supports `Setting 1/2/3` for various reflection definitions
* Easily extendable to real-time or video scenarios

---
