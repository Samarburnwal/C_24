# Low-Resolution Surveillance Image Upscaling using Super-Resolution

Rebuild sharper, 4× larger images from small, blurry, noisy and compressed CCTV-style frames, and
measure how much each method really helps.

We simulate CCTV damage on high-resolution photos, upscale the result with three methods
(**Bicubic**, **SRCNN** trained by us, pretrained **Real-ESRGAN**), and score them with **PSNR, SSIM
and LPIPS**. An optional script fine-tunes an **ESRGAN** generator on our own degradation. A
Streamlit demo and a command-line tool upscale single images and videos.

**Contents:** [Results](#1-results-summary) · [Project structure](#2-project-structure) ·
[Implementation details](#3-implementation-details) · [How to run (step by step)](#4-step-by-step-how-to-run-the-code) ·
[Dataset](#5-dataset) · [Limitations](#6-limitations) · [Suggested extensions](#7-suggested-extensions) ·
[References](#8-references)

---

## 1. Results summary

Average over 80 unseen test images, ×4 upscaling, *medium* degradation, identical degraded input for
every model (higher is better for PSNR and SSIM, lower is better for LPIPS):

| Model | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Bicubic (baseline) | 23.35 | 0.5806 | 0.6604 |
| SRCNN (trained by us) | 23.73 | 0.6006 | 0.5459 |
| Real-ESRGAN (pretrained) | 22.57 | 0.5777 | **0.3040** |
| ESRGAN-FT-L1 (our fine-tune, phase 1) | **24.27** | **0.6310** | 0.4737 |
| ESRGAN-FT (our fine-tune, GAN phase) | 21.59 | 0.5055 | 0.4056 |

**How to read the results**

* **PSNR / SSIM** reward pixel-faithfulness. L1-trained models (SRCNN, EDSR) score highest here.
* **LPIPS** (lower is better) and visual inspection reward perceptual sharpness. GAN-based models
  such as Real-ESRGAN usually win here but often lose on PSNR, because they add plausible texture
  rather than exact pixels. This PSNR vs. perceptual trade-off is a standard discussion point.
* Always compare against **Bicubic**: a learned model that does not beat it is not working.
* Your numbers will differ slightly with a different dataset split or training run.

---

## 2. Project structure

```
surveillance_sr/
├── README.md
├── requirements.txt
├── prepare_data.py        split HR images into train / val / test (or make a synthetic test set)
├── make_subimages.py      cut big images into small crops once (much faster training)
├── train.py               train SRCNN / EDSR
├── train_esrgan.py        fine-tune ESRGAN (RRDBNet generator + GAN)
├── interpolate_models.py  blend the L1 and GAN ESRGAN weights (network interpolation)
├── evaluate.py            compare all models: tables + side-by-side images
├── upscale.py             upscale one image or a whole video
├── app.py                 Streamlit demo
└── sr/
    ├── degrade.py         CCTV degradation simulator
    ├── data.py            (LR, HR) pair dataset
    ├── models.py          Bicubic, SRCNN, EDSR, RRDBNet (ESRGAN generator)
    ├── metrics.py         PSNR, SSIM, LPIPS
    ├── infer.py           tiled inference for large frames
    └── realesrgan_wrapper.py   pretrained Real-ESRGAN wrapper
```

---

## 3. Implementation details

### 3.1 Pipeline

```
HR photo → CCTV degradation → LR image → Upscaler (Bicubic / SRCNN / Real-ESRGAN / ESRGAN-FT)
        → 4× output → compare with the original HR photo → PSNR, SSIM, LPIPS → tables + images
```

We never store low-resolution images. They are generated **on the fly** from the high-resolution
photos, so every training step sees a fresh random degradation, and the original photo is always the
exact ground truth.

### 3.2 Degradation simulator (`sr/degrade.py`)

Applied in this order, each setting drawn at random inside a range:

1. **Blur**: Gaussian blur (70% of images) or motion blur (30%).
2. **Downscale ×4** using a randomly chosen interpolation (area, linear or cubic).
3. **Sensor noise**: Gaussian noise.
4. **JPEG compression** at a random quality.

| Strength | Gaussian σ | Motion blur length (px) | Noise σ | JPEG quality |
|---|---|---|---|---|
| light | 0.5–1.2 | 3–5 | 0–3 | 65–90 |
| **medium** (used for all results) | 0.8–2.0 | 3–7 | 2–8 | 35–75 |
| heavy | 1.2–3.0 | 5–11 | 5–15 | 15–50 |

For testing, the random choices are **seeded per image**, so every model is evaluated on exactly the
same degraded input.

### 3.3 Data loading (`sr/data.py`, `prepare_data.py`, `make_subimages.py`)

* `prepare_data.py` splits the HR images 80 / 10 / 10 into `train`, `val`, `test` (fixed random seed).
* Training samples are random **96×96 HR crops** (LR crop 24×24) with random flips. Validation uses
  deterministic centre crops.
* Decoding a full 2K image just to use one small crop is slow. `make_subimages.py` cuts the images
  into 256×256 sub-images once, so each training sample loads a small file. **Use this for training.**

### 3.4 Models (`sr/models.py`)

| Model | Description |
|---|---|
| **Bicubic** | Non-learned interpolation. The baseline every learned model must beat. |
| **SRCNN** | Three conv layers (9×9 → 5×5 → 5×5, 64 and 32 channels) on a bicubic-upscaled input. It predicts a residual added to the bicubic image. |
| **EDSR** | Compact version (16 residual blocks, 64 channels, sub-pixel upsampling) with a bicubic skip connection. Included in the code; not trained in the reported results. |
| **RRDBNet (ESRGAN)** | The Real-ESRGAN / ESRGAN generator: 23 residual-in-residual dense blocks. Layer names match the official one, so the pretrained `RealESRGAN_x4plus.pth` loads directly. |
| **Real-ESRGAN (pretrained)** | Official pretrained ×4 model, run through the `realesrgan` package (tiled). No training needed. |

### 3.5 Training (`train.py`)

* Loss: **L1**. Optimiser: **Adam**, learning rate 2e-4 with cosine decay, gradient clipping at 1.0.
* Every epoch is validated by PSNR on the validation crops; the **best checkpoint is saved** to
  `checkpoints/<model>_x4.pt`.
* Runs on a CUDA GPU if available, otherwise on the CPU (the script prints `device: cuda` or `cpu`).

### 3.6 ESRGAN fine-tuning (`train_esrgan.py`)

Starts from the pretrained Real-ESRGAN weights and trains on our own CCTV degradation in two phases:

* **Phase 1 (`--iters_l1`)**: L1 loss only. Saved as `esrgan_x4_l1.pt` (the pixel-accurate model).
* **Phase 2 (`--iters_gan`)**: L1 + **VGG19 perceptual loss** (conv5_4 features) + **GAN loss** from a
  spectral-norm patch discriminator (weights 1 : 1 : 0.1). Saved as `esrgan_x4.pt`.
* Memory savers for a 4 GB GPU: mixed precision (fp16), gradient checkpointing (`--checkpoint`),
  small 128×128 HR patches.
* `interpolate_models.py` blends the two checkpoints, `(1−α)·L1 + α·GAN`, to trade fidelity for
  sharpness without retraining.

### 3.7 Evaluation (`evaluate.py`, `sr/metrics.py`)

* Each HR test image is centre-cropped to 256×256, degraded to 64×64, upscaled by every model, and
  compared with the original crop.
* **PSNR** and **SSIM** (scikit-image, all colour channels) and **LPIPS** (AlexNet backbone).
* Outputs: `results/metrics.csv` (per image), `results/summary.csv` (averages) and
  `results/compare_*.png` (input | each model | ground truth).

### 3.8 Inference (`upscale.py`, `app.py`, `sr/infer.py`)

* Large images are processed **tile by tile** with overlap, so they fit in 4 GB of GPU memory.
* `upscale.py` handles single images and videos (frame by frame with OpenCV).
* `app.py` is a Streamlit demo with a side-by-side comparison and a download button.

### 3.9 Technology

Python 3.10+, PyTorch, torchvision, OpenCV, scikit-image, LPIPS, Streamlit, `realesrgan` + `basicsr`
(optional). Developed on a laptop with an NVIDIA RTX 3050 (4 GB). A free Google Colab / Kaggle GPU
also works for training.

### 3.10 Notes and design choices

* **EDSR** is a lightweight variant (16 blocks / 64 channels) with one deliberate change from the
  paper: a **bicubic skip connection**, so the network learns only the missing detail and converges
  quickly on short runs. For the paper-size model use `EDSR(n_blocks=32, n_feats=256)`; it will not
  fit on a 4 GB GPU.
* **SRCNN** also predicts a residual on top of the bicubic image.
* The ESRGAN generator output is not clamped during training (clamping would zero the gradients); it
  is clamped to [0, 1] when saved or measured.
* Real-ESRGAN weights (~64 MB) download automatically on first use, so that step needs internet.
* LPIPS also downloads small pretrained weights on first use; if it can't, the code reports
  PSNR/SSIM only.
* Train and test on the same degradation strength, or train on `heavy` for robustness. Try
  `--strength heavy` for harsher CCTV conditions.

---

## 4. Step-by-step: how to run the code

Commands are shown for Windows PowerShell; on Linux/macOS they are identical (use `/` in paths).

### Step 0: Requirements

* Python 3.10 or newer, and about 5 GB free disk space.
* Recommended: an NVIDIA GPU. Everything also runs on the CPU, but training is slow.

### Step 1: Get the project and install packages

```powershell
cd surveillance_sr
pip install -r requirements.txt
```

### Step 2: Enable the GPU (NVIDIA only)

Check that PyTorch can see your GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

If this prints `False` (or the version ends in `+cpu`), install the CUDA build from
<https://pytorch.org/get-started/locally/> (pick your OS, Pip and a CUDA version), for example:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Download the dataset

Download **DIV2K training HR images** (800 photos, ~3.5 GB) from the official page
<https://data.vision.ee.ethz.ch/cvl/DIV2K/> (see [Dataset](#5-dataset)) and unzip it, e.g. to
`C:\datasets\DIV2K_train_HR`.

### Step 4: Split into train / validation / test

```powershell
python prepare_data.py --src C:\datasets\DIV2K_train_HR --dst data
```

This creates `data/train` (640 images), `data/val` (80) and `data/test` (80).

### Step 5: Cut sub-images (recommended, makes training much faster)

```powershell
python make_subimages.py --src data/train --dst data_sub/train
```

### Step 6: Train SRCNN

```powershell
python train.py --model srcnn --train_dir data_sub/train --val_dir data/val --epochs 30 --workers 2
```

Saves `checkpoints/srcnn_x4.pt`. If a Windows multiprocessing error appears, add `--workers 0`.
Out of GPU memory? Lower `--batch` (default 16) to 8.

### Step 7: Evaluate and compare models

Bicubic and SRCNN only:

```powershell
python evaluate.py --test_dir data/test --ckpt checkpoints/srcnn_x4.pt --save_dir results
```

Add the pretrained Real-ESRGAN (needs `pip install realesrgan basicsr` and internet on the first run;
the 64 MB weights are saved in `weights/`):

```powershell
pip install realesrgan basicsr
python evaluate.py --test_dir data/test --ckpt checkpoints/srcnn_x4.pt --realesrgan --save_dir results
```

Look in `results/`: `summary.csv` (the table for your report) and `compare_*.png` (pictures).
LPIPS downloads a small backbone on first use; if offline, PSNR and SSIM are still reported.

### Step 8: Run the demo app

```powershell
streamlit run app.py
```

Upload an image. Tick **simulate CCTV degradation** to upload a sharp photo and see metrics against
the original; leave it unticked to upscale a real low-resolution frame.

### Step 9: Upscale your own image or video

```powershell
python upscale.py --input frame.png --output frame_x4.png --model srcnn --ckpt checkpoints/srcnn_x4.pt
python upscale.py --input cctv.mp4  --output cctv_x4.mp4  --model realesrgan
```

`--model` can be `bicubic`, `srcnn`, `edsr`, `esrgan` or `realesrgan`. Checkpoint-based models need `--ckpt`.

### Step 10 (optional): Fine-tune ESRGAN on your degradation

Needs `weights/RealESRGAN_x4plus.pth` (created automatically by the Real-ESRGAN evaluation in Step 7).

```powershell
python train_esrgan.py --train_dir data_sub/train --val_dir data/val --checkpoint --workers 2 --iters_l1 5000 --iters_gan 10000
```

The first progress lines show seconds per iteration and an estimated time left; lower `--iters_l1` and
`--iters_gan` if it is too slow. Ctrl+C is safe: the latest checkpoint is saved every 500 iterations.
Then compare everything:

```powershell
python evaluate.py --test_dir data/test --ckpt checkpoints/srcnn_x4.pt checkpoints/esrgan_x4_l1.pt checkpoints/esrgan_x4.pt --realesrgan --save_dir results
```

Optionally blend the two ESRGAN checkpoints and evaluate the in-between models:

```powershell
python interpolate_models.py --l1 checkpoints/esrgan_x4_l1.pt --gan checkpoints/esrgan_x4.pt
python evaluate.py --test_dir data/test --ckpt checkpoints/esrgan_interp_0.2.pt checkpoints/esrgan_interp_0.6.pt --save_dir results_interp
```

### Quick smoke test (no downloads)

To confirm the setup works before downloading anything:

```powershell
python prepare_data.py --synthetic --dst data_demo
python train.py --model srcnn --train_dir data_demo/train --val_dir data_demo/val --epochs 3 --patch 64 --batch 8 --workers 0
python evaluate.py --test_dir data_demo/test --ckpt checkpoints/srcnn_x4.pt
```

The synthetic images are procedural shapes, so the scores are meaningless; this only checks the code runs.

### Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `--batch` (e.g. 8, 4 or 2); for ESRGAN also add `--checkpoint` or `--no_vgg`. |
| Dataloader or multiprocessing error on Windows | Add `--workers 0`. |
| Training is very slow | Run Step 5 and train from `data_sub/train`; check `device: cuda` is printed; raise `--workers`. |
| `weights/RealESRGAN_x4plus.pth not found` | Run the Real-ESRGAN evaluation once (Step 7), or pass the file location with `--pretrained <path>`. |
| LPIPS shows `nan` | The AlexNet weights could not be downloaded (no internet); PSNR and SSIM are still valid. |
| Real-ESRGAN import error | `pip install realesrgan basicsr` (a compatibility patch for new torchvision is built in). |

---

## 5. Dataset

**DIV2K** (DIVerse 2K resolution image dataset), 800 training + 100 validation high-resolution photos.

* Official page: <https://data.vision.ee.ethz.ch/cvl/DIV2K/>
* Direct download used in this project (training HR images):
  <https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip>
* Mirror on Hugging Face: <https://huggingface.co/datasets/eugenesiow/Div2k>

Licence note: DIV2K is provided for **academic research purposes only**; copyright belongs to the
original image owners.

For a more surveillance-specific study, mix in face images (e.g. CelebA, WIDER FACE) and vehicle /
number-plate images from a public dataset; the pipeline accepts any folder of high-resolution images.
No real CCTV dataset is needed, because the CCTV look is simulated by `sr/degrade.py`.

---

## 6. Limitations

* Super-resolution **infers** plausible detail; it does not recover information the camera never
  captured. GAN-based outputs can look sharp while inventing wrong textures, characters or facial
  features (a hallucinated digit on a number plate is a real risk). Upscaled frames are an aid for
  human review, **not forensic evidence**. A good extra experiment: run OCR on plates (or face
  recognition) before vs. after upscaling and check both accuracy and the wrong-but-confident cases.
* The degradation is simulated, so results on real CCTV footage may differ.
* The pretrained Real-ESRGAN was not trained on our degradation, so comparing it with models trained
  on it is not strictly like for like.
* Results come from one training run and 80 test images; treat differences of a few tenths of a dB as
  small.

---

## 7. Suggested extensions

* Fine-tune on faces / plates only (domain-specific data helps a lot).
* Add **SwinIR** (`pip install basicsr`, pretrained weights available) as a transformer comparison.
* Train **EDSR** (`python train.py --model edsr ...`) to complete the model comparison.
* Face restoration with **GFPGAN / CodeFormer** on detected face crops.
* **Multi-frame SR**: align neighbouring video frames before upscaling.
* Downstream test: OCR on number plates and face recognition, before vs. after upscaling.

---

## 8. References

* Agustsson & Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study* (DIV2K).
* Dong et al., *Image Super-Resolution Using Deep Convolutional Networks* (SRCNN), 2014.
* Lim et al., *Enhanced Deep Residual Networks for Single Image Super-Resolution* (EDSR), 2017.
* Wang et al., *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*, 2018.
* Wang et al., *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*, 2021.
* Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS), 2018.