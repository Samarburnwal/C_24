# Low-Resolution Surveillance Image Upscaling using Super-Resolution

Reconstruct higher-resolution (4x) images from blurry, noisy, compressed CCTV-style frames
and measure how much each method helps.

## What's included

| File | Purpose |
|---|---|
| `sr/degrade.py` | Simulates CCTV degradation: Gaussian/motion blur -> 4x downscale -> sensor noise -> JPEG compression (3 strengths) |
| `sr/models.py` | Bicubic baseline, **SRCNN**, compact **EDSR** |
| `sr/data.py` | On-the-fly (LR, HR) pair dataset with random crops + flips |
| `sr/metrics.py` | PSNR, SSIM, LPIPS |
| `sr/realesrgan_wrapper.py` | Optional pretrained **Real-ESRGAN** (GAN-based, real-world degradation) |
| `prepare_data.py` | Split your HR images into train/val/test (or make a synthetic smoke-test set) |
| `train.py` | Train SRCNN / EDSR |
| `evaluate.py` | Compare all models -> `metrics.csv`, `summary.csv`, side-by-side images |
| `upscale.py` | Upscale one image or a whole video, frame by frame |
| `app.py` | Streamlit demo (upload -> before/after for several models) |

## Setup

```bash
pip install -r requirements.txt
# optional, for Real-ESRGAN:
pip install realesrgan basicsr
```
A GPU is strongly recommended for training (free Google Colab / Kaggle T4 is enough).

## Step-by-step

### 1. Get data
Download high-resolution images, e.g. **DIV2K** (800 train images), **Flickr2K**, plus domain images
(faces from CelebA/WIDER FACE, vehicles/number plates from a public plate dataset).
Put them all in one folder and split:
```bash
python prepare_data.py --src path/to/hr_images --dst data      # -> data/train, data/val, data/test
```
You never create low-res images by hand: `degrade.py` generates them on the fly from the HR images.

### 2. Train
```bash
python train.py --model srcnn --train_dir data/train --val_dir data/val --epochs 30
python train.py --model edsr  --train_dir data/train --val_dir data/val --epochs 100 --strength medium
```
Checkpoints go to `checkpoints/`. Try `--strength heavy` for harsher CCTV conditions
(train and test on the same strength, or train on `heavy` for robustness).

### 3. Evaluate
```bash
python evaluate.py --test_dir data/test --ckpt checkpoints/srcnn_x4.pt checkpoints/edsr_x4.pt --realesrgan
```
Prints a PSNR / SSIM / LPIPS table and saves `results/compare_*.png`
(input | Bicubic | SRCNN | EDSR | Real-ESRGAN | ground truth). Put these in your report.

### 4. Demo
```bash
streamlit run app.py
```
Tick "simulate CCTV degradation" to upload a sharp photo and get metrics against the original.
Untick it to upscale a real low-res frame.

### 5. Video
```bash
python upscale.py --input cctv.mp4 --output cctv_x4.mp4 --model edsr --ckpt checkpoints/edsr_x4.pt
```

## How to read the results
* **PSNR/SSIM** reward pixel-faithfulness. L1-trained models (SRCNN, EDSR) score highest here.
* **LPIPS** (lower = better) and visual inspection reward perceptual sharpness. Real-ESRGAN usually
  wins here but often **loses on PSNR** - this PSNR vs. perceptual trade-off is a standard
  discussion point for your report.
* Always compare against **Bicubic**: a learned model that does not beat it is not working.

## Notes and design choices
* EDSR here is a lightweight variant (16 blocks / 64 channels) with one deliberate change from the
  paper: a **bicubic skip connection**, so the network learns only the missing detail and converges
  quickly on short runs. For the paper-size model use `EDSR(n_blocks=32, n_feats=256)`.
* SRCNN also predicts a residual on top of bicubic.
* Real-ESRGAN weights (~64 MB) download automatically on first use, so that step needs internet.
* LPIPS also downloads small pretrained weights on first use; if it can't, the code reports PSNR/SSIM only.

## Important limitation (include this in your report)
Super-resolution **infers** plausible detail; it does not recover information that the camera never
captured. GAN-based outputs in particular can look sharp while inventing wrong textures, characters
or facial features. Upscaled surveillance frames should therefore be treated as an aid for human
review, **not as forensic evidence** (a hallucinated digit on a number plate is a real risk).
A good extra experiment: run OCR on plates (or face recognition) before vs. after upscaling and
check both accuracy and the *wrong-but-confident* cases.

## Suggested extensions
* Fine-tune on faces / plates only (domain-specific data helps a lot).
* Add SwinIR (`pip install basicsr`, pretrained weights available) as a transformer comparison.
* Face restoration with GFPGAN / CodeFormer on detected face crops.
* Multi-frame SR: align neighbouring video frames before upscaling.
