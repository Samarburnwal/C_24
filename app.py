"""Streamlit demo.   Run:  streamlit run app.py"""
import glob
import os

import cv2
import numpy as np
import streamlit as st
import torch

from sr.degrade import degrade
from sr.models import Bicubic, build_model
from sr.data import to_bgr, to_tensor

st.set_page_config(page_title="Surveillance Image Super-Resolution", layout="wide")
st.title("Low-Resolution Surveillance Image Upscaling")
st.caption("Super-resolution is *inference*, not recovery: output detail is plausible, not guaranteed real.")

dev = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load(name, path=None):
    if name == "Bicubic":
        return Bicubic(4).eval()
    if name == "Real-ESRGAN":
        from sr.realesrgan_wrapper import RealESRGANUpscaler
        return RealESRGANUpscaler(4, dev)
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["model"], ck["scale"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval()


def run(name, path, bgr):
    m = load(name, path)
    if name == "Real-ESRGAN":
        return m.upscale(bgr)
    with torch.no_grad():
        return to_bgr(m(to_tensor(bgr).unsqueeze(0).to(dev))[0])


ckpts = {os.path.basename(p): p for p in glob.glob("checkpoints/*.pt")}
options = ["Bicubic"] + list(ckpts) + ["Real-ESRGAN"]
chosen = st.sidebar.multiselect("Models to compare", options, default=["Bicubic"] + list(ckpts)[:2])
simulate = st.sidebar.checkbox("Input is a high-res photo: simulate CCTV degradation first", value=False)
strength = st.sidebar.selectbox("Degradation strength", ["light", "medium", "heavy"], index=1)

up = st.file_uploader("Upload a low-resolution frame (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp"])
if up is not None:
    bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    hr_ref = None
    if simulate:
        hr_ref = bgr[: bgr.shape[0] // 4 * 4, : bgr.shape[1] // 4 * 4]
        bgr = degrade(hr_ref, 4, strength)
    st.write(f"Input size: {bgr.shape[1]}x{bgr.shape[0]}  ->  output {bgr.shape[1]*4}x{bgr.shape[0]*4}")
    cols = st.columns(len(chosen) + 1)
    cols[0].image(cv2.cvtColor(cv2.resize(bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
                               cv2.COLOR_BGR2RGB), caption="Input (nearest-neighbour view)")
    for col, name in zip(cols[1:], chosen):
        with st.spinner(name):
            out = run("Real-ESRGAN" if name == "Real-ESRGAN" else
                      ("Bicubic" if name == "Bicubic" else name), ckpts.get(name), bgr)
        cap = name
        if hr_ref is not None and out.shape == hr_ref.shape:
            from sr.metrics import psnr, ssim
            cap += f"  |  PSNR {psnr(out, hr_ref):.2f} dB, SSIM {ssim(out, hr_ref):.3f}"
        col.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=cap)
        ok, png = cv2.imencode(".png", out)
        col.download_button("Download", png.tobytes(), file_name=f"sr_{name}.png", mime="image/png",
                            key=f"dl_{name}")
else:
    st.info("Upload an image to begin. Train models first with train.py to see them in the list.")
