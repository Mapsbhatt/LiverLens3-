# 🧬 LiverLens3+ – Advanced MRI-based HCC Segmentation Framework

**A lightweight and accurate deep learning model for Hepatocellular Carcinoma (HCC) segmentation in liver MRIs, built on an enhanced UNet3+ backbone with depthwise convolutions and multi-attention blocks.**

---

## 🧠 Background

Hepatocellular carcinoma (HCC) is the most common primary liver cancer. Accurate segmentation from MRI is crucial for early diagnosis, treatment planning, and prognosis. **LiverLens3+** addresses key challenges:

- ⚖️ Class imbalance (tiny tumors vs large liver regions)  
- 🧠 Precise small-lesion segmentation  
- 🏃 Efficient inference for real-world hospital deployment

---

## 🏗️ Model Architecture

- Based on **UNet3+** with **dense skip connections**
- **Depthwise separable convolutions** for reduced parameter count
- **Multi-attention gates** for guided feature refinement
- **Multi-scale aggregation** for robust context understanding

### 🧩 Components

| Block | Description |
|-------|-------------|
| `multi_attention_block()` | Computes adaptive attention maps between paths |
| `depthwise_conv_block()` | Two-layer depthwise-separable CNN block |
| `unetpp()` | Main model definition with skip-connections, attention, and multi-resolution fusion |

> 📦 Total Parameters: ~3.5M  
> 🧠 Ideal for 2D axial slices of MRI (can be adapted for 3D input)

---

## 📊 Performance Highlights

| Metric | Tumor Seg. | Liver Seg. |
|--------|------------|------------|
| Dice Score | 0.863 | 0.995 |
| Accuracy | 98.86% | 99.4% |

> 📈 Evaluated on real-world clinical dataset with MRI scans from multiple contrast phases: **Plain, Arterial, PV, Delay**

---

