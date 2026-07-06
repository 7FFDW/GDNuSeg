# GDNuSeg

Official implementation of **GDNuSeg: Integrating GLCM-based Uncertainty Learning with Density Map Guidance for Nuclei Segmentation**.

GDNuSeg is a nuclei segmentation framework that combines **GLCM-based statistical texture learning** with **density-map-guided positional prompts**. The model is designed to enhance the representation of nuclear texture patterns and spatial distribution cues for robust nuclei segmentation in histopathological images.

---

## Overview

Nuclei segmentation is a fundamental task in computational pathology. However, accurate segmentation remains challenging due to complex nuclear morphology, dense cell distribution, blurred boundaries, and heterogeneous tissue textures.

GDNuSeg addresses these challenges using two complementary components:

- **Dirichlet Texture Learner (DTL)**  
  Learns GLCM-derived statistical texture information and adaptively fuses it with deep semantic features through uncertainty-aware weighting.

- **Position Hint Encoder (PH-Encoder)**  
  Generates nuclear density maps from ground-truth masks during training and uses them as positional and semantic prompts to guide feature learning.

During training, the model uses image, GLCM texture maps, and density maps. During validation and testing, density maps are not required.

---

## Main Features

- GLCM-based statistical texture learning
- Density-map-guided positional prompt encoding
- Dirichlet uncertainty-based feature fusion
- Support for binary nuclei segmentation
- Support for multi-class nuclei segmentation
- Training-stage density map guidance
- No density map required during inference
- Compatible with datasets such as MoNuSeg, TNBC, DSB2018, CoNIC, and CoNSeP

---

## Network Input

The model takes the following inputs:

```python
output = model(image, glcm, densitymap)
