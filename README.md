# VLALeaks
VLALeaks: Membership Inference Attacks against Vision-Language-Action Models

<div align="center">
<img src="overview.png" width="750" alt="Overview of VLALeaks.">
</div>

## Overview
💡 VLALeaks is a membership inference attack targeting vision-language-action models.
The attack pipeline is divided into two strictly sequential stages:
1. **Stage 1**: Extract sensitive membership leakage features from the target VLA model.
2. **Stage 2**: Train and evaluate a binary attack model to distinguish member and non-member samples.

## Environment Setup
### Basic Hardware & Software Requirements
- Python = 3.10
- PyTorch = 2.2.0
- CUDA = 12.1

### Install Dependencies

🧠 This project is built on top of [OpenVLA](https://github.com/openvla/openvla), so please follow its installation instructions to configure the base environment first.

🧪 Experiments are conducted in the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation environment. Make sure to install LIBERO and its dependencies as described in their official documentation. Additional support can be provided by [LIBERO+](https://github.com/sylvestf/LIBERO-plus).

## Run Attack Pipeline
⚠️ Execution order cannot be reversed: Stage 1 must finish before launching Stage 2

### Stage 1: Membership Feature Extraction
Script path: 
```bash
vla-scripts/vlaleaks_stage1.py
```
### Stage 2: Attack Model Construction
Script path: 
```bash
vla-scripts/vlaleaks_stage2.py
```
