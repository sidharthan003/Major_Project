# Custom Sonar Detection Framework (YOLO-Swin + ExDeepSORT)

## Overview
This document details the custom, framework-agnostic detection and tracking architecture designed specifically for high-noise acoustic environments. [cite_start]It mitigates the limitations of standard convolutions by integrating Swin Transformer Windowed Multi-Head Self-Attention (W-MSA) into the YOLO backbone[cite: 52, 199]. [cite_start]It also incorporates bounding-box expansion (ExDeepSORT) to capture vital acoustic scattering noise prior to tracking[cite: 480, 481]. 

## Structural Framework
* **Backbone:** Custom modified architecture. The deep local feature extraction blocks (e.g., C2f) are replaced with a custom `C2f_Swin` module. [cite_start]This forces the network to fuse global image context, preventing the loss of sparse human features in sonar noise[cite: 246, 283].
* **Neck & Head:** Standard YOLO architecture to maintain compatibility with the loss functions and bounding box regression logic.
* [cite_start]**Tracker Adapter (SonarBoxExpander):** An intermediate logic layer that intercepts the YOLO output and expands the bounding box dimensions by a ratio of $R=3$[cite: 525, 528, 550]. This expanded box is fed into the local tracker.

## Implementation Steps
1.  **Module Definition:** Write the PyTorch `C2f_Swin` class, wrapping standard YOLO convolutions around a Swin Transformer block.
2.  **YAML Configuration:** Create a custom `yolov8n-sonar-swin.yaml` file, swapping the standard deep C2f backbone blocks with the registered `C2f_Swin` module.
3.  **Training:** Train the model from scratch on the sonar dataset using the custom YAML architecture.
4.  **Adapter Integration:** Implement the `SonarBoxExpander` logic to mathematically scale the output bounding box width and height.
5.  **Payload Generation:** Format the final tracked output into the standardized JSON payload required by the central $P_{human}$ fusion engine.

## Metrics for Justification (Expected Deltas)
Compare these results against the baseline to justify the architectural shift:
* **mAP & Recall Improvement:** The Swin Transformer integration should yield a measurable increase in both mAP and Recall. [cite_start]Prior research replacing C3 with STr blocks increased mAP from 92.6% to 94.4% and effectively eliminated misdetections of human targets[cite: 281, 282, 421].
* [cite_start]**Heatmap Concentration:** Grad-CAM heatmaps must demonstrate a tighter, more centralized focus on the actual target, proving the global attention mechanism successfully filters out background reverberation noise[cite: 434, 435].
* **Tracking Stability (Frag Ratio & ID Switch):** The ExDeepSORT expansion logic should drastically reduce tracking interruptions. [cite_start]Expect ID switches to drop near zero for continuous targets, as the expanded box captures necessary material shape noise that standard denoising destroys[cite: 56, 481, 847].
* **Hardware Overhead (FPS Drop):** Record the exact drop in FPS. Transformers are computationally heavier. [cite_start]You must prove that the FPS remains above the minimum operational threshold for your two-tier sensing protocol on the Jetson Nano[cite: 278, 281].