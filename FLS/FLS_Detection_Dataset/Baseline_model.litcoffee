# Baseline Sonar Detection Framework (Standard YOLO)

## Overview
This document outlines the baseline detection architecture for the sonar processing module of the multi-environment human detection system. It utilizes a standard, unmodified YOLO framework (starting with YOLOv8n) to establish benchmark metrics for detecting human presence in submerged, high-noise environments before the data is passed to the $P_{human}$ probability fusion engine.

## Structural Framework
* **Backbone:** Standard CSPDarknet / C2f (Cross Stage Partial Bottleneck) blocks. Relies entirely on local convolutional operations for feature extraction.
* **Neck:** Standard PANet (Path Aggregation Network) for multiscale feature fusion.
* **Head:** Decoupled head for objectness and bounding box regression.
* **Tracker Adapter:** Raw YOLO bounding boxes are passed directly to the standard tracking module without spatial expansion.

## Implementation Steps
1.  **Dataset Preparation:** Format the annotated sonar dataset (e.g., diver/human motion sequences) into standard YOLO format.
2.  **Environment Setup:** Initialize the standard Ultralytics YOLO environment. 
3.  **Training:** Train the base model (e.g., `yolov8n.pt`) on the sonar dataset. 
4.  **Inference & Tracking:** Run the trained weights on test sequences. 
5.  **Data Export:** Output the bounding box coordinates, class IDs, and confidence scores to be logged for metric comparison.

## Key Performance Metrics (Baseline)
To establish the baseline, record the following metrics:
* **mAP (Mean Average Precision) @ 0.5:** The standard measure of detection accuracy. 
* **Recall:** Crucial for life-critical systems; measures the percentage of actual humans successfully detected. [cite_start]Traditional CNN structures often struggle with missed detections (false negatives) in noisy sonar data[cite: 420, 425].
* **FPS (Frames Per Second):** The baseline inference speed on the target edge hardware (e.g., Jetson Nano).
* **Heatmap Focus:** Generate Grad-CAM heatmaps to visualize where the network is looking. [cite_start]Standard convolutions often focus on bright noise artifacts rather than the sparse target features[cite: 435].
* [cite_start]**Frag Ratio (Trajectory Interruption):** The percentage of tracking frames where the target ID is lost[cite: 742, 743].
* [cite_start]**ID Switches:** The number of times the tracker assigns a new ID to the same continuous target[cite: 739].