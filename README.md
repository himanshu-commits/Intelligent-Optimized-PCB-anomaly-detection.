# Intelligent-Optimized-PCB-anomaly-detection.

## Overview
This project focuses on detecting anomalies in Printed Circuit Boards (PCBs) using image processing and machine learning techniques. The implemented model identifies missing components, misalignments, and soldering defects while maintaining robustness to variations in lighting and orientation.

## Approach and Methodology
The methodology consists of three main stages:

1. **Preprocessing** – Image extraction, alignment, and enhancement.
2. **Defect Detection** – Feature extraction and clustering using PCA-Kmeans.
3. **Post-Processing** – Analysis of detected changes to remove false positives.

## Preprocessing Steps
### 1. Image Registration using SIFT and RANSAC
- Extract key features using Scale-Invariant Feature Transform (SIFT).
- Perform feature matching and eliminate incorrect matches using the Ratio Test.
- Compute the best transformation matrix using RANSAC to align images.

### 2. Histogram Matching for Illumination Correction
- Apply Exact Histogram Specification to match color and brightness distribution.
- Reduce false defect detections caused by lighting variations.

## Model/Algorithm Used – PCA-Kmeans Change Detection
The defect detection algorithm follows an unsupervised learning approach:

### 1. Feature Extraction using Change Descriptors
- RGB channel difference.
- Grayscale difference.
- Window-based extraction around each pixel.

### 2. Dimensionality Reduction using Principal Component Analysis (PCA)
- Reduce feature dimensionality while retaining key information.

### 3. K-Means Clustering for Change Classification
- Cluster pixels based on defect-related features.
- Assign defect confidence levels (high, medium, low).

### 4. Post-Processing using MSE Heuristic and DBSCAN
- Analyze clusters using Mean Squared Error (MSE) to filter out noise.
- Use Density-Based Spatial Clustering (DBSCAN) to remove false positives.

## Evaluation Metrics
- **Recall:** 0.82 – Detects most real PCB defects, minimizing missed defects.
- **Precision:** 0.86 – Some false positives exist but are relatively minor.
- Metrics are computed using the `evaluation.py` script.

## Challenges and Potential Improvements
### Challenges:
- Imaging variability (lighting, angles, reflections) affecting accuracy.
- Sensitivity to registration errors.
- Detecting very fine PCB defects.

### Potential Improvements:
- Improved image processing and registration techniques.
- Supervised fine-tuning using labeled data.
- Exploring deep learning-based segmentation and domain adaptation.

## Installation & Setup
To run the scripts, create a Conda environment and install the required dependencies:
```bash
conda install pytorch torchvision -c pytorch
conda install numpy scipy matplotlib
conda install opencv pillow scikit-learn scikit-image
conda install keras tensorflow
```

## Running the Model
- Execute the `./run_example.sh` script to run the model and detect defects.
- The `clustering_data.csv` file has been removed due to size constraints.

## Project Files
- `main.py` – Main script executing the application.
- `registration.py` – Handles image registration.
- `light_differences_elimination.py` – Corrects lighting differences.
- `evaluation.py` – Evaluates model performance.
- `PCA_Kmeans.py` – Implements PCA and K-means clustering.
- `global_variables.py` – Contains shared global variables.
- `histogram_matching.py` – Aligns pixel value distributions.
- `run_example.sh` – Shell script to execute the model.

## References
1. David G. Lowe – SIFT Feature Matching.
2. Konstantinos G Derpanis – RANSAC Algorithm.
3. Philippe Bolon Coltuc et al. – Histogram Specification.
4. Turgay Celik – PCA and K-Means Clustering.
5. Kamran Khan et al. – DBSCAN Clustering.

---
**Author:** Himanshu Gupta  
**Email:** himanshu.gupta@tu-ilmenau.de

