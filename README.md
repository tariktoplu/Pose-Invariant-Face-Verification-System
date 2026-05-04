# Pose-Invariant Face Verification (Classical CV)

This project is a face verification system designed to handle extreme occlusion (masks, sunglasses) and pose variations using **strictly classical computer vision**—no Deep Learning or Neural Networks were used.

## My Approach
The core challenge was to prevent occlusions from corrupting the feature matching process. I implemented a modular pipeline to isolate the "clean" facial features from the "noise" (masks).

### 1. Robust Detection & Alignment
Standard detectors often fail on masked faces. I used an ensemble approach:
*   **Primary:** Dlib HOG detector.
*   **Fallback:** Haar Cascades (Frontal/Profile) and eye-pair extrapolation to guess face bounds when the mouth is covered.
*   **Alignment:** 68-point landmarks to perform a 2D Affine transform, normalizing faces to a standard 300x300 grid.

### 2. Grid-Based Occlusion Masking
Instead of global matching, I divided the face into a **6x6 grid**. I used YCrCb skin-color thresholding and entropy checks to flag occluded blocks. 
*   **Why?** This allows the system to ignore keypoints detected on a mask, focusing only on the visible skin (eyes/forehead).

### 3. Mask-Aware Matching
*   **Features:** SIFT descriptors are extracted only from "valid" (non-occluded) blocks.
*   **Verification:** Lowe's Ratio Test followed by **RANSAC**. The system requires a minimum number of inliers (geometrically consistent matches) to verify identity.

## Performance (Masked-LFW Dataset)
Tested on 200 pairs from the Masked-LFW dataset using a Single Sample Per Person (SSPP) protocol:

*   **Accuracy:** ~48.00%
*   **False Acceptance Rate (FAR):** ~3% (High security; almost zero imposter pass)
*   **False Rejection Rate (FRR):** ~90% (Strict; fails on extreme yaw rotations due to 2D limits)

**Conclusion:** The system is biased toward high security. It successfully ignores occluders (Low FAR) but, without a 3D Morphable Model (3DMM) for pose recovery, it defaults to rejecting extreme profile views rather than risking a false match.

## Quick Start
1. `pip install -r requirements.txt`
2. Run evaluation: `python evaluate_dataset.py`
3. Single test: `python face_verification.py <img1> <img2>`
