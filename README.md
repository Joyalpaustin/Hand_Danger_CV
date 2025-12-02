# Hand Danger Detection – Computer Vision POC  
### Arvyax Internship Assignment (Real-Time CV Prototype)

This project demonstrates a **real-time hand tracking system** using **classical computer vision techniques**, without MediaPipe, OpenPose, or any cloud-based pose APIs.  
The system detects the user's hand from a webcam feed and triggers a clear on-screen warning when the hand approaches a virtual danger boundary.

---

## 🚀 Features Implemented

### ✔ Real-time Hand Tracking (CPU Only)
- Uses **skin-color segmentation** in YCrCb color space  
- Morphological operations to clean noise  
- Finds the **largest contour** → considered as the hand  
- Computes:
  - Centroid of the contour  
  - Fingertip → farthest point from the centroid  

### ✔ Virtual Boundary on Screen
- A vertical line drawn at 75% width of the frame  
- Hand distance to this line determines the state  

### ✔ State Classification (Dynamic Logic)
- **SAFE** – hand far from boundary  
- **WARNING** – hand approaching boundary  
- **DANGER** – hand extremely close or touching  

### ✔ Visual Overlays
- State label on screen  
- Distance in pixels  
- Hand contour, fingertip, and centroid markers  
- BIG RED **“DANGER DANGER”** text when danger is triggered  

### ✔ Performance
- Achieves **≥ 8 FPS** using CPU only  
- No GPU required  
- Uses lightweight operations (OpenCV + NumPy)

---

## 🧠 Approach Summary

1. Capture live webcam feed using OpenCV  
2. Convert each frame to **YCrCb** → apply **skin mask**  
3. Clean mask using **Gaussian blur + morphology**  
4. Extract the **largest hand contour**  
5. Compute:
   - Centroid using image moments  
   - Fingertip as farthest contour point  
6. Draw a virtual danger boundary  
7. Compute fingertip-to-boundary distance  
8. Classify into:
   - SAFE  
   - WARNING  
   - DANGER  
9. Overlay visual feedback and run in real time  

---

## 📂 Project Files

# Hand Danger Detection – Computer Vision POC  
### Arvyax Internship Assignment (Real-Time CV Prototype)

This project demonstrates a **real-time hand tracking system** using **classical computer vision techniques**, without MediaPipe, OpenPose, or any cloud-based pose APIs.  
The system detects the user's hand from a webcam feed and triggers a clear on-screen warning when the hand approaches a virtual danger boundary.

---

## 🚀 Features Implemented

### ✔ Real-time Hand Tracking (CPU Only)
- Uses **skin-color segmentation** in YCrCb color space  
- Morphological operations to clean noise  
- Finds the **largest contour** → considered as the hand  
- Computes:
  - Centroid of the contour  
  - Fingertip → farthest point from the centroid  

### ✔ Virtual Boundary on Screen
- A vertical line drawn at 75% width of the frame  
- Hand distance to this line determines the state  

### ✔ State Classification (Dynamic Logic)
- **SAFE** – hand far from boundary  
- **WARNING** – hand approaching boundary  
- **DANGER** – hand extremely close or touching  

### ✔ Visual Overlays
- State label on screen  
- Distance in pixels  
- Hand contour, fingertip, and centroid markers  
- BIG RED **“DANGER DANGER”** text when danger is triggered  

### ✔ Performance
- Achieves **≥ 8 FPS** using CPU only  
- No GPU required  
- Uses lightweight operations (OpenCV + NumPy)

---

## 🧠 Approach Summary

1. Capture live webcam feed using OpenCV  
2. Convert each frame to **YCrCb** → apply **skin mask**  
3. Clean mask using **Gaussian blur + morphology**  
4. Extract the **largest hand contour**  
5. Compute:
   - Centroid using image moments  
   - Fingertip as farthest contour point  
6. Draw a virtual danger boundary  
7. Compute fingertip-to-boundary distance  
8. Classify into:
   - SAFE  
   - WARNING  
   - DANGER  
9. Overlay visual feedback and run in real time  

---

## 📂 Project Files

