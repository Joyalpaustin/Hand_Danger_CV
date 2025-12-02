
"""
Arvyax Internship Assignment - Hand Danger POC
Author: Joyal Paustin
Tech: Python, OpenCV, NumPy
Description: Tracks hand using skin-color segmentation and shows SAFE/WARNING/DANGER
based on distance to a virtual boundary line.
"""

import cv2
import numpy as np
import time

# ------------- CONFIG -------------
# Virtual boundary position (as a fraction of frame width)
BOUNDARY_POS_X_RATIO = 0.75  # 75% of width (right side)

# Distance thresholds in pixels (these will depend on your camera resolution)
SAFE_THRESHOLD = 150    # > this = SAFE
DANGER_THRESHOLD = 60   # < this = DANGER; between = WARNING

# Minimum contour area to consider as a valid hand
MIN_HAND_AREA = 3000

# ------------- UTILS -------------
def get_fingertip_point(contour):
    """
    Approximate fingertip as the contour point farthest from the contour centroid.
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = np.array([cx, cy])

    # Flatten contour points
    pts = contour.reshape(-1, 2)
    # Compute squared distances from centroid
    dists = np.sum((pts - center) ** 2, axis=1)
    max_idx = np.argmax(dists)
    fx, fy = pts[max_idx]

    return (cx, cy), (int(fx), int(fy))


def classify_state(distance):
    if distance is None:
        return "SAFE"  # No hand detected -> treat as SAFE

    if distance > SAFE_THRESHOLD:
        return "SAFE"
    elif distance > DANGER_THRESHOLD:
        return "WARNING"
    else:
        return "DANGER"


# ------------- MAIN -------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # For a rough FPS estimate
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror view

        h, w, _ = frame.shape
        boundary_x = int(w * BOUNDARY_POS_X_RATIO)

        # --- Skin detection (YCrCb) ---
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

        # Skin color range in YCrCb (might need slight tuning depending on lighting)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)

        mask = cv2.inRange(ycrcb, lower, upper)

        # Morphological operations to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # --- Find contours (hand candidate) ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_contour = None
        fingertip = None
        center = None
        distance_to_boundary = None

        if contours:
            # Largest contour by area
            hand_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(hand_contour)

            if area > MIN_HAND_AREA:
                # Draw contour
                cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)

                center, fingertip = get_fingertip_point(hand_contour)

                # If we got a fingertip, compute distance to boundary line
                if fingertip is not None:
                    fx, fy = fingertip

                    # Distance in x-direction to the virtual boundary
                    if fx >= boundary_x:
                        distance_to_boundary = 0  # touching / crossing
                    else:
                        distance_to_boundary = boundary_x - fx

                    # Draw fingertip
                    cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)
                    # Draw center
                    if center is not None:
                        cv2.circle(frame, center, 5, (0, 255, 255), -1)

        # --- Classify interaction state ---
        state = classify_state(distance_to_boundary)

        # --- Draw virtual boundary ---
        cv2.line(frame, (boundary_x, 0), (boundary_x, h), (0, 255, 255), 2)
        cv2.putText(frame, "Virtual Boundary", (boundary_x - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Overlay status text ---
        if state == "SAFE":
            color = (0, 255, 0)
        elif state == "WARNING":
            color = (0, 255, 255)
        else:  # DANGER
            color = (0, 0, 255)

        cv2.putText(frame, f"STATE: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        if distance_to_boundary is not None:
            cv2.putText(frame, f"Distance: {int(distance_to_boundary)} px",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # DANGER overlay
        if state == "DANGER":
            cv2.putText(frame, "DANGER DANGER", (int(w * 0.2), int(h * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # FPS estimation (optional)
        frame_count += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            prev_time = current_time
            frame_count = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frames
        cv2.imshow("Hand Danger POC - Camera", frame)
        cv2.imshow("Skin Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
