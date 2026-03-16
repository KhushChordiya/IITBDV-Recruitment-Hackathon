from ultralytics import YOLO
import cv2

# ──────────────────────────────────────────
# CONFIGURATION — change these if needed
# ──────────────────────────────────────────
MODEL_PATH  = "best.pt"          # path to your YOLO weights
IMAGE_PATH  = "image.png"   # path to your input image
OUTPUT_PATH = "output.png"       # where to save the result

REAL_HEIGHT_CM = 30              # real-world cone height in cm
FOCAL_LENGTH_MM = 1000           # camera focal length in mm

# Unit conversion: make H and f the same unit
# H = 30 cm = 300 mm  →  result (d) will be in mm, then convert to meters
REAL_HEIGHT_MM = REAL_HEIGHT_CM * 10   # 300 mm


def calculate_distance(pixel_height):
    """
    d = (H * f) / h
    H = real height in mm
    f = focal length in mm
    h = bounding box height in pixels
    returns distance in meters
    """
    if pixel_height == 0:
        return None
    distance_mm = (REAL_HEIGHT_MM * FOCAL_LENGTH_MM) / pixel_height
    distance_m  = distance_mm / 1000
    return round(distance_m, 2)


def main():
    # ── 1. Load model ──────────────────────────────────────────────
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # ── 2. Load image ──────────────────────────────────────────────
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"ERROR: Could not load image at '{IMAGE_PATH}'")
        return

    # ── 3. Run inference ───────────────────────────────────────────
    print("Running detection...")
    results = model(IMAGE_PATH)[0]

    # ── 4. Process detections ──────────────────────────────────────
    print("\nDetected Cones:")
    print(f"{'Cone':<6} {'BBox (x1,y1,x2,y2)':<30} {'Pixel Height':<15} {'Distance'}")
    print("-" * 65)

    cone_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence      = float(box.conf[0])
        pixel_height    = y2 - y1
        distance        = calculate_distance(pixel_height)

        cone_count += 1
        label = f"Dist: {distance}m" if distance else "N/A"

        print(f"{cone_count:<6} ({x1},{y1},{x2},{y2})        "
              f"{pixel_height}px          {distance}m  (conf: {confidence:.2f})")

        # ── 5. Draw bounding box ───────────────────────────────────
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── 6. Draw label background ───────────────────────────────
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image,
                      (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 6, y1),
                      (0, 255, 0), -1)

        # ── 7. Draw label text ─────────────────────────────────────
        cv2.putText(image, label,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)

    # ── 8. Save output image ───────────────────────────────────────
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"\nTotal cones detected : {cone_count}")
    print(f"Annotated image saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()