import cv2
from ultralytics import YOLO

print("Memuat Object Detection AI...")
object_model = YOLO("yolo26n.pt")

print("Memuat Material Classification AI...")
material_model = YOLO("runs/classify/train/weights/best.pt")

print("Semua model siap!")
print("Kamera sedang dibuka...")
print("Tekan Q untuk keluar.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    input("Tekan Enter untuk keluar...")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca kamera.")
        break

    # 1. Deteksi objek
    results = object_model(frame, verbose=False)

    display_frame = frame.copy()
    bottle_found = False

    for box in results[0].boxes:

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        object_name = object_model.names[class_id]

        # Kita hanya tertarik pada BOTOL
        if object_name == "bottle" and confidence >= 0.50:

            bottle_found = True

            # Ambil koordinat bounding box
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            # Pastikan koordinat tidak keluar frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop botol
            bottle_crop = frame[y1:y2, x1:x2]

            if bottle_crop.size > 0:

                # 2. Klasifikasi material botol
                material_results = material_model(
                    bottle_crop,
                    verbose=False
                )

                material_result = material_results[0]

                material_id = material_result.probs.top1
                material_confidence = float(
                    material_result.probs.top1conf
                )

                material = material_result.names[material_id]

                # Gambar bounding box
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Informasi hasil
                label = (
                    f"Bottle | {material} | "
                    f"{material_confidence:.2f}"
                )

                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Kalau tidak menemukan botol
    if not bottle_found:
        cv2.putText(
            display_frame,
            "Arahkan kamera ke BOTOL",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        display_frame,
        "Q = EXIT",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Raw Material AI - Object + Material",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()