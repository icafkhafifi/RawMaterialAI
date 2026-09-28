import cv2
import os
from ultralytics import YOLO

# Folder dataset
TRAIN_PET = "dataset/train/PET"
TRAIN_GLASS = "dataset/train/Glass"

# Pastikan folder tersedia
os.makedirs(TRAIN_PET, exist_ok=True)
os.makedirs(TRAIN_GLASS, exist_ok=True)

print("Memuat YOLO...")

model = YOLO("yolo26n.pt")

print("YOLO siap.")
print()
print("==============================================")
print("       MATERIAL DATA COLLECTOR")
print("==============================================")
print()
print("Arahkan kamera ke BOTOL.")
print()
print("P = simpan sebagai PET")
print("G = simpan sebagai GLASS")
print("Q = keluar")
print()
print("==============================================")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    input("Tekan Enter untuk keluar...")
    exit()

pet_count = 0
glass_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca kamera.")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    bottle_found = False
    bottle_crop = None

    for box in results[0].boxes:

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        if object_name == "bottle" and confidence >= 0.50:

            bottle_found = True

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            bottle_crop = frame[y1:y2, x1:x2]

            break

    if bottle_found:

        cv2.putText(
            annotated_frame,
            "BOTTLE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            "P = PET | G = GLASS | Q = EXIT",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            annotated_frame,
            "Arahkan kamera ke BOTOL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Raw Material AI - Data Collector",
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Simpan PET
    if key == ord("p"):

        if bottle_found and bottle_crop is not None:

            filename = os.path.join(
                TRAIN_PET,
                f"pet_{pet_count:04d}.jpg"
            )

            cv2.imwrite(filename, bottle_crop)

            pet_count += 1

            print("PET disimpan:", filename)

        else:

            print("Tidak ada bottle terdeteksi.")

    # Simpan GLASS
    elif key == ord("g"):

        if bottle_found and bottle_crop is not None:

            filename = os.path.join(
                TRAIN_GLASS,
                f"glass_{glass_count:04d}.jpg"
            )

            cv2.imwrite(filename, bottle_crop)

            glass_count += 1

            print("GLASS disimpan:", filename)

        else:

            print("Tidak ada bottle terdeteksi.")

    # Keluar
    elif key == ord("q"):

        break

cap.release()
cv2.destroyAllWindows()

print()
print("==============================================")
print("DATA COLLECTION SELESAI")
print("==============================================")
print("PET images   :", pet_count)
print("Glass images :", glass_count)