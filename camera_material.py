import cv2
from ultralytics import YOLO

print("Memuat Material AI...")
model = YOLO("runs/classify/train/weights/best.pt")

print("Model siap!")
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

    # Prediksi material
    results = model(frame, verbose=False)
    result = results[0]

    # Ambil hasil terbaik
    class_id = result.probs.top1
    confidence = float(result.probs.top1conf)
    material = result.names[class_id]

    # Tampilkan hasil
    text = f"{material} | Confidence: {confidence:.2f}"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q = EXIT",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Raw Material AI - Material Classification",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()