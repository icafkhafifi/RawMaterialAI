import cv2
import json
import yfinance as yf
from ultralytics import YOLO

# ==========================================
# LOAD DATABASE
# ==========================================

with open("data/materials.json", "r", encoding="utf-8") as f:
    materials_db = json.load(f)

with open("data/companies.json", "r", encoding="utf-8") as f:
    companies_db = json.load(f)

# ==========================================
# MATERIAL NAME MAPPING
# ==========================================

material_mapping = {
    "PET": "PET",
    "Glass": "Soda-lime glass"
}

# ==========================================
# LOAD AI MODELS
# ==========================================

print("Memuat Object Detection AI...")
object_model = YOLO("yolo26n.pt")

print("Memuat Material Classification AI...")
material_model = YOLO(
    "runs/classify/train/weights/best.pt"
)

print("Semua AI siap!")
print()
print("==========================================")
print("          RAW MATERIAL AI V3")
print("==========================================")
print("Kamera aktif.")
print("Tekan Q untuk keluar.")
print()

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    input("Tekan Enter untuk keluar...")
    exit()

# Menyimpan hasil terakhir
last_material = None
last_object = None

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca kamera.")
        break

    display_frame = frame.copy()

    # ======================================
    # OBJECT DETECTION
    # ======================================

    results = object_model(
        frame,
        verbose=False
    )

    bottle_found = False

    for box in results[0].boxes:

        detection_conf = float(box.conf[0])
        class_id = int(box.cls[0])
        object_name = object_model.names[class_id]

        if object_name == "bottle" and detection_conf >= 0.50:

            bottle_found = True

            # Bounding box
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

            # Crop bottle
            bottle_crop = frame[y1:y2, x1:x2]

            if bottle_crop.size == 0:
                continue

            # ==================================
            # MATERIAL CLASSIFICATION
            # ==================================

            material_results = material_model(
                bottle_crop,
                verbose=False
            )

            material_result = material_results[0]

            material_id = material_result.probs.top1

            material_conf = float(
                material_result.probs.top1conf
            )

            detected_material = material_result.names[
                material_id
            ]

            # ==================================
            # DATABASE MATERIAL
            # ==================================

            db_material = material_mapping.get(
                detected_material,
                detected_material
            )

            # ==================================
            # RAW MATERIALS
            # ==================================

            if db_material in materials_db:

                raw_materials = materials_db[
                    db_material
                ]["raw_materials"]

            else:

                raw_materials = []

            # ==================================
            # DRAW BOX
            # ==================================

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ==================================
            # DISPLAY RESULT ON CAMERA
            # ==================================

            cv2.putText(
                display_frame,
                f"Bottle: {detected_material}",
                (x1, max(30, y1 - 45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Confidence: {material_conf:.2f}",
                (x1, max(30, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ==================================
            # PRINT ONLY WHEN RESULT CHANGES
            # ==================================

            if (
                detected_material != last_material
                or object_name != last_object
            ):

                print()
                print("==========================================")
                print("             RAW MATERIAL AI")
                print("==========================================")

                print(
                    f"Object              : {object_name}"
                )

                print(
                    f"Object confidence   : {detection_conf:.2f}"
                )

                print(
                    f"Material            : {detected_material}"
                )

                print(
                    f"Database material   : {db_material}"
                )

                print(
                    f"Material confidence : {material_conf:.2f}"
                )

                print()
                print("RAW MATERIALS:")

                if raw_materials:

                    for raw in raw_materials:
                        print(f" - {raw}")

                else:

                    print(" - Belum tersedia di database")

                # ==================================
                # RELATED COMPANIES
                # ==================================

                print()
                print("RELATED COMPANIES:")
                print("------------------------------------------")

                company_found = False

                for ticker, company in companies_db.items():

                    if db_material in company["materials"]:

                        company_found = True

                        print(
                            f" - {company['name']}"
                        )

                        print(
                            f"   Ticker   : {company['ticker']}"
                        )

                        print(
                            f"   Country  : {company['country']}"
                        )

                        print(
                            f"   Status   : {company['status']}"
                        )

                        print(
                            f"   Exchange : {company['exchange']}"
                        )

                        # Stock data
                        try:

                            stock = yf.Ticker(ticker)

                            history = stock.history(
                                period="2d"
                            )

                            if len(history) >= 2:

                                previous = float(
                                    history["Close"].iloc[-2]
                                )

                                current = float(
                                    history["Close"].iloc[-1]
                                )

                                change = (
                                    (current - previous)
                                    / previous
                                    * 100
                                )

                                print(
                                    f"   Price    : {current:.2f}"
                                )

                                print(
                                    f"   Change   : {change:.2f}%"
                                )

                            elif len(history) == 1:

                                current = float(
                                    history["Close"].iloc[-1]
                                )

                                print(
                                    f"   Price    : {current:.2f}"
                                )

                        except Exception:

                            print(
                                "   Stock data unavailable"
                            )

                        print()

                if not company_found:

                    print(
                        " - Belum ada perusahaan yang cocok"
                    )

                print("==========================================")

                # Simpan hasil terakhir
                last_material = detected_material
                last_object = object_name

            break

    # ======================================
    # NO BOTTLE
    # ======================================

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

        # Reset supaya ketika botol muncul lagi,
        # hasil akan dicetak kembali.
        last_material = None
        last_object = None

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
        "Raw Material AI V3",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Raw Material AI V3 selesai.")
print("==========================================")