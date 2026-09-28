import cv2
import json
import yfinance as yf
from ultralytics import YOLO


# ============================================================
# LOAD DATABASE
# ============================================================

print("Memuat database...")

with open("data/materials.json", "r", encoding="utf-8") as f:
    materials_db = json.load(f)

with open("data/companies.json", "r", encoding="utf-8") as f:
    companies_db = json.load(f)

with open("data/object_map.json", "r", encoding="utf-8") as f:
    object_map = json.load(f)

print("Database berhasil dimuat.")


# ============================================================
# LOAD AI MODEL
# ============================================================

print("Memuat AI YOLO...")

model = YOLO("yolo26n.pt")

print("AI siap!")
print("Kamera sedang dibuka...")
print("Tekan Q untuk keluar.")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    input("Tekan Enter untuk keluar...")
    exit()


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_object(object_name):

    if object_name not in object_map:
        print("\nObjek terdeteksi:", object_name)
        print("Belum ada database material untuk objek ini.")
        return

    object_data = object_map[object_name]

    display_name = object_data["display_name"]
    material_candidates = object_data["material_candidates"]

    print("\n")
    print("=" * 60)
    print("              OBJECT ANALYSIS")
    print("=" * 60)

    print("Object       :", display_name)

    print("\nPOSSIBLE MATERIALS:")

    for material in material_candidates:
        print(" -", material)

    print("\nRAW MATERIALS:")

    raw_materials_found = []

    for material in material_candidates:

        if material in materials_db:

            raw_materials = materials_db[material]["raw_materials"]

            for raw_material in raw_materials:

                if raw_material not in raw_materials_found:
                    raw_materials_found.append(raw_material)

    for raw_material in raw_materials_found:
        print(" -", raw_material)


    # ========================================================
    # FIND RELATED COMPANIES
    # ========================================================

    print("\nRELATED COMPANIES:")

    related_companies = []

    for ticker, company in companies_db.items():

        company_materials = company.get("materials", [])

        for material in material_candidates:

            if material in company_materials:

                if ticker not in related_companies:
                    related_companies.append(ticker)


    if not related_companies:

        print(" - Tidak ada perusahaan dalam database.")

    else:

        for ticker in related_companies:

            company = companies_db[ticker]

            print("\n----------------------------------------")

            print("Company      :", company["name"])
            print("Ticker       :", company["ticker"])
            print("Country      :", company["country"])
            print("Status       :", company["status"])
            print("Exchange     :", company["exchange"])


            # ================================================
            # STOCK MARKET DATA
            # ================================================

            try:

                stock = yf.Ticker(ticker)

                history = stock.history(period="5d")

                if not history.empty:

                    latest_price = float(history["Close"].iloc[-1])

                    print("Stock Price  :", round(latest_price, 2))

                    if len(history) >= 2:

                        previous_price = float(history["Close"].iloc[-2])

                        if previous_price != 0:

                            change = (
                                (latest_price - previous_price)
                                / previous_price
                            ) * 100

                            print(
                                "Change       :",
                                round(change, 2),
                                "%"
                            )

                else:

                    print("Stock Price  : Data tidak tersedia.")

            except Exception as e:

                print("Stock Price  : Gagal mengambil data.")


    print("=" * 60)


# ============================================================
# CAMERA LOOP
# ============================================================

last_object = None

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Tidak dapat membaca kamera.")
        break


    # YOLO detection

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()


    # ========================================================
    # CHECK DETECTED OBJECTS
    # ========================================================

    detected_object = None

    for box in results[0].boxes:

        confidence = float(box.conf[0])

        if confidence < 0.60:
            continue

        class_id = int(box.cls[0])

        object_name = model.names[class_id]

        # Hanya analisis objek yang ada di database

        if object_name in object_map:

            detected_object = object_name
            break


    # ========================================================
    # RUN ANALYSIS ONLY WHEN OBJECT CHANGES
    # ========================================================

    if detected_object != last_object:

        if detected_object is not None:

            analyze_object(detected_object)

        last_object = detected_object


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Raw Material AI - Object Detection",
        annotated_frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLOSE
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nProgram selesai.")