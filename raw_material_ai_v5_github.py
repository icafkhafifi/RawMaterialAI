import cv2
import json
import yfinance as yf
from ultralytics import YOLO


# =========================================================
# RAW MATERIAL AI V5
# Multiple Related Companies
# =========================================================

print("Memuat Object Detection AI...")
detector = YOLO("yolo26n.pt")

print("Memuat Material Classification AI...")
classifier = YOLO("runs/classify/train/weights/best.pt")

print("Memuat database...")
with open("data/materials.json", "r", encoding="utf-8") as f:
    materials_db = json.load(f)

with open("data/companies.json", "r", encoding="utf-8") as f:
    companies_db = json.load(f)

print("Semua AI siap!")


# =========================================================
# MATERIAL MAPPING
# =========================================================

material_mapping = {
    "PET": "PET",
    "Glass": "Soda-lime glass"
}


# =========================================================
# FIND ALL RELATED COMPANIES
# =========================================================

def get_related_companies(material):

    db_material = material_mapping.get(material, material)

    companies = []

    for ticker, company in companies_db.items():

        if db_material in company.get("materials", []):

            companies.append({
                "ticker": ticker,
                "name": company.get("name", "Unknown"),
                "country": company.get("country", "Unknown"),
                "status": company.get("status", "Unknown"),
                "exchange": company.get("exchange", "Unknown")
            })

    return companies


# =========================================================
# GET MARKET DATA
# =========================================================

def get_market_data(ticker):

    try:

        stock = yf.Ticker(ticker)

        history = stock.history(period="5d")

        if history.empty:
            return None, None

        close_prices = history["Close"].dropna()

        if len(close_prices) < 2:
            price = float(close_prices.iloc[-1])
            return price, None

        current_price = float(close_prices.iloc[-1])
        previous_price = float(close_prices.iloc[-2])

        change = (
            (current_price - previous_price)
            / previous_price
        ) * 100

        return current_price, change

    except Exception:

        return None, None


# =========================================================
# DRAW DASHBOARD
# =========================================================

def create_dashboard(
    object_name,
    object_conf,
    material,
    material_conf,
    raw_materials,
    companies
):

    dashboard = cv2.UMat(
        570,
        520,
        cv2.CV_8UC3
    ).get()

    dashboard[:] = (25, 25, 25)

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    cv2.putText(
        dashboard,
        "RAW MATERIAL AI",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        dashboard,
        "Supply Chain Intelligence",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )

    # -----------------------------------------------------
    # OBJECT
    # -----------------------------------------------------

    cv2.putText(
        dashboard,
        "OBJECT",
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )

    cv2.putText(
        dashboard,
        object_name.title(),
        (25, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        dashboard,
        f"Detection: {object_conf:.2f}",
        (25, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1
    )

    # -----------------------------------------------------
    # MATERIAL
    # -----------------------------------------------------

    cv2.putText(
        dashboard,
        "MATERIAL",
        (25, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )

    cv2.putText(
        dashboard,
        material,
        (25, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        dashboard,
        f"Confidence: {material_conf:.2f}",
        (25, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1
    )

    # -----------------------------------------------------
    # RAW MATERIALS
    # -----------------------------------------------------

    cv2.putText(
        dashboard,
        "RAW MATERIALS",
        (25, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )

    y = 375

    for raw in raw_materials:

        cv2.putText(
            dashboard,
            "• " + raw,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1
        )

        y += 23

        if y > 500:
            break

    # -----------------------------------------------------
    # COMPANIES
    # -----------------------------------------------------

    company_x = 275

    cv2.putText(
        dashboard,
        "RELATED COMPANIES",
        (company_x, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )

    company_y = 155

    for company in companies:

        name = company["name"]

        # Shorten long company names
        if len(name) > 25:
            name = name[:23] + "..."

        cv2.putText(
            dashboard,
            name,
            (company_x, company_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        company_y += 25

        cv2.putText(
            dashboard,
            f"Ticker: {company['ticker']}",
            (company_x, company_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1
        )

        company_y += 22

        cv2.putText(
            dashboard,
            f"Country: {company['country']}",
            (company_x, company_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1
        )

        company_y += 22

        cv2.putText(
            dashboard,
            f"Status: {company['status']}",
            (company_x, company_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1
        )

        company_y += 25

        price, change = get_market_data(company["ticker"])

        if price is not None:

            cv2.putText(
                dashboard,
                f"Price: {price:.2f}",
                (company_x, company_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (230, 230, 230),
                1
            )

            company_y += 22

            if change is not None:

                cv2.putText(
                    dashboard,
                    f"Change: {change:+.2f}%",
                    (company_x, company_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (230, 230, 230),
                    1
                )

        else:

            cv2.putText(
                dashboard,
                "Market data unavailable",
                (company_x, company_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1
            )

        company_y += 40

        if company_y > 540:
            break

    return dashboard


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Kamera tidak dapat dibuka.")

    input("Tekan Enter untuk keluar...")

    exit()


# =========================================================
# STATE
# =========================================================

last_material = None
last_object = None

dashboard = cv2.UMat(
    570,
    520,
    cv2.CV_8UC3
).get()

dashboard[:] = (25, 25, 25)


print()
print("==========================================")
print("RAW MATERIAL AI V5")
print("==========================================")
print()
print("Tekan Q untuk keluar.")
print()


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize camera
    frame = cv2.resize(
        frame,
        (760, 570)
    )

    results = detector(
        frame,
        verbose=False,
        conf=0.50
    )

    annotated = frame.copy()

    current_object = None
    current_object_conf = 0

    current_material = None
    current_material_conf = 0

    # -----------------------------------------------------
    # DETECTION
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(box.cls[0])

            object_name = detector.names[cls_id]

            confidence = float(box.conf[0])

            if object_name.lower() != "bottle":
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            bottle_crop = frame[
                y1:y2,
                x1:x2
            ]

            if bottle_crop.size == 0:
                continue

            # -------------------------------------------------
            # MATERIAL CLASSIFICATION
            # -------------------------------------------------

            material_results = classifier(
                bottle_crop,
                verbose=False
            )

            material_result = material_results[0]

            top1 = material_result.probs.top1

            material_conf = float(
                material_result.probs.top1conf
            )

            material = material_result.names[top1]

            current_object = object_name
            current_object_conf = confidence

            current_material = material
            current_material_conf = material_conf

            # -------------------------------------------------
            # DRAW BOUNDING BOX
            # -------------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            label = f"{material} {material_conf:.2f}"

            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - 40)),
                (x1 + 290, y1),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 8, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            break


    # =====================================================
    # UPDATE DASHBOARD WHEN RESULT CHANGES
    # =====================================================

    if (
        current_material is not None
        and (
            current_material != last_material
            or current_object != last_object
        )
    ):

        db_material = material_mapping.get(
            current_material,
            current_material
        )

        material_info = materials_db.get(
            db_material,
            {}
        )

        raw_materials = material_info.get(
            "raw_materials",
            []
        )

        companies = get_related_companies(
            current_material
        )

        dashboard = create_dashboard(
            current_object,
            current_object_conf,
            current_material,
            current_material_conf,
            raw_materials,
            companies
        )

        # -------------------------------------------------
        # CMD OUTPUT
        # -------------------------------------------------

        print()
        print("==========================================")
        print("RAW MATERIAL ANALYSIS")
        print("==========================================")

        print(
            f"Object              : {current_object}"
        )

        print(
            f"Object confidence   : {current_object_conf:.2f}"
        )

        print(
            f"Material            : {current_material}"
        )

        print(
            f"Material confidence : {current_material_conf:.2f}"
        )

        print()
        print("RAW MATERIALS:")

        for raw in raw_materials:
            print(f" - {raw}")

        print()
        print("RELATED COMPANIES:")

        for company in companies:

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

            price, change = get_market_data(
                company["ticker"]
            )

            if price is not None:

                print(
                    f"   Price    : {price:.2f}"
                )

                if change is not None:

                    print(
                        f"   Change   : {change:+.2f}%"
                    )

        print("==========================================")

        last_material = current_material
        last_object = current_object


    # =====================================================
    # COMBINE CAMERA + DASHBOARD
    # =====================================================

    combined = cv2.hconcat(
        [
            annotated,
            dashboard
        ]
    )

    cv2.imshow(
        "RAW MATERIAL AI V5",
        combined
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print()
print("==========================================")
print("RAW MATERIAL AI V5 SELESAI")
print("==========================================")