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
# MATERIAL MAPPING
# ==========================================

material_mapping = {
    "PET": "PET",
    "Glass": "Soda-lime glass"
}

# ==========================================
# LOAD MODELS
# ==========================================

print("Memuat Object Detection AI...")
object_model = YOLO("yolo26n.pt")

print("Memuat Material Classification AI...")
material_model = YOLO(
    "runs/classify/train/weights/best.pt"
)

print("Semua AI siap!")

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    input("Tekan Enter untuk keluar...")
    exit()

# ==========================================
# CURRENT RESULT
# ==========================================

current_object = "Waiting..."
current_object_conf = 0.0

current_material = "Waiting..."
current_material_conf = 0.0

current_db_material = ""

current_raw_materials = []

current_company = None

current_stock_price = None
current_stock_change = None

# Supaya data saham tidak diambil terus-menerus
last_material = None

# ==========================================
# FUNCTION: GET COMPANY DATA
# ==========================================

def get_company_data(material):

    for ticker, company in companies_db.items():

        if material in company["materials"]:

            stock_price = None
            stock_change = None

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

                    stock_price = current
                    stock_change = change

                elif len(history) == 1:

                    stock_price = float(
                        history["Close"].iloc[-1]
                    )

            except Exception:

                pass

            return (
                company,
                stock_price,
                stock_change
            )

    return None, None, None


# ==========================================
# FUNCTION: DRAW TEXT
# ==========================================

def draw_text(
    image,
    text,
    x,
    y,
    size=0.6,
    thickness=1
):

    cv2.putText(
        image,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (230, 230, 230),
        thickness,
        cv2.LINE_AA
    )


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca kamera.")
        break

    # ======================================
    # RESIZE CAMERA
    # ======================================

    frame = cv2.resize(
        frame,
        (760, 570)
    )

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

        detection_conf = float(
            box.conf[0]
        )

        class_id = int(
            box.cls[0]
        )

        object_name = object_model.names[
            class_id
        ]

        # Hanya proses bottle
        if (
            object_name == "bottle"
            and detection_conf >= 0.50
        ):

            bottle_found = True

            current_object = "Bottle"
            current_object_conf = detection_conf

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

            # ==================================
            # CROP BOTTLE
            # ==================================

            bottle_crop = frame[
                y1:y2,
                x1:x2
            ]

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

            material_id = (
                material_result.probs.top1
            )

            material_conf = float(
                material_result.probs.top1conf
            )

            detected_material = (
                material_result.names[
                    material_id
                ]
            )

            db_material = material_mapping.get(
                detected_material,
                detected_material
            )

            current_material = (
                detected_material
            )

            current_material_conf = (
                material_conf
            )

            current_db_material = (
                db_material
            )

            # ==================================
            # RAW MATERIALS
            # ==================================

            if db_material in materials_db:

                current_raw_materials = (
                    materials_db[
                        db_material
                    ]["raw_materials"]
                )

            else:

                current_raw_materials = []

            # ==================================
            # UPDATE COMPANY ONLY WHEN MATERIAL
            # CHANGES
            # ==================================

            if detected_material != last_material:

                (
                    current_company,
                    current_stock_price,
                    current_stock_change
                ) = get_company_data(
                    db_material
                )

                last_material = (
                    detected_material
                )

            # ==================================
            # DRAW BOUNDING BOX
            # ==================================

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ==================================
            # BOTTLE LABEL
            # ==================================

            label = (
                f"{detected_material} "
                f"{material_conf:.2f}"
            )

            cv2.rectangle(
                display_frame,
                (x1, max(0, y1 - 40)),
                (x1 + 220, y1),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display_frame,
                label,
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            break

    # ======================================
    # RESET WHEN NO BOTTLE
    # ======================================

    if not bottle_found:

        current_object = "Waiting..."
        current_object_conf = 0.0

        current_material = "Waiting..."
        current_material_conf = 0.0

        current_db_material = ""

        current_raw_materials = []

        current_company = None

        current_stock_price = None
        current_stock_change = None

        last_material = None

        cv2.putText(
            display_frame,
            "Arahkan kamera ke BOTOL",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # ======================================
    # CREATE DASHBOARD PANEL
    # ======================================

    panel = (
        760,
        570,
        420
    )

    dashboard = cv2.imread(
        "data/dashboard_background.jpg"
    )

    # Jika background belum ada,
    # buat panel kosong
    if dashboard is None:

        dashboard = (
            25
            * 0
        )

        dashboard = cv2.UMat(
            570,
            420,
            cv2.CV_8UC3
        ).get()

        dashboard[:] = (
            25,
            25,
            25
        )

    else:

        dashboard = cv2.resize(
            dashboard,
            (420, 570)
        )

    # ======================================
    # DASHBOARD TITLE
    # ======================================

    draw_text(
        dashboard,
        "RAW MATERIAL AI",
        25,
        40,
        0.9,
        2
    )

    draw_text(
        dashboard,
        "Material Intelligence",
        25,
        70,
        0.6,
        1
    )

    # ======================================
    # OBJECT
    # ======================================

    draw_text(
        dashboard,
        "OBJECT",
        25,
        115,
        0.5,
        1
    )

    draw_text(
        dashboard,
        current_object,
        25,
        145,
        0.8,
        2
    )

    if current_object_conf > 0:

        draw_text(
            dashboard,
            f"Detection: {current_object_conf:.2f}",
            25,
            172,
            0.55,
            1
        )

    # ======================================
    # MATERIAL
    # ======================================

    draw_text(
        dashboard,
        "MATERIAL",
        25,
        215,
        0.5,
        1
    )

    draw_text(
        dashboard,
        current_material,
        25,
        245,
        0.8,
        2
    )

    if current_material_conf > 0:

        draw_text(
            dashboard,
            f"Confidence: {current_material_conf:.2f}",
            25,
            272,
            0.55,
            1
        )

    # ======================================
    # RAW MATERIALS
    # ======================================

    draw_text(
        dashboard,
        "RAW MATERIALS",
        25,
        315,
        0.5,
        1
    )

    y = 345

    if current_raw_materials:

        for raw in current_raw_materials:

            draw_text(
                dashboard,
                "• " + raw,
                30,
                y,
                0.55,
                1
            )

            y += 24

            if y > 445:
                break

    else:

        draw_text(
            dashboard,
            "Waiting for object...",
            30,
            y,
            0.5,
            1
        )

    # ======================================
    # COMPANY
    # ======================================

    draw_text(
        dashboard,
        "RELATED COMPANY",
        220,
        115,
        0.5,
        1
    )

    if current_company:

        company_name = (
            current_company["name"]
        )

        # Pendekkan nama supaya
        # tidak keluar panel
        if len(company_name) > 23:
            company_name = (
                company_name[:23] + "..."
            )

        draw_text(
            dashboard,
            company_name,
            220,
            145,
            0.48,
            1
        )

        draw_text(
            dashboard,
            f"Ticker: {current_company['ticker']}",
            220,
            175,
            0.5,
            1
        )

        draw_text(
            dashboard,
            f"Country: {current_company['country']}",
            220,
            200,
            0.5,
            1
        )

        draw_text(
            dashboard,
            f"Status: {current_company['status']}",
            220,
            225,
            0.5,
            1
        )

        # ==================================
        # MARKET DATA
        # ==================================

        draw_text(
            dashboard,
            "MARKET",
            220,
            275,
            0.5,
            1
        )

        if current_stock_price is not None:

            draw_text(
                dashboard,
                f"Price: {current_stock_price:.2f}",
                220,
                305,
                0.65,
                1
            )

        else:

            draw_text(
                dashboard,
                "Price: unavailable",
                220,
                305,
                0.5,
                1
            )

        if current_stock_change is not None:

            draw_text(
                dashboard,
                f"Change: {current_stock_change:+.2f}%",
                220,
                335,
                0.6,
                1
            )

        else:

            draw_text(
                dashboard,
                "Change: unavailable",
                220,
                335,
                0.5,
                1
            )

    else:

        draw_text(
            dashboard,
            "No company data",
            220,
            145,
            0.5,
            1
        )

    # ======================================
    # COMBINE CAMERA + DASHBOARD
    # ======================================

    final_window = cv2.hconcat(
        [
            display_frame,
            dashboard
        ]
    )

    # ======================================
    # FOOTER
    # ======================================

    cv2.putText(
        final_window,
        "Q = EXIT",
        (20, 555),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imshow(
        "RAW MATERIAL AI V4",
        final_window
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
print("RAW MATERIAL AI V4 SELESAI")
print("==========================================")