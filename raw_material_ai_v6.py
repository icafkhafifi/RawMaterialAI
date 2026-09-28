import cv2
import json
import yfinance as yf
from ultralytics import YOLO


# ============================================================
# RAW MATERIAL AI V6
# Visual Material & Supply Chain Intelligence
# ============================================================

print("==============================================")
print("        RAW MATERIAL AI - V6")
print("   Visual Material & Supply Chain Intelligence")
print("==============================================")
print()

# ------------------------------------------------------------
# LOAD DATABASE
# ------------------------------------------------------------

with open("data/materials.json", "r", encoding="utf-8") as f:
    materials_db = json.load(f)

with open("data/companies.json", "r", encoding="utf-8") as f:
    companies_db = json.load(f)


# ------------------------------------------------------------
# MATERIAL MAPPING
# ------------------------------------------------------------

material_mapping = {
    "PET": "PET",
    "Glass": "Soda-lime glass"
}


# ------------------------------------------------------------
# LOAD AI MODELS
# ------------------------------------------------------------

print("Loading object detection model...")
object_model = YOLO("yolo26n.pt")

print("Loading material classification model...")
material_model = YOLO("models/best.pt")

print("AI models ready!")
print()


# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------

def get_related_companies(material_name):

    db_material = material_mapping.get(
        material_name,
        material_name
    )

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


def get_market_data(ticker):

    try:

        stock = yf.Ticker(ticker)

        history = stock.history(period="5d")

        if history.empty or len(history) < 2:
            return None, None

        current_price = float(history["Close"].iloc[-1])
        previous_price = float(history["Close"].iloc[-2])

        if previous_price == 0:
            change = 0
        else:
            change = (
                (current_price - previous_price)
                / previous_price
            ) * 100

        return current_price, change

    except Exception:
        return None, None


def shorten_text(text, max_length):

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def draw_text(
    image,
    text,
    position,
    size=0.6,
    thickness=1,
    color=(230, 230, 230)
):

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_section_title(image, text, y):

    draw_text(
        image,
        text,
        (25, y),
        size=0.58,
        thickness=2,
        color=(120, 210, 255)
    )

    cv2.line(
        image,
        (25, y + 8),
        (495, y + 8),
        (80, 80, 80),
        1
    )


def create_dashboard(
    object_name,
    detection_confidence,
    material_name,
    material_confidence,
    raw_materials,
    companies
):

    width = 560
    height = 620

    dashboard = (
        20 *
        __import__("numpy").ones(
            (height, width, 3),
            dtype="uint8"
        )
    )

    # Dark professional background
    dashboard[:] = (22, 25, 30)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    cv2.rectangle(
        dashboard,
        (0, 0),
        (width, 78),
        (30, 34, 42),
        -1
    )

    draw_text(
        dashboard,
        "RAW MATERIAL AI",
        (25, 34),
        size=0.9,
        thickness=2,
        color=(245, 245, 245)
    )

    draw_text(
        dashboard,
        "VISUAL MATERIAL & SUPPLY CHAIN INTELLIGENCE",
        (25, 58),
        size=0.38,
        thickness=1,
        color=(150, 160, 175)
    )

    # LIVE indicator
    cv2.circle(
        dashboard,
        (510, 29),
        7,
        (80, 220, 120),
        -1
    )

    draw_text(
        dashboard,
        "LIVE",
        (525, 34),
        size=0.42,
        thickness=1,
        color=(120, 220, 140)
    )

    # --------------------------------------------------------
    # OBJECT DETECTION
    # --------------------------------------------------------

    draw_section_title(
        dashboard,
        "OBJECT DETECTION",
        110
    )

    draw_text(
        dashboard,
        "Object",
        (25, 145),
        size=0.45,
        color=(145, 150, 160)
    )

    draw_text(
        dashboard,
        object_name.capitalize(),
        (170, 145),
        size=0.65,
        thickness=2
    )

    draw_text(
        dashboard,
        "Confidence",
        (25, 175),
        size=0.45,
        color=(145, 150, 160)
    )

    draw_text(
        dashboard,
        f"{detection_confidence:.2f}",
        (170, 175),
        size=0.65,
        thickness=2
    )

    # --------------------------------------------------------
    # MATERIAL INTELLIGENCE
    # --------------------------------------------------------

    draw_section_title(
        dashboard,
        "MATERIAL INTELLIGENCE",
        215
    )

    draw_text(
        dashboard,
        "Material",
        (25, 250),
        size=0.45,
        color=(145, 150, 160)
    )

    draw_text(
        dashboard,
        material_name,
        (170, 250),
        size=0.7,
        thickness=2,
        color=(100, 220, 255)
    )

    draw_text(
        dashboard,
        "Confidence",
        (25, 280),
        size=0.45,
        color=(145, 150, 160)
    )

    draw_text(
        dashboard,
        f"{material_confidence:.2f}",
        (170, 280),
        size=0.65,
        thickness=2
    )

    # --------------------------------------------------------
    # RAW MATERIALS
    # --------------------------------------------------------

    draw_section_title(
        dashboard,
        "UPSTREAM RAW MATERIALS",
        320
    )

    y = 350

    for material in raw_materials[:5]:

        draw_text(
            dashboard,
            "• " + material,
            (35, y),
            size=0.48,
            thickness=1
        )

        y += 25

    # --------------------------------------------------------
    # RELATED COMPANIES
    # --------------------------------------------------------

    draw_section_title(
        dashboard,
        "RELATED COMPANIES",
        490
    )

    y = 520

    if not companies:

        draw_text(
            dashboard,
            "No company data available",
            (35, y),
            size=0.45,
            color=(150, 150, 150)
        )

    else:

        for company in companies[:2]:

            company_name = shorten_text(
                company["name"],
                34
            )

            draw_text(
                dashboard,
                company_name,
                (35, y),
                size=0.47,
                thickness=2
            )

            draw_text(
                dashboard,
                (
                    company["ticker"]
                    + "  |  "
                    + company["country"]
                    + "  |  "
                    + company["status"]
                ),
                (35, y + 21),
                size=0.36,
                color=(155, 165, 175)
            )

            price, change = get_market_data(
                company["ticker"]
            )

            if price is not None:

                draw_text(
                    dashboard,
                    f"Market: {price:.2f}  ({change:+.2f}%)",
                    (35, y + 41),
                    size=0.38,
                    color=(110, 210, 140)
                    if change >= 0
                    else (220, 120, 120)
                )

            y += 70

    return dashboard


# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

print("Starting camera...")
print("Press Q to exit.")
print()

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    input("Press Enter to exit...")
    exit()


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

last_material = None
last_object = None

cached_companies = []
cached_raw_materials = []

material_confidence = 0.0
object_confidence = 0.0

dashboard = create_dashboard(
    "Waiting...",
    0.0,
    "Waiting...",
    0.0,
    [],
    []
)


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Cannot read camera.")
        break


    # --------------------------------------------------------
    # OBJECT DETECTION
    # --------------------------------------------------------

    results = object_model(
        frame,
        verbose=False,
        conf=0.50
    )

    annotated_frame = results[0].plot()


    detected_bottle = False


    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = object_model.names[cls_id]

        if class_name.lower() != "bottle":
            continue


        detected_bottle = True

        object_confidence = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # ----------------------------------------------------
        # CROP BOTTLE
        # ----------------------------------------------------

        crop = frame[
            max(0, y1):min(frame.shape[0], y2),
            max(0, x1):min(frame.shape[1], x2)
        ]


        if crop.size == 0:
            continue


        # ----------------------------------------------------
        # MATERIAL CLASSIFICATION
        # ----------------------------------------------------

        material_results = material_model(
            crop,
            verbose=False
        )

        probs = material_results[0].probs

        material_id = int(
            probs.top1
        )

        material_name = material_model.names[
            material_id
        ]

        material_confidence = float(
            probs.top1conf
        )


        # ----------------------------------------------------
        # UPDATE INTELLIGENCE
        # ----------------------------------------------------

        if (
            material_name != last_material
            or class_name != last_object
        ):

            last_material = material_name
            last_object = class_name

            db_material = material_mapping.get(
                material_name,
                material_name
            )

            material_info = materials_db.get(
                db_material,
                {}
            )

            cached_raw_materials = (
                material_info.get(
                    "raw_materials",
                    []
                )
            )

            cached_companies = (
                get_related_companies(
                    material_name
                )
            )

            print()
            print("==============================================")
            print("RAW MATERIAL AI")
            print("==============================================")

            print(
                f"Object              : {class_name}"
            )

            print(
                f"Object confidence   : "
                f"{object_confidence:.2f}"
            )

            print(
                f"Material            : "
                f"{material_name}"
            )

            print(
                f"Material confidence : "
                f"{material_confidence:.2f}"
            )

            print()

            print("RAW MATERIALS:")

            for material in cached_raw_materials:
                print(f" - {material}")

            print()

            print("RELATED COMPANIES:")

            for company in cached_companies:

                print(
                    f" - {company['name']}"
                )

                print(
                    f"   Ticker   : "
                    f"{company['ticker']}"
                )

                print(
                    f"   Country  : "
                    f"{company['country']}"
                )

                print(
                    f"   Status   : "
                    f"{company['status']}"
                )

                price, change = get_market_data(
                    company["ticker"]
                )

                if price is not None:

                    print(
                        f"   Market   : "
                        f"{price:.2f} "
                        f"({change:+.2f}%)"
                    )

            print(
                "=============================================="
            )


        # ----------------------------------------------------
        # DRAW DETECTION LABEL
        # ----------------------------------------------------

        label = (
            f"{material_name} "
            f"{material_confidence:.2f}"
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (80, 210, 255),
            2
        )

        cv2.rectangle(
            annotated_frame,
            (x1, max(0, y1 - 32)),
            (
                x1 + 190,
                y1
            ),
            (25, 30, 35),
            -1
        )

        draw_text(
            annotated_frame,
            label,
            (x1 + 8, max(22, y1 - 10)),
            size=0.52,
            thickness=2,
            color=(100, 220, 255)
        )

        break


    # --------------------------------------------------------
    # DASHBOARD UPDATE
    # --------------------------------------------------------

    if detected_bottle:

        dashboard = create_dashboard(
            class_name,
            object_confidence,
            material_name,
            material_confidence,
            cached_raw_materials,
            cached_companies
        )


    # --------------------------------------------------------
    # COMBINE CAMERA + DASHBOARD
    # --------------------------------------------------------

    camera_view = cv2.resize(
        annotated_frame,
        (760, 620)
    )

    combined = cv2.hconcat(
        [camera_view, dashboard]
    )


    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "Raw Material AI - V6",
        combined
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print()
print("Raw Material AI V6 closed.")