import base64
import html
import io
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from openai_car_price import get_price_estimate
import torch
from PIL import Image
from torchvision import models, transforms


DATA_PATH = Path("/workspace/dataset/all_cars/test")
MODEL_PATH = Path("/workspace/models/car_classifier_best.pth")

PORT = 8000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Loading model on: {device}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Test folder not found: {DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

classes = checkpoint["classes"]

print(f"Model contains {len(classes)} classes.")

if len(classes) != 196:
    print(
        "Warning: this model does not contain all "
        "196 Stanford Cars classes."
    )

model = models.mobilenet_v3_small(weights=None)

input_features = model.classifier[3].in_features

model.classifier[3] = torch.nn.Linear(
    input_features,
    len(classes),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image_files = [
    file
    for file in DATA_PATH.rglob("*")
    if file.is_file()
    and file.suffix.lower() in IMAGE_EXTENSIONS
]

if not image_files:
    raise RuntimeError(
        f"No images found inside: {DATA_PATH}"
    )

print(f"Found {len(image_files)} test images.")


def make_prediction():
    # Randomly choose one image from the test dataset.
    image_path = random.choice(image_files)

    actual_class = image_path.parent.name

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = image_transform(
        original_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        min(5, len(classes)),
    )

    predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices,
    ):
        predictions.append({
            "class": classes[index.item()],
            "confidence": probability.item() * 100,
        })

    predicted_class = predictions[0]["class"]
    predicted_confidence = predictions[0]["confidence"]
    correct = predicted_class == actual_class

    price_info = get_price_estimate(predicted_class)

    display_image = original_image.copy()
    display_image.thumbnail((1000, 650))

    image_buffer = io.BytesIO()

    display_image.save(
        image_buffer,
        format="JPEG",
        quality=90,
    )

    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    return {
        "image": encoded_image,
        "filename": image_path.name,
        "actual": actual_class,
        "predicted": predicted_class,
        "confidence": predicted_confidence,
        "correct": correct,
        "price": price_info,
        "predictions": predictions,
    }



def build_openai_price_html(price):
    def money(value):
        if value is None:
            return "Not found"

        return f"${value:,.0f}"

    if not price.get("available"):
        message = html.escape(
            str(
                price.get(
                    "notice",
                    "Pricing information unavailable.",
                )
            )
        )

        return f"""
        <div class="result-box">
            <h2>Estimated Vehicle Pricing</h2>
            <p>{message}</p>
        </div>
        """

    trend = html.escape(
        str(price.get("trend", "unknown"))
    )

    change = price.get("change_percent")
    annual = price.get(
        "annual_change_percent"
    )
    change_amount = price.get(
        "change_amount"
    )

    if change is None:
        total_change = "Unknown"

    else:
        total_change = (
            f"{abs(change):.1f}% {trend}"
        )

    if change_amount is None:
        dollar_change = "Unknown"

    elif change_amount < 0:
        dollar_change = (
            f"Decreased by "
            f"${abs(change_amount):,.0f}"
        )

    elif change_amount > 0:
        dollar_change = (
            f"Increased by "
            f"${change_amount:,.0f}"
        )

    else:
        dollar_change = "No estimated change"

    if annual is None:
        annual_change = "Unknown"

    elif annual < 0:
        annual_change = (
            f"{abs(annual):.1f}% average "
            f"decrease per year"
        )

    else:
        annual_change = (
            f"{annual:.1f}% average "
            f"increase per year"
        )

    source_items = []

    for source in price.get("sources", []):
        title = html.escape(
            str(
                source.get(
                    "title",
                    "Pricing source",
                )
            )
        )

        url = html.escape(
            str(source.get("url", "")),
            quote=True,
        )

        if url:
            source_items.append(
                f'<li><a href="{url}" '
                f'target="_blank" '
                f'rel="noopener noreferrer">'
                f'{title}</a></li>'
            )

    if source_items:
        sources_html = (
            "<p><strong>Sources checked:"
            "</strong></p>"
            "<ul>"
            + "".join(source_items)
            + "</ul>"
        )

    else:
        sources_html = (
            "<p><strong>Sources:</strong> "
            "No source links returned.</p>"
        )

    summary = html.escape(
        str(price.get("summary", ""))
    )

    notice = html.escape(
        str(price.get("notice", ""))
    )

    method = html.escape(
        str(price.get("method", ""))
    )

    checked_date = html.escape(
        str(price.get("checked_date", ""))
    )

    car = html.escape(
        str(price.get("car", ""))
    )

    return f"""
    <div class="result-box">
        <h2>Estimated Vehicle Pricing</h2>

        <p>
            <strong>Vehicle priced:</strong>
            {car}
        </p>

        <p>
            <strong>Original base MSRP:</strong>
            {money(price.get("original_msrp"))}
        </p>

        <p>
            <strong>Current used-retail range:</strong>
            {money(price.get("current_low"))}
            –
            {money(price.get("current_high"))}
        </p>

        <p>
            <strong>Estimated current midpoint:</strong>
            {money(price.get("current_midpoint"))}
        </p>

        <p>
            <strong>Price direction since new:</strong>
            {total_change}
        </p>

        <p>
            <strong>Estimated dollar change:</strong>
            {dollar_change}
        </p>

        <p>
            <strong>Average yearly change:</strong>
            {annual_change}
        </p>

        <p>
            <strong>Explanation:</strong>
            {summary}
        </p>

        <p>
            <strong>Research method:</strong>
            {method}
        </p>

        <p>
            <strong>Checked date:</strong>
            {checked_date}
        </p>

        <p><em>{notice}</em></p>

        {sources_html}
    </div>
    """


def make_html(result):
    openai_price_section = (
        build_openai_price_html(
            result.get("price", {})
        )
    )
    rows = ""

    for rank, prediction in enumerate(
        result["predictions"],
        start=1,
    ):
        rows += f"""
        <tr>
            <td>{rank}</td>
            <td>{html.escape(prediction["class"])}</td>
            <td>{prediction["confidence"]:.2f}%</td>
        </tr>
        """

    status = (
        "CORRECT"
        if result["correct"]
        else "INCORRECT"
    )

    status_class = (
        "correct"
        if result["correct"]
        else "incorrect"
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">

    <title>Jetson Car Classifier</title>

    <style>
        body {{
            margin: 0;
            padding: 30px;
            font-family: Arial, sans-serif;
            background: #eeeeee;
        }}

        .container {{
            max-width: 1100px;
            margin: auto;
            padding: 25px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.15);
        }}

        h1 {{
            text-align: center;
        }}

        .car-image {{
            display: block;
            max-width: 100%;
            max-height: 650px;
            margin: 20px auto;
            border-radius: 8px;
        }}

        .result-box {{
            padding: 18px;
            margin-top: 20px;
            background: #f4f4f4;
            border-radius: 8px;
        }}

        .correct {{
            color: green;
            font-weight: bold;
        }}

        .incorrect {{
            color: red;
            font-weight: bold;
        }}

        table {{
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 12px;
            border: 1px solid #cccccc;
            text-align: left;
        }}

        th {{
            color: white;
            background: #333333;
        }}

        button {{
            display: block;
            margin: 25px auto 0;
            padding: 14px 25px;
            font-size: 18px;
            cursor: pointer;
        }}
    
        .loading-overlay {{
            display: none;
            position: fixed;
            inset: 0;
            z-index: 9999;
            align-items: center;
            justify-content: center;
            background: rgba(0, 0, 0, 0.72);
        }}

        .loading-overlay.visible {{
            display: flex;
        }}

        .loading-card {{
            width: min(520px, 85%);
            padding: 28px;
            text-align: center;
            background: white;
            border-radius: 12px;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.35);
        }}

        .loading-track {{
            width: 100%;
            height: 18px;
            margin-top: 18px;
            overflow: hidden;
            background: #dddddd;
            border-radius: 10px;
        }}

        .loading-bar {{
            width: 35%;
            height: 100%;
            background: #333333;
            border-radius: 10px;
            animation: loadingMove 1.1s infinite ease-in-out;
        }}

        @keyframes loadingMove {{
            0% {{
                transform: translateX(-110%);
            }}

            100% {{
                transform: translateX(310%);
            }}
        }}

    
        /* polished-loading-v2 */

        html {{
            min-height: 100%;
            scroll-behavior: smooth;
        }}

        body {{
            min-height: 100vh;
            margin: 0;
            padding: clamp(18px, 4vw, 48px);
            box-sizing: border-box;
            color: #172033;
            background:
                radial-gradient(
                    circle at top left,
                    rgba(94, 129, 244, 0.20),
                    transparent 34%
                ),
                radial-gradient(
                    circle at bottom right,
                    rgba(83, 203, 183, 0.16),
                    transparent 32%
                ),
                linear-gradient(
                    145deg,
                    #eef3fb,
                    #f8fafc
                );
        }}

        *, *::before, *::after {{
            box-sizing: border-box;
        }}

        .container {{
            width: 100%;
            max-width: 1120px;
            margin: 0 auto;
            padding: clamp(20px, 4vw, 42px);
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 24px;
            box-shadow:
                0 24px 60px rgba(15, 23, 42, 0.14),
                0 4px 14px rgba(15, 23, 42, 0.06);
            overflow: hidden;
        }}

        h1 {{
            margin: 0 0 10px;
            font-size: clamp(30px, 5vw, 48px);
            line-height: 1.05;
            letter-spacing: -0.04em;
            color: #111827;
        }}

        h2 {{
            margin-top: 30px;
            color: #1f2937;
            letter-spacing: -0.02em;
        }}

        p {{
            line-height: 1.65;
        }}

        .car-image {{
            display: block;
            width: auto;
            max-width: 100%;
            max-height: 640px;
            margin: 28px auto;
            padding: 8px;
            object-fit: contain;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            box-shadow:
                0 18px 40px rgba(15, 23, 42, 0.16);
        }}

        .result-box {{
            margin-top: 22px;
            padding: clamp(18px, 3vw, 28px);
            background:
                linear-gradient(
                    145deg,
                    #f8fafc,
                    #ffffff
                );
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            box-shadow:
                0 8px 24px rgba(15, 23, 42, 0.06);
        }}

        .result-box p {{
            margin: 10px 0;
        }}

        table {{
            width: 100%;
            margin-top: 18px;
            overflow: hidden;
            border: 1px solid #dbe3ee;
            border-spacing: 0;
            border-collapse: separate;
            border-radius: 16px;
            background: white;
        }}

        th {{
            padding: 14px;
            color: white;
            text-align: left;
            background:
                linear-gradient(
                    135deg,
                    #1e293b,
                    #334155
                );
        }}

        td {{
            padding: 14px;
            border-top: 1px solid #e2e8f0;
            border-left: 0;
            border-right: 0;
            border-bottom: 0;
        }}

        tr:nth-child(even) td {{
            background: #f8fafc;
        }}

        a {{
            color: #3157c8;
            font-weight: 600;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        button {{
            display: block;
            min-width: min(100%, 310px);
            margin: 30px auto 0;
            padding: 15px 26px;
            color: white;
            font-size: 17px;
            font-weight: 700;
            letter-spacing: 0.01em;
            cursor: pointer;
            background:
                linear-gradient(
                    135deg,
                    #3157c8,
                    #5375e8
                );
            border: 0;
            border-radius: 999px;
            box-shadow:
                0 12px 25px rgba(49, 87, 200, 0.30);
            transition:
                transform 160ms ease,
                box-shadow 160ms ease,
                opacity 160ms ease;
        }}

        button:hover:not(:disabled) {{
            transform: translateY(-2px);
            box-shadow:
                0 16px 30px rgba(49, 87, 200, 0.36);
        }}

        button:active:not(:disabled) {{
            transform: translateY(0);
        }}

        button:disabled {{
            opacity: 0.65;
            cursor: wait;
        }}

        html.is-loading,
        body.is-loading {{
            height: 100%;
            overflow: hidden !important;
            overscroll-behavior: none;
            touch-action: none;
        }}

        body.is-loading > *:not(#loadingOverlay) {{
            pointer-events: none !important;
            user-select: none !important;
        }}

        .loading-overlay {{
            display: none;
            position: fixed;
            inset: 0;
            z-index: 999999;
            align-items: center;
            justify-content: center;
            padding: 24px;
            overflow: hidden;
            pointer-events: auto;
            background: rgba(15, 23, 42, 0.78);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            overscroll-behavior: contain;
        }}

        .loading-overlay.visible {{
            display: flex;
        }}

        .loading-card {{
            width: min(540px, 100%);
            padding: clamp(26px, 5vw, 42px);
            text-align: center;
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(255, 255, 255, 0.55);
            border-radius: 24px;
            box-shadow:
                0 30px 80px rgba(0, 0, 0, 0.38);
            animation: loadingCardEnter 220ms ease-out;
        }}

        .loading-card h2 {{
            margin: 0 0 10px;
            font-size: clamp(24px, 5vw, 34px);
        }}

        .loading-card p {{
            margin: 0;
            color: #526075;
        }}

        .loading-track {{
            width: 100%;
            height: 14px;
            margin-top: 24px;
            overflow: hidden;
            background: #dce4f0;
            border-radius: 999px;
            box-shadow:
                inset 0 1px 3px rgba(15, 23, 42, 0.14);
        }}

        .loading-bar {{
            width: 38%;
            height: 100%;
            background:
                linear-gradient(
                    90deg,
                    #3157c8,
                    #5f83f1,
                    #3157c8
                );
            border-radius: inherit;
            animation:
                loadingMove 1.05s infinite ease-in-out;
        }}

        @keyframes loadingMove {{
            0% {{
                transform: translateX(-115%);
            }}

            100% {{
                transform: translateX(300%);
            }}
        }}

        @keyframes loadingCardEnter {{
            from {{
                opacity: 0;
                transform: translateY(10px) scale(0.98);
            }}

            to {{
                opacity: 1;
                transform: translateY(0) scale(1);
            }}
        }}

        @media (max-width: 700px) {{
            body {{
                padding: 12px;
            }}

            .container {{
                padding: 20px 16px;
                border-radius: 18px;
            }}

            th, td {{
                padding: 10px 8px;
                font-size: 14px;
            }}

            button {{
                width: 100%;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                scroll-behavior: auto !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}

    
        /* premium-dark-mode-v3-start */

        :root {{
            color-scheme: dark;

            --page-background: #070a12;
            --panel-background: rgba(17, 24, 39, 0.84);
            --panel-background-solid: #111827;
            --panel-secondary: rgba(30, 41, 59, 0.72);

            --border: rgba(148, 163, 184, 0.18);
            --border-bright: rgba(129, 140, 248, 0.40);

            --text-primary: #f8fafc;
            --text-secondary: #aebbd0;
            --text-muted: #7f8da3;

            --accent-primary: #818cf8;
            --accent-secondary: #38bdf8;
            --accent-tertiary: #2dd4bf;

            --success: #4ade80;
            --danger: #fb7185;
            --warning: #fbbf24;

            --shadow-large:
                0 36px 90px rgba(0, 0, 0, 0.58);

            --shadow-medium:
                0 18px 45px rgba(0, 0, 0, 0.34);
        }}

        html {{
            min-height: 100%;
            color-scheme: dark;
            background: var(--page-background);
            scroll-behavior: smooth;
        }}

        body {{
            min-height: 100vh;
            margin: 0;
            padding: clamp(14px, 3vw, 42px);
            box-sizing: border-box;
            overflow-x: hidden;

            color: var(--text-primary);
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            background:
                radial-gradient(
                    circle at 10% 0%,
                    rgba(99, 102, 241, 0.24),
                    transparent 32%
                ),
                radial-gradient(
                    circle at 95% 12%,
                    rgba(14, 165, 233, 0.18),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 50% 100%,
                    rgba(20, 184, 166, 0.12),
                    transparent 36%
                ),
                linear-gradient(
                    145deg,
                    #05070d 0%,
                    #090d18 50%,
                    #070a12 100%
                );

            background-attachment: fixed;
        }}

        body::before {{
            content: "";
            position: fixed;
            inset: 0;
            z-index: -1;
            opacity: 0.25;
            pointer-events: none;

            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.025) 1px,
                    transparent 1px
                ),
                linear-gradient(
                    90deg,
                    rgba(255, 255, 255, 0.025) 1px,
                    transparent 1px
                );

            background-size: 48px 48px;
            mask-image:
                linear-gradient(
                    to bottom,
                    black,
                    transparent 82%
                );
        }}

        *,
        *::before,
        *::after {{
            box-sizing: border-box;
        }}

        ::selection {{
            color: white;
            background: rgba(99, 102, 241, 0.78);
        }}

        ::-webkit-scrollbar {{
            width: 11px;
            height: 11px;
        }}

        ::-webkit-scrollbar-track {{
            background: #080c15;
        }}

        ::-webkit-scrollbar-thumb {{
            background:
                linear-gradient(
                    180deg,
                    #4f46e5,
                    #0284c7
                );
            border: 3px solid #080c15;
            border-radius: 999px;
        }}

        .container {{
            position: relative;
            width: 100%;
            max-width: 1160px;
            margin: 0 auto;
            padding: clamp(22px, 4vw, 48px);
            overflow: hidden;

            background:
                linear-gradient(
                    145deg,
                    rgba(17, 24, 39, 0.94),
                    rgba(9, 14, 26, 0.92)
                );

            border: 1px solid var(--border);
            border-radius: 28px;

            box-shadow:
                var(--shadow-large),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);

            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
        }}

        .container::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 7%;
            width: 86%;
            height: 1px;

            background:
                linear-gradient(
                    90deg,
                    transparent,
                    rgba(129, 140, 248, 0.95),
                    rgba(56, 189, 248, 0.90),
                    transparent
                );
        }}

        .container::after {{
            content: "";
            position: absolute;
            top: -180px;
            right: -180px;
            width: 380px;
            height: 380px;
            z-index: 0;
            opacity: 0.20;
            pointer-events: none;

            background:
                radial-gradient(
                    circle,
                    var(--accent-primary),
                    transparent 68%
                );
        }}

        .container > * {{
            position: relative;
            z-index: 1;
        }}

        h1 {{
            margin: 0 0 12px;
            text-align: center;

            color: transparent;
            font-size: clamp(34px, 6vw, 58px);
            font-weight: 850;
            line-height: 1.02;
            letter-spacing: -0.055em;

            background:
                linear-gradient(
                    115deg,
                    #ffffff 15%,
                    #c7d2fe 47%,
                    #7dd3fc 75%,
                    #5eead4 100%
                );

            background-clip: text;
            -webkit-background-clip: text;

            text-shadow:
                0 0 40px rgba(99, 102, 241, 0.20);
        }}

        h2 {{
            margin-top: 34px;
            margin-bottom: 16px;

            color: var(--text-primary);
            font-size: clamp(21px, 3vw, 29px);
            font-weight: 760;
            letter-spacing: -0.025em;
        }}

        h2::after {{
            content: "";
            display: block;
            width: 58px;
            height: 3px;
            margin-top: 9px;

            background:
                linear-gradient(
                    90deg,
                    var(--accent-primary),
                    var(--accent-secondary)
                );

            border-radius: 999px;
        }}

        p {{
            color: var(--text-secondary);
            line-height: 1.68;
        }}

        body > .container > p:first-of-type {{
            max-width: 720px;
            margin: 0 auto 28px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 16px;
        }}

        strong {{
            color: #eef2ff;
            font-weight: 720;
        }}

        em {{
            color: var(--text-muted);
        }}

        .car-image {{
            display: block;
            width: auto;
            max-width: 100%;
            max-height: 650px;
            margin: 30px auto;
            padding: 8px;
            object-fit: contain;

            background:
                linear-gradient(
                    145deg,
                    rgba(30, 41, 59, 0.86),
                    rgba(15, 23, 42, 0.95)
                );

            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 22px;

            box-shadow:
                0 26px 55px rgba(0, 0, 0, 0.52),
                0 0 0 1px rgba(255, 255, 255, 0.025),
                0 0 40px rgba(99, 102, 241, 0.09);

            transition:
                transform 240ms ease,
                border-color 240ms ease,
                box-shadow 240ms ease;
        }}

        .car-image:hover {{
            transform: translateY(-3px);
            border-color: var(--border-bright);

            box-shadow:
                0 32px 68px rgba(0, 0, 0, 0.62),
                0 0 48px rgba(99, 102, 241, 0.15);
        }}

        .result-box {{
            position: relative;
            margin-top: 22px;
            padding: clamp(20px, 3vw, 30px);
            overflow: hidden;

            color: var(--text-primary);

            background:
                linear-gradient(
                    145deg,
                    rgba(30, 41, 59, 0.72),
                    rgba(15, 23, 42, 0.86)
                );

            border: 1px solid var(--border);
            border-radius: 20px;

            box-shadow:
                var(--shadow-medium),
                inset 0 1px 0 rgba(255, 255, 255, 0.035);

            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }}

        .result-box::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 22px;
            right: 22px;
            height: 1px;

            background:
                linear-gradient(
                    90deg,
                    transparent,
                    rgba(129, 140, 248, 0.55),
                    transparent
                );
        }}

        .result-box p {{
            display: grid;
            grid-template-columns:
                minmax(180px, 0.38fr)
                minmax(0, 1fr);

            gap: 14px;
            align-items: start;
            margin: 11px 0;
            padding-bottom: 11px;

            color: var(--text-secondary);
            border-bottom:
                1px solid rgba(148, 163, 184, 0.10);
        }}

        .result-box p:last-child {{
            padding-bottom: 0;
            border-bottom: 0;
        }}

        .correct {{
            display: inline-flex;
            align-items: center;
            width: fit-content;
            padding: 5px 11px;

            color: #bbf7d0;
            font-weight: 800;

            background: rgba(34, 197, 94, 0.13);
            border: 1px solid rgba(74, 222, 128, 0.30);
            border-radius: 999px;
        }}

        .incorrect {{
            display: inline-flex;
            align-items: center;
            width: fit-content;
            padding: 5px 11px;

            color: #fecdd3;
            font-weight: 800;

            background: rgba(244, 63, 94, 0.13);
            border: 1px solid rgba(251, 113, 133, 0.32);
            border-radius: 999px;
        }}

        table {{
            width: 100%;
            margin-top: 18px;
            overflow: hidden;

            color: var(--text-secondary);
            background: rgba(15, 23, 42, 0.72);

            border: 1px solid var(--border);
            border-spacing: 0;
            border-collapse: separate;
            border-radius: 18px;

            box-shadow:
                0 16px 40px rgba(0, 0, 0, 0.30);
        }}

        th {{
            padding: 15px 16px;
            color: #f8fafc;
            text-align: left;
            font-size: 13px;
            font-weight: 780;
            letter-spacing: 0.065em;
            text-transform: uppercase;

            background:
                linear-gradient(
                    135deg,
                    rgba(67, 56, 202, 0.86),
                    rgba(3, 105, 161, 0.84)
                );

            border: 0;
        }}

        td {{
            padding: 15px 16px;
            color: #cbd5e1;

            background: rgba(15, 23, 42, 0.64);
            border-top:
                1px solid rgba(148, 163, 184, 0.11);
            border-left: 0;
            border-right: 0;
            border-bottom: 0;
        }}

        tbody tr {{
            transition:
                background 150ms ease,
                transform 150ms ease;
        }}

        tbody tr:hover td {{
            color: #f1f5f9;
            background: rgba(51, 65, 85, 0.72);
        }}

        tbody tr:first-child td {{
            color: #eef2ff;
            font-weight: 680;
            background: rgba(79, 70, 229, 0.13);
        }}

        a {{
            color: #7dd3fc;
            font-weight: 680;
            text-decoration: none;
            text-underline-offset: 4px;
        }}

        a:hover {{
            color: #bae6fd;
            text-decoration: underline;
        }}

        button {{
            position: relative;
            display: block;
            min-width: min(100%, 330px);
            margin: 34px auto 0;
            padding: 16px 30px;
            overflow: hidden;

            color: white;
            font-size: 17px;
            font-weight: 800;
            letter-spacing: 0.012em;

            cursor: pointer;

            background:
                linear-gradient(
                    120deg,
                    #4f46e5,
                    #2563eb 50%,
                    #0891b2
                );

            background-size: 180% 100%;
            border: 1px solid rgba(165, 180, 252, 0.38);
            border-radius: 999px;

            box-shadow:
                0 15px 34px rgba(37, 99, 235, 0.34),
                inset 0 1px 0 rgba(255, 255, 255, 0.20);

            transition:
                transform 180ms ease,
                box-shadow 180ms ease,
                background-position 260ms ease,
                opacity 180ms ease;
        }}

        button::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -35%;
            width: 28%;
            height: 200%;

            opacity: 0;
            transform: rotate(24deg);

            background:
                linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.45),
                    transparent
                );

            transition:
                left 420ms ease,
                opacity 160ms ease;
        }}

        button:hover:not(:disabled) {{
            transform: translateY(-3px);
            background-position: 100% 0;

            box-shadow:
                0 21px 42px rgba(37, 99, 235, 0.46),
                0 0 28px rgba(56, 189, 248, 0.12);
        }}

        button:hover:not(:disabled)::before {{
            left: 112%;
            opacity: 1;
        }}

        button:active:not(:disabled) {{
            transform: translateY(-1px) scale(0.99);
        }}

        button:focus-visible {{
            outline: 3px solid rgba(125, 211, 252, 0.62);
            outline-offset: 4px;
        }}

        button:disabled {{
            opacity: 0.58;
            cursor: wait;
            filter: saturate(0.70);
        }}

        html.is-loading,
        body.is-loading {{
            height: 100%;
            overflow: hidden !important;
            overscroll-behavior: none;
            touch-action: none;
        }}

        body.is-loading > *:not(#loadingOverlay) {{
            pointer-events: none !important;
            user-select: none !important;
        }}

        .loading-overlay {{
            display: none;
            position: fixed;
            inset: 0;
            z-index: 9999999;
            align-items: center;
            justify-content: center;
            padding: 24px;
            overflow: hidden;

            pointer-events: auto;
            overscroll-behavior: contain;
            touch-action: none;

            background:
                radial-gradient(
                    circle at center,
                    rgba(30, 41, 59, 0.78),
                    rgba(2, 6, 23, 0.96)
                );

            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }}

        .loading-overlay.visible {{
            display: flex;
        }}

        .loading-card {{
            position: relative;
            width: min(560px, 100%);
            padding: clamp(28px, 5vw, 46px);
            overflow: hidden;

            text-align: center;

            background:
                linear-gradient(
                    145deg,
                    rgba(30, 41, 59, 0.97),
                    rgba(15, 23, 42, 0.98)
                );

            border: 1px solid rgba(129, 140, 248, 0.35);
            border-radius: 26px;

            box-shadow:
                0 36px 100px rgba(0, 0, 0, 0.72),
                0 0 55px rgba(79, 70, 229, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.07);

            animation:
                darkLoadingCardEnter 240ms ease-out;
        }}

        .loading-card::before {{
            content: "";
            position: absolute;
            top: -90px;
            left: 50%;
            width: 260px;
            height: 180px;
            transform: translateX(-50%);

            opacity: 0.22;

            background:
                radial-gradient(
                    circle,
                    var(--accent-primary),
                    transparent 70%
                );
        }}

        .loading-card h2 {{
            position: relative;
            margin: 0 0 12px;

            color: #f8fafc;
            font-size: clamp(25px, 5vw, 36px);
            font-weight: 820;
        }}

        .loading-card h2::after {{
            margin-left: auto;
            margin-right: auto;
        }}

        .loading-card p {{
            position: relative;
            margin: 0;
            color: var(--text-secondary);
        }}

        .loading-track {{
            position: relative;
            width: 100%;
            height: 14px;
            margin-top: 26px;
            overflow: hidden;

            background: rgba(2, 6, 23, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;

            box-shadow:
                inset 0 2px 6px rgba(0, 0, 0, 0.52);
        }}

        .loading-bar {{
            width: 38%;
            height: 100%;

            background:
                linear-gradient(
                    90deg,
                    #6366f1,
                    #38bdf8,
                    #2dd4bf,
                    #6366f1
                );

            background-size: 220% 100%;
            border-radius: inherit;

            box-shadow:
                0 0 18px rgba(56, 189, 248, 0.65);

            animation:
                darkLoadingMove 1.05s infinite ease-in-out,
                darkLoadingGlow 1.6s infinite linear;
        }}

        @keyframes darkLoadingMove {{
            0% {{
                transform: translateX(-120%);
            }}

            100% {{
                transform: translateX(300%);
            }}
        }}

        @keyframes darkLoadingGlow {{
            0% {{
                background-position: 0% 50%;
            }}

            100% {{
                background-position: 220% 50%;
            }}
        }}

        @keyframes darkLoadingCardEnter {{
            from {{
                opacity: 0;
                transform:
                    translateY(14px)
                    scale(0.975);
            }}

            to {{
                opacity: 1;
                transform:
                    translateY(0)
                    scale(1);
            }}
        }}

        @media (max-width: 720px) {{
            body {{
                padding: 10px;
            }}

            .container {{
                padding: 22px 15px;
                border-radius: 20px;
            }}

            .result-box {{
                border-radius: 17px;
            }}

            .result-box p {{
                display: block;
            }}

            .result-box p strong {{
                display: block;
                margin-bottom: 4px;
            }}

            table {{
                display: block;
                overflow-x: auto;
                white-space: nowrap;
            }}

            th,
            td {{
                padding: 12px 10px;
                font-size: 13px;
            }}

            button {{
                width: 100%;
                min-width: 0;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *,
            *::before,
            *::after {{
                scroll-behavior: auto !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}

        /* premium-dark-mode-v3-end */

    
        /* top-five-dark-table-v1 */

        h2 + table {{
            color: #dbeafe !important;
            background:
                linear-gradient(
                    145deg,
                    #0b1220,
                    #101827
                ) !important;

            border:
                1px solid rgba(
                    129,
                    140,
                    248,
                    0.30
                ) !important;

            box-shadow:
                0 18px 45px rgba(
                    0,
                    0,
                    0,
                    0.42
                ) !important;
        }}

        h2 + table th {{
            color: #ffffff !important;

            background:
                linear-gradient(
                    135deg,
                    #312e81,
                    #1d4ed8,
                    #0369a1
                ) !important;

            border-color:
                rgba(
                    255,
                    255,
                    255,
                    0.08
                ) !important;
        }}

        h2 + table td {{
            color: #cbd5e1 !important;
            background: #0f172a !important;

            border-top:
                1px solid rgba(
                    148,
                    163,
                    184,
                    0.14
                ) !important;
        }}

        h2 + table tr:nth-child(even) td {{
            background: #111c30 !important;
        }}

        h2 + table tbody tr:first-child td {{
            color: #eef2ff !important;
            font-weight: 750;

            background:
                linear-gradient(
                    90deg,
                    rgba(
                        79,
                        70,
                        229,
                        0.28
                    ),
                    rgba(
                        30,
                        64,
                        175,
                        0.20
                    )
                ) !important;
        }}

        h2 + table tbody tr:hover td {{
            color: #ffffff !important;
            background: #1e293b !important;
        }}

        h2 + table td:first-child {{
            color: #93c5fd !important;
            font-weight: 800;
        }}

        h2 + table td:last-child {{
            color: #5eead4 !important;
            font-weight: 750;
        }}

    </style>
    <meta name="theme-color" content="#070a12">
</head>

<body>
    <div class="container">
        <h1>Car Classifier</h1>

        <p style="text-align:center;">
            Random test image from 196 Stanford Cars classes
        </p>

        <img
            class="car-image"
            src="data:image/jpeg;base64,{result["image"]}"
            alt="Randomly selected car"
        >

        <div class="result-box">
            <p>
                <strong>Image file:</strong>
                {html.escape(result["filename"])}
            </p>

            <p>
                <strong>Actual car:</strong>
                {html.escape(result["actual"])}
            </p>

            <p>
                <strong>Predicted car:</strong>
                {html.escape(result["predicted"])}
            </p>

            <p>
                <strong>Confidence:</strong>
                {result["confidence"]:.2f}%
            </p>

            <p>
                <strong>Result:</strong>
                <span class="{status_class}">
                    {status}
                </span>
            </p>
        </div>

        {openai_price_section}

        <h2>Top Five Predictions</h2>

        <table>
            <tr>
                <th>Rank</th>
                <th>Car Make and Model</th>
                <th>Confidence</th>
            </tr>

            {rows}
        </table>

        
        
        <button
            id="randomButton"
            type="button"
            onclick="showLoadingAndChooseCar()"
        >
            Choose Another Random Car
        </button>



        <script>
            function loadAnotherCar() {{
                const button =
                    document.getElementById("randomButton");

                button.disabled = true;
                button.textContent = "Loading prediction...";

                window.location.href =
                    "/random?time=" + Date.now();
            }}
        </script>
    </div>

    <div
        id="loadingOverlay"
        class="loading-overlay"
        aria-live="polite"
    >
        <div class="loading-card">
            <h2>Loading Another Car</h2>

            <p>
                Running the vehicle classifier and
                researching estimated pricing...
            </p>

            <div class="loading-track">
                <div class="loading-bar"></div>
            </div>
        </div>
    </div>

    <script>
        function showLoadingAndChooseCar() {{
            const overlay =
                document.getElementById(
                    "loadingOverlay"
                );

            const button =
                document.getElementById(
                    "randomButton"
                );

            if (button) {{
                button.disabled = true;
                button.textContent =
                    "Loading prediction...";
            }}

            if (overlay) {{
                overlay.classList.add("visible");
            }}

            window.location.href =
                "/random?time=" + Date.now();
        }}
    </script>


    <script>
        // loading-interaction-lock-v2
        (function () {{
            const htmlElement =
                document.documentElement;

            const body =
                document.body;

            function loadingIsActive() {{
                return body.classList.contains(
                    "is-loading"
                );
            }}

            function blockInteraction(event) {{
                if (!loadingIsActive()) {{
                    return;
                }}

                event.preventDefault();
                event.stopPropagation();

                if (event.stopImmediatePropagation) {{
                    event.stopImmediatePropagation();
                }}
            }}

            document.addEventListener(
                "wheel",
                blockInteraction,
                {{
                    capture: true,
                    passive: false
                }}
            );

            document.addEventListener(
                "touchmove",
                blockInteraction,
                {{
                    capture: true,
                    passive: false
                }}
            );

            [
                "click",
                "dblclick",
                "mousedown",
                "mouseup",
                "pointerdown",
                "pointerup",
                "contextmenu",
                "keydown"
            ].forEach(function (eventName) {{
                document.addEventListener(
                    eventName,
                    blockInteraction,
                    true
                );
            }});

            window.showLoadingAndChooseCar =
                function () {{
                    if (loadingIsActive()) {{
                        return;
                    }}

                    const overlay =
                        document.getElementById(
                            "loadingOverlay"
                        );

                    const button =
                        document.getElementById(
                            "randomButton"
                        );

                    htmlElement.classList.add(
                        "is-loading"
                    );

                    body.classList.add(
                        "is-loading"
                    );

                    if (button) {{
                        button.disabled = true;
                        button.textContent =
                            "Loading vehicle…";
                    }}

                    if (overlay) {{
                        overlay.classList.add(
                            "visible"
                        );

                        overlay.setAttribute(
                            "aria-hidden",
                            "false"
                        );
                    }}

                    setTimeout(function () {{
                        window.location.assign(
                            "/random?time="
                            + Date.now()
                        );
                    }}, 100);
                }};

            window.addEventListener(
                "pageshow",
                function () {{
                    htmlElement.classList.remove(
                        "is-loading"
                    );

                    body.classList.remove(
                        "is-loading"
                    );

                    const overlay =
                        document.getElementById(
                            "loadingOverlay"
                        );

                    const button =
                        document.getElementById(
                            "randomButton"
                        );

                    if (overlay) {{
                        overlay.classList.remove(
                            "visible"
                        );

                        overlay.setAttribute(
                            "aria-hidden",
                            "true"
                        );
                    }}

                    if (button) {{
                        button.disabled = false;
                        button.textContent =
                            "Choose Another Random Car";
                    }}
                }}
            );
        }})();
    </script>

</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Ignore the browser's automatic favicon request.
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        # Allow the homepage and randomized requests.
        if not (
            self.path == "/"
            or self.path.startswith("/random")
        ):
            self.send_response(404)
            self.end_headers()
            return

        try:
            result = make_prediction()
            page = make_html(result).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(page)),
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                self.wfile.write(page)
            except (BrokenPipeError, ConnectionResetError):
                # Browser refreshed or disconnected.
                return

        except Exception as error:
            print(f"Demo error: {error}")
            message = f"Demo error: {error}".encode("utf-8")

            try:
                self.send_response(500)
                self.send_header(
                    "Content-Type",
                    "text/plain; charset=utf-8",
                )
                self.send_header(
                    "Content-Length",
                    str(len(message)),
                )
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(message)

            except (BrokenPipeError, ConnectionResetError):
                return

    def log_message(self, format, *args):
        return


server = ThreadingHTTPServer(
    ("0.0.0.0", PORT),
    DemoHandler,
)

server.daemon_threads = True

server.daemon_threads = True

print(f"Demo running on port {PORT}")
print("Open http://JETSON_IP:8000")
print("Press Ctrl+C to stop.")

try:
    server.serve_forever()

except KeyboardInterrupt:
    print("\nStopping demo.")
    server.server_close()
