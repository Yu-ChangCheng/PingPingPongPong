"""Static share card: Tomorrow's basket + prices (IG-friendly PNG/JPEG).

Optional email of the image via SMTP env vars (see maybe_email_share_card).
"""
from __future__ import annotations

import mimetypes
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Instagram feed square (px)
CARD_W, CARD_H = 1080, 1080
MARGIN = 56
ROW_H = 72
HEADER_EXTRA = 140


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        sub = "Fonts"
        if bold:
            candidates += [
                os.path.join(windir, sub, "arialbd.ttf"),
                os.path.join(windir, sub, "segoeuib.ttf"),
            ]
        candidates += [
            os.path.join(windir, sub, "arial.ttf"),
            os.path.join(windir, sub, "segoeui.ttf"),
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_tomorrow_predictions_card(
    latest: pd.DataFrame,
    panel: pd.DataFrame,
    long_n: int,
    out_png: Path,
    run_at: datetime | None = None,
    *,
    also_jpeg: bool | None = None,
) -> tuple[Path, Path | None]:
    """Draw top-`long_n` basket with adj. close + pred xret. Returns (png_path, jpeg_or_none)."""
    run_at = run_at or datetime.now(timezone.utc)
    last_date = pd.to_datetime(panel["date"].max())
    close_date_str = last_date.strftime("%Y-%m-%d")
    run_date_str = run_at.strftime("%Y-%m-%d %H:%M UTC")

    last_prices = (
        panel[pd.to_datetime(panel["date"]) == last_date]
        .set_index("ticker")["adj_close"]
    )
    df = latest.copy()
    df["close"] = df["ticker"].map(last_prices).astype(float)
    df = df.sort_values("y_pred", ascending=False).reset_index(drop=True)
    basket = df.head(long_n)

    if also_jpeg is None:
        also_jpeg = os.environ.get("SHARE_CARD_JPEG", "").strip().lower() in (
            "1", "true", "yes",
        )

    nrows = len(basket)
    # Dynamic height if many rows (still cap for IG)
    inner_h = HEADER_EXTRA + ROW_H + nrows * ROW_H + MARGIN
    h = min(max(inner_h, 900), CARD_H)

    img = Image.new("RGB", (CARD_W, h), color=(246, 246, 243))
    draw = ImageDraw.Draw(img)
    title_font = _load_font(42, bold=True)
    sub_font = _load_font(26, bold=False)
    head_font = _load_font(24, bold=True)
    cell_font = _load_font(28, bold=False)
    small_font = _load_font(22, bold=False)

    accent = (11, 110, 79)
    muted = (106, 106, 106)
    fg = (26, 26, 26)

    y = MARGIN
    draw.text((MARGIN, y), "Tomorrow's basket", fill=accent, font=title_font)
    y += 52
    draw.text(
        (MARGIN, y),
        f"Close data: {close_date_str}  ·  Generated: {run_date_str}",
        fill=muted,
        font=sub_font,
    )
    y += 56

    # Header row background
    hdr_y1, hdr_y2 = y, y + ROW_H
    draw.rectangle((MARGIN, hdr_y1, CARD_W - MARGIN, hdr_y2), fill=(240, 240, 235))
    cols = [("#", 70), ("Ticker", 200), ("Adj. close", 320), ("Pred xret", 360)]
    x = MARGIN + 12
    for label, w in cols:
        draw.text((x, y + 18), label, fill=fg, font=head_font)
        x += w

    y += ROW_H
    for i, row in enumerate(basket.itertuples(), start=1):
        yy1, yy2 = y, y + ROW_H
        stripe = (255, 255, 255) if i % 2 else (250, 250, 248)
        draw.rectangle((MARGIN, yy1, CARD_W - MARGIN, yy2), fill=stripe)
        x = MARGIN + 12
        draw.text((x + 10, y + 18), f"{i}", fill=fg, font=cell_font)
        x += cols[0][1]
        draw.text((x, y + 18), str(row.ticker), fill=accent, font=cell_font)
        x += cols[1][1]
        close_v = float(row.close) if np.isfinite(row.close) else float("nan")
        close_txt = f"${close_v:,.2f}" if np.isfinite(close_v) else "n/a"
        draw.text((x, y + 18), close_txt, fill=fg, font=cell_font)
        x += cols[2][1]
        px = float(row.y_pred) * 100
        col = accent if px >= 0 else (179, 64, 48)
        draw.text((x, y + 18), f"{px:+.2f}%", fill=col, font=cell_font)
        y += ROW_H

    y += 18
    draw.text(
        (MARGIN, y),
        "Educational only — not investment advice.",
        fill=muted,
        font=small_font,
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # IG: save at full width; if h < CARD_H, paste onto square canvas
    if h < CARD_H:
        square = Image.new("RGB", (CARD_W, CARD_H), color=(246, 246, 243))
        square.paste(img, (0, (CARD_H - h) // 2))
        img = square
    elif h > CARD_H:
        img = img.crop((0, 0, CARD_W, CARD_H))

    img.save(out_png, format="PNG", optimize=True)

    jpg_path: Path | None = None
    if also_jpeg:
        jpg_path = out_png.with_suffix(".jpg")
        rgb = img.convert("RGB")
        rgb.save(jpg_path, format="JPEG", quality=92, optimize=True)

    return out_png, jpg_path


def maybe_email_share_card(paths: list[Path], subject: str, body: str) -> bool:
    """If DAILY_EMAIL_TO is set, email the given files via SMTP. Returns True if sent."""
    to_raw = os.environ.get("DAILY_EMAIL_TO", "").strip()
    if not to_raw or not paths:
        return False
    host = os.environ.get("SMTP_HOST", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    if not host or not user or not password:
        print("  email: DAILY_EMAIL_TO set but SMTP_HOST / SMTP_USER / SMTP_PASSWORD "
              "missing — skip send")
        return False

    port = int(os.environ.get("SMTP_PORT", "587") or 587)
    use_tls = os.environ.get("SMTP_TLS", "1").strip().lower() not in ("0", "false", "no")
    mail_from = os.environ.get("SMTP_FROM", user).strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = to_raw
    msg.set_content(body)
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        ctype, _ = mimetypes.guess_type(str(p))
        if ctype is None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        msg.add_attachment(
            p.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=p.name,
        )

    with smtplib.SMTP(host, port, timeout=60) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"  email: sent share card to {to_raw}")
    return True
