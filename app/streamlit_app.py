"""Tunisair Delay-Alert — a booking-time delay-risk demo.

Pick a flight (by number or route) and a date; get a delay-risk category, an honest
"typical delay" range, action suggestions, and lower-risk alternative departure times.

Run:  uv run streamlit run app/streamlit_app.py
"""
import base64
import datetime as dt
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delay_core as dc

st.set_page_config(page_title="Tunisair Delay-Alert System", page_icon="✈️", layout="centered")

ASSETS = Path(__file__).parent / "assets"


def _find(stem):
    """First existing image file 'stem.ext' in the assets folder, or None."""
    return next((ASSETS / f"{stem}{ext}" for ext in (".png", ".jpg", ".jpeg")
                 if (ASSETS / f"{stem}{ext}").exists()), None)


def _data_uri(path):
    mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def inject_style():
    """Full-page red background (image at ~85% opacity if provided, else on-theme gradient)."""
    img = _find("background")
    if img:
        # 15% dark scrim over the image ≈ image shown at 85% opacity, and keeps text legible.
        layer = (f'linear-gradient(rgba(18,2,4,0.15), rgba(18,2,4,0.15)), '
                 f'url("{_data_uri(img)}")')
    else:
        layer = "radial-gradient(circle at 50% 22%, #7a0d12 0%, #3d0509 55%, #160609 100%)"
    st.markdown(
        f"""
        <style>
          .stApp {{ background: {layer} center/cover no-repeat fixed; }}
          .app-header {{ text-align: center; margin: 0.3rem 0 1.4rem; }}
          .app-header h1 {{ margin-bottom: 0.25rem; }}
          .app-header p {{ opacity: 0.9; margin-top: 0; }}
          /* Logo rendered entirely white, in place of the title emoji */
          .app-logo {{ height: 1.5em; vertical-align: -0.35em; margin-right: 0.4rem;
                       filter: brightness(0) invert(1); }}
          /* Departure-hour slider (keyed via key="hour_slider"): white track, blue dot, white labels */
          .st-key-hour_slider [role="slider"] {{ background-color: #1d4ed8 !important;
             border-color: #1d4ed8 !important; box-shadow: 0 0 0 3px rgba(29,78,216,0.35) !important; }}
          .st-key-hour_slider [data-baseweb="slider"] > div > div,
          .st-key-hour_slider [data-baseweb="slider"] > div > div > div,
          .st-key-hour_slider [data-baseweb="slider"] > div > div > div > div {{
             background: #ffffff !important; }}
          .st-key-hour_slider [data-baseweb="slider"] * {{ color: #ffffff !important; }}
          /* White rounded result card */
          .result-card {{ background:#ffffff; border-radius:18px; padding:1.3rem 1.6rem;
             margin:0.4rem 0 0.8rem; color:#1a1a1a; box-shadow:0 8px 28px rgba(0,0,0,0.30); }}
          .risk-pill {{ display:inline-block; padding:0.3rem 0.95rem; border-radius:999px;
             font-weight:700; font-size:1.1rem; }}
          .risk-low {{ background:#e6f5ec; color:#1b7e3f; }}
          .risk-moderate {{ background:#fff3df; color:#9a5b00; }}
          .risk-high {{ background:#fde6e6; color:#b3261e; }}
          .route-line {{ color:#555; margin:0.55rem 0 1.1rem; font-size:1.02rem; font-weight:600; }}
          .metric-row {{ display:flex; gap:3rem; flex-wrap:wrap; }}
          .metric-label {{ color:#666; font-size:0.82rem; }}
          .metric-value {{ color:#111; font-size:1.7rem; font-weight:700; line-height:1.15; }}
          .metric-note {{ color:#8a8a8a; font-size:0.74rem; margin-top:0.15rem; }}
          .logo-word {{ color:#ffffff; letter-spacing:1px; }}
          .page-footer {{ text-align:center; opacity:0.82; font-size:0.83rem; margin-top:0.4rem; }}
          .page-footer b {{ color:#ffffff; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_style()


@st.cache_resource
def get_bundle():
    return dc.load_bundle()


bundle = get_bundle()
schedule = bundle["schedule"]

_logo = _find("logo")
# Use the white Tunisair logo once saved to app/assets/logo.png; until then, show the
# wordmark (no airplane emoji).
_logo_html = (f"<img src='{_data_uri(_logo)}' class='app-logo' alt='Tunisair'/> "
              if _logo else "<span class='logo-word'>TUNISAIR</span> ")
st.markdown(
    "<div class='app-header'>"
    f"<h1>{_logo_html}Tunisair Delay-Alert System</h1>"
    "<p>Estimate a flight's delay <b>risk</b> before you book — from route, date and time of day.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---- Inputs ----
mode = st.radio("Choose a flight by:", ["Flight number", "Route"], horizontal=True)

if mode == "Flight number":
    fltid = st.selectbox(
        "Flight number", schedule["FLTID"].tolist(),
        help="Tunisair flight numbers seen in 2016–2018, most frequent first.",
    )
    row = schedule[schedule["FLTID"] == fltid].iloc[0]
    dep, arr, default_hour = row["DEPSTN"], row["ARRSTN"], int(row["typical_hour"])
    st.write(f"Route: **{dep} → {arr}**  ·  typical departure ~**{default_hour:02d}:00**")
else:
    airports = sorted(set(schedule["DEPSTN"]) | set(schedule["ARRSTN"]))
    c1, c2 = st.columns(2)
    dep = c1.selectbox("From (departure)", airports, index=airports.index("TUN") if "TUN" in airports else 0)
    arr = c2.selectbox("To (arrival)", airports, index=airports.index("ORY") if "ORY" in airports else 1)
    route_rows = schedule.query("DEPSTN == @dep and ARRSTN == @arr")
    default_hour = int(route_rows["typical_hour"].mode().iloc[0]) if not route_rows.empty else 12

def fmt_hour(h):
    suffix = "AM" if h < 12 else "PM"
    return f"{h % 12 or 12} {suffix}"


c3, c4 = st.columns(2)
date = c3.date_input("Date", value=dt.date(2018, 8, 12), min_value=dt.date(2016, 1, 1))
hour = c4.select_slider(
    "Departure hour", options=list(range(24)), value=default_hour, format_func=fmt_hour,
    key="hour_slider",
)

# ---- Prediction ----
if dep == arr:
    st.info("Pick two different airports.")
    st.stop()

result = dc.predict(dep, arr, str(date), hour, bundle)
risk, emoji = result["risk"], result["emoji"]

st.divider()
risk_class = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}[risk]
note = (f"middle-half of {result['route_n']:,} past flights · median {result['p50']:.0f} min"
        if result["route_n"] else "network-wide typical range (little history for this route)")
st.markdown(
    f"""
    <div class="result-card">
      <span class="risk-pill {risk_class}">{emoji} {risk} delay risk</span>
      <div class="route-line">{dep} → {arr} · {date:%a %d %b %Y} · {fmt_hour(hour)}</div>
      <div class="metric-row">
        <div class="metric">
          <div class="metric-label">Model risk estimate</div>
          <div class="metric-value">~{result['pred_minutes']:.0f} min</div>
        </div>
        <div class="metric">
          <div class="metric-label">Typical delay on this route</div>
          <div class="metric-value">{result['p25']:.0f}–{result['p75']:.0f} min</div>
          <div class="metric-note">{note}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if date.year > 2018:
    st.caption(
        "ℹ️ This date is beyond our 2016–2018 data. The estimate projects historical patterns "
        "for this route, month and time of day forward — it is **not** a year-specific forecast, "
        "and assumes delay behaviour hasn't fundamentally changed since 2018."
    )

st.divider()
st.markdown("**What to do:**")
for tip in dc.action_suggestions(risk):
    st.markdown(f"- {tip}")
st.divider()

# ---- Alternatives ----
alts = dc.alternatives(dep, arr, str(date), bundle)
if not alts.empty and len(alts) > 1:
    st.markdown(f"**Calmer departure times on {dep} → {arr}:**")
    show = alts.assign(
        Date=f"{date:%a %d %b %Y}",
        Departure=alts["hour"].map(fmt_hour),
        Risk=alts["emoji"] + " " + alts["risk"],
        Estimate=alts["pred_minutes"].map(lambda m: f"~{m:.0f} min"),
    )[["Date", "Departure", "Risk", "Estimate"]]
    st.dataframe(show, hide_index=True, width="stretch")

# ---- Honest footer ----
st.divider()
with st.expander("How this works (and its limits)"):
    st.markdown(
        f"""
This tool uses a **booking-time model** — it knows only what a traveller does before a
flight: the route, date, and time of day. It deliberately does **not** use day-of signals
(the aircraft's earlier delays, congestion, weather), because you can't know those in advance.

- **What it's good at:** the **risk category** — it separates low- from high-risk flights well.
- **What it can't do:** predict the exact minutes. Held-out error is {bundle['holdout_rmse']:.0f} min
  (vs {bundle['baseline_rmse']:.0f} for guessing the average) — *how many* minutes a specific flight
  slips is driven by day-of factors no booking-time tool can see. That's why we show a **range**, not
  a false-precise number.
- **MVP scope:** a demo on 2016–2018 historical Tunisair data, not a live system.
- **Future dates:** the model has no notion of *year* — it maps a flight's route, month, weekday
  and time of day to the delay patterns seen in 2016–2018. So a 2026 flight gets the historical
  pattern for a flight like it, projected forward — useful as a typical-risk guide, but not a
  year-specific forecast, and it assumes those patterns still hold.

*An airline operations team, with the aircraft's real-time prior-leg status, can predict far more
accurately (held-out RMSE ~109 min) — but that data isn't available at booking time.*
"""
    )

# ---- Disclaimer & authors ----
st.divider()
st.markdown(
    """
    <div class="page-footer">
      <p><b>Disclaimer:</b> This is an educational student prototype — not an official Tunisair
      product, and not affiliated with or endorsed by Tunisair. Predictions are statistical estimates
      based on historical 2016–2018 data and are <b>not guarantees</b> of actual flight performance.</p>
      <p>Created by <b>Alex</b> &amp; <b>Sulu</b> · Tunisair Flight Delay Prediction capstone (2026).</p>
    </div>
    """,
    unsafe_allow_html=True,
)
