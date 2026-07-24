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

# --- Line-style weather icons (inline SVG), matched to the scenario condition labels ---
_SVG = ("<svg class='wx-ic' viewBox='0 0 24 24' fill='none' stroke-width='1.7' "
        "stroke-linecap='round' stroke-linejoin='round'>{}</svg>")
_AMBER, _SLATE, _BLUE, _CYAN, _GREY = "#f5a623", "#64748b", "#2f6df0", "#38bdf8", "#6b7280"
_CLOUD = f"<path stroke='{_SLATE}' d='M20 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25'/>"
WEATHER_ICONS = {
    "sun": _SVG.format(
        f"<g stroke='{_AMBER}'><circle cx='12' cy='12' r='4.2'/>"
        "<path d='M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2"
        "M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4'/></g>"),
    "partly": _SVG.format(
        f"<g stroke='{_AMBER}'><circle cx='8' cy='7.5' r='2.6'/>"
        "<path d='M8 2.4v1.3M3.4 7.5H2.1M4.9 4.4l-.9-.9M11.1 4.4l.9-.9'/></g>"
        f"<path stroke='{_SLATE}' d='M18 13h-1.05A6 6 0 1 0 8 18h10a4 4 0 0 0 0-8z'/>"),
    "cloud": _SVG.format(f"<path stroke='{_SLATE}' d='M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z'/>"),
    "rain": _SVG.format(_CLOUD + f"<path stroke='{_BLUE}' d='M8 15v3M12 16v4M16 15v3'/>"),
    "snow": _SVG.format(_CLOUD + f"<g stroke='{_CYAN}'><path d='M8 16h.01M12 18h.01M16 16h.01M10 20h.01M14 20h.01'/></g>"),
    "wind": _SVG.format(
        f"<path stroke='{_GREY}' d='M9.6 4.6A2 2 0 1 1 11 8H2m10.6 11.4A2 2 0 1 0 14 16H2"
        "m15.7-8.3A2.5 2.5 0 1 1 19.5 12H2'/>"),
    "storm": _SVG.format(
        f"<path stroke='{_SLATE}' d='M19 16.9A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 5.1 16'/>"
        f"<path stroke='{_AMBER}' d='M13 11l-4 6h6l-4 6'/>"),
    "fog": _SVG.format(f"<path stroke='{_SLATE}' d='M3 9h14M5 13h15M4 17h13'/>"),
}


def weather_icon(label):
    """Pick the line icon for a scenario's condition label (precip beats wind)."""
    l = label.lower()
    if "thunder" in l:
        return WEATHER_ICONS["storm"]
    if "snow" in l:
        return WEATHER_ICONS["snow"]
    if "rain" in l or "drizzle" in l:
        return WEATHER_ICONS["rain"]
    if "fog" in l:
        return WEATHER_ICONS["fog"]
    if "wind" in l or "gale" in l:
        return WEATHER_ICONS["wind"]
    if "overcast" in l:
        return WEATHER_ICONS["cloud"]
    if "partly" in l:
        return WEATHER_ICONS["partly"]
    return WEATHER_ICONS["sun"]


RISK_TEXT_COLOR = {"Low": "#1b7e3f", "Moderate": "#9a5b00", "High": "#b3261e"}


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
          .app-logo {{ height: 2em; vertical-align: -0.48em; margin-right: 0.45rem;
                       filter: brightness(0) invert(1); }}
          /* Departure-hour slider (key="hour_slider"). Streamlit 1.60's slider is REACT-ARIA,
             styled by emotion. The stable hooks are emotion `target` classes (confirmed from
             Slider.js in this build): .e23vpic5 = the fill bar (linear-gradient of primary =
             the red track), .e23vpic3 = the thumb circle (primary), .e23vpic4 = the value text.
             (react-aria's own classes are overridden by emotion so they don't render.) */
          .st-key-hour_slider .e23vpic5,
          .st-key-hour_slider .react-aria-SliderTrack {{
             background: #ffffff !important; background-color: #ffffff !important;
             background-image: none !important; }}
          .st-key-hour_slider .e23vpic3,
          .st-key-hour_slider .react-aria-SliderThumb {{
             background: #ffffff !important; background-color: #ffffff !important;
             background-image: none !important;
             border: 2px solid rgba(120,13,18,0.65) !important;
             box-shadow: 0 1px 5px rgba(0,0,0,0.45) !important; }}
          /* value text (e.g. "8 AM") and any tick labels: white */
          .st-key-hour_slider .e23vpic4,
          .st-key-hour_slider [data-testid="stSliderThumbValue"],
          .st-key-hour_slider [data-testid="stSliderTickBar"] * {{ color: #ffffff !important; }}
          /* White rounded result card */
          .result-card {{ background:#ffffff; border-radius:18px; padding:1.3rem 1.6rem;
             margin:0.4rem 0 0.8rem; color:#1a1a1a; box-shadow:0 8px 28px rgba(0,0,0,0.30); }}
          .risk-pill {{ display:inline-block; padding:0.3rem 0.95rem; border-radius:999px;
             font-weight:700; font-size:1.1rem; }}
          .risk-low {{ background:#e6f5ec; color:#1b7e3f; }}
          .risk-moderate {{ background:#fff3df; color:#9a5b00; }}
          .risk-high {{ background:#fde6e6; color:#b3261e; }}
          .route-line {{ color:#555; margin:0.55rem 0 1.1rem; font-size:1.02rem; font-weight:600; }}
          .metric-row {{ display:flex; justify-content:space-between; align-items:flex-start;
             gap:1.5rem; flex-wrap:wrap; }}
          .metric-right {{ text-align:right; }}
          .metric-label {{ color:#666; font-size:0.82rem; }}
          .metric-label.primary {{ color:#800020; font-size:1.08rem; font-weight:700; }}
          .metric-value {{ color:#111; font-size:1.7rem; font-weight:700; line-height:1.15; }}
          .metric-value.primary {{ color:#800020; font-size:3rem; }}
          .metric-note {{ color:#8a8a8a; font-size:0.74rem; margin-top:0.15rem; }}
          /* Weather-sensitivity card */
          .wx-title {{ font-weight:700; font-size:1.05rem; color:#1a1a1a; }}
          .wx-sub {{ color:#666; font-size:0.84rem; margin:0.25rem 0 0.9rem; }}
          .wx-row {{ display:flex; align-items:center; justify-content:space-between;
             padding:0.5rem 0; border-top:1px solid #eee; }}
          .wx-cond {{ font-weight:600; color:#1a1a1a; flex:1.5; display:flex; align-items:center; }}
          .wx-prob {{ color:#777; font-size:0.86rem; flex:1; text-align:center; }}
          .wx-min {{ font-weight:700; color:#111; flex:0.8; text-align:right; }}
          .wx-ic {{ width:24px; height:24px; margin-right:0.6rem; flex-shrink:0; }}
          .wx-outlook {{ margin-top:0.8rem; padding-top:0.7rem; border-top:2px solid #ddd;
             color:#1a1a1a; font-size:0.98rem; }}
          .wx-note {{ color:#8a8a8a; font-size:0.78rem; margin-top:0.4rem; }}
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
ladder = dc.weather_ladder(dep, arr, str(date), hour, bundle)
# When we have weather coverage, the headline is the weather-weighted outlook (the ladder
# below decomposes it); otherwise it's the typical-day prediction.
src = ladder["weighted"] if ladder else result
risk, emoji = src["risk"], src["emoji"]
q50, q75, q90 = src["q50"], src["q75"], src["q90"]
p_late, p_severe = src["p_late"], src["p_severe"]
route_n = result["route_n"]

st.divider()
risk_class = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}[risk]
hist = f"from {route_n:,} past flights on this route" if route_n else "little route history"
st.markdown(
    f"""
    <div class="result-card">
      <span class="risk-pill {risk_class}">{emoji} {risk} delay risk</span>
      <div class="route-line">{dep} → {arr} · {date:%a %d %b %Y} · {fmt_hour(hour)}</div>
      <div class="metric-row">
        <div class="metric">
          <div class="metric-label primary">Typical delay</div>
          <div class="metric-value primary">~{q50:.0f} min</div>
          <div class="metric-note">usual up to ~{q75:.0f} min · bad day ~{q90:.0f} min</div>
        </div>
        <div class="metric metric-right">
          <div class="metric-label">Chance of a delay</div>
          <div class="metric-value">{p_late*100:.0f}% <span style="font-size:0.95rem;color:#666">≥ 15 min</span></div>
          <div class="metric-note">{p_severe*100:.0f}% chance of a 60+ min delay · {hist}</div>
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

# ---- Weather sensitivity ladder ----
if ladder:
    rows_html = ""
    for r in ladder["rungs"]:
        color = RISK_TEXT_COLOR.get(r["risk"], "#111")
        rows_html += (
            f"<div class='wx-row'>"
            f"<span class='wx-cond'>{weather_icon(r['label'])}<span>{r['label']}</span></span>"
            f"<span class='wx-prob'>{r['prob']*100:.0f}% of days</span>"
            f"<span class='wx-min' style='color:{color}'>{r['emoji']} ~{r['disp_q50']:.0f} min "
            f"<span style='font-weight:500;font-size:0.82rem'>· {r['p_severe']*100:.0f}% 60+</span></span>"
            f"</div>"
        )
    w = ladder["weighted"]
    note = (
        "" if ladder["sensitive"]
        else "<div class='wx-note'>This flight isn't very weather-sensitive — "
             "weather shifts the typical delay by only a few minutes.</div>"
    )
    st.markdown(
        f"""
        <div class="result-card">
          <div class="wx-title">🌦️ Weather sensitivity at {dep} in {date:%B}</div>
          <div class="wx-sub">You can't know the weather when booking, so here's the typical delay
          (and chance of a 60+ min delay) under {dep}'s plausible {date:%B} conditions, and how
          often each occurs.</div>
          {rows_html}
          <div class="wx-outlook">Weather-weighted outlook: <b>{w['emoji']} {w['risk']}</b> ·
          typically ~{w['q50']:.0f} min · {w['p_severe']*100:.0f}% chance of 60+ min
          — this is the headline above.</div>
          {note}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Weather scenarios use 2016–2018 ERA5 records for the departure airport. Fog and "
        "thunderstorms aren't in this data source, so those aren't shown yet."
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
        **{"Chance of delay": alts["p_late"].map(lambda p: f"{p*100:.0f}% ≥ 15 min")},
        **{"Typical": alts["q50"].map(lambda m: f"~{m:.0f} min")},
    )[["Date", "Departure", "Risk", "Chance of delay", "Typical"]]
    st.dataframe(show, hide_index=True, width="stretch")

# ---- Honest footer ----
st.divider()
with st.expander("How this works (and its limits)"):
    st.markdown(
        f"""
This tool uses a **booking-time model** — it knows only what a traveller does before a
flight: the route, date, and time of day. It does **not** use signals you can't know in
advance (the aircraft's earlier delays, congestion). For **weather** — which you also can't
know at booking — it doesn't guess a forecast; instead it shows how the flight typically runs
across that airport's plausible weather for the month (the *weather sensitivity* card).

- **What it predicts:** the **chance** of a delay (a calibrated probability) and a **typical
  delay range** for this specific flight — not a single false-precise number.
- **How good is it:** on a held-out future period it separates 15-min-late flights from on-time
  ones with ROC-AUC **{bundle['metrics']['auc_late']}** (60+ min delays at **{bundle['metrics']['auc_severe']}**),
  and the probabilities are calibrated (Brier {bundle['metrics']['brier_late']}). *How many* minutes a
  specific flight slips is driven by day-of factors no booking-time tool can see — hence a range.
- **Weather sensitivity:** available for the busiest ~15 departure airports (most Tunisair
  departures); other airports show the estimate without a weather breakdown.
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
