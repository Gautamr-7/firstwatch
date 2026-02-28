import streamlit as st
import pandas as pd
import json
import time
import os
from datetime import datetime

# --- UI CONFIG ---
st.set_page_config(page_title="Volt AI Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #f0f2f5 !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAV BAR ── */
.nav-bar { background: #1a1a2e; padding: 0 28px; height: 60px; display: flex; align-items: center; justify-content: space-between; border-bottom: 3px solid #e63946; }
.nav-logo { font-size: 20px; font-weight: 700; color: #ffffff; letter-spacing: 1px; }
.nav-logo span { color: #e63946; }
.nav-right { display: flex; align-items: center; gap: 24px; font-size: 13px; color: #aab0c0; }
.live-pill { background: #e63946; color: white; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; display: flex; align-items: center; gap: 5px; }
.live-dot { width: 6px; height: 6px; background: white; border-radius: 50%; animation: blink 1.2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── STATUS BANNER ── */
.banner-alert { background: #fff1f0; border-left: 5px solid #e63946; padding: 12px 28px; display: flex; align-items: center; gap: 14px; font-size: 14px; color: #c0392b; font-weight: 500; }
.banner-ok { background: #f0faf4; border-left: 5px solid #27ae60; padding: 12px 28px; display: flex; align-items: center; gap: 14px; font-size: 14px; color: #1e8449; font-weight: 500; }
.banner-tag { background: #e63946; color: white; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 1px; white-space: nowrap; }
.banner-tag-ok { background: #27ae60; color: white; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 1px; white-space: nowrap; }

/* ── SECTION TITLE ── */
.section-title { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; color: #7a8299; text-transform: uppercase; margin-bottom: 10px; margin-top: 18px; padding-bottom: 6px; border-bottom: 1px solid #dde1ea; }

/* ── VIDEO PANEL ── */
.video-wrap { background: #1a1a2e; border-radius: 8px; overflow: hidden; border: 1px solid #d0d5e0; }

/* ── METRIC CARDS ── */
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px; }
.metric-card { background: white; border-radius: 8px; padding: 16px; border: 1px solid #dde1ea; border-top: 3px solid #3a86ff; }
.metric-card.warn  { border-top-color: #e63946; }
.metric-card.green { border-top-color: #27ae60; }
.metric-card.orange { border-top-color: #f39c12; }
.metric-label { font-size: 11px; font-weight: 600; color: #7a8299; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; line-height: 1; }
.metric-value.red { color: #e63946; }
.metric-sub { font-size: 12px; color: #9aa0b0; margin-top: 4px; }

/* ── CARD ── */
.card { background: white; border-radius: 8px; border: 1px solid #dde1ea; overflow: hidden; margin-bottom: 14px; }
.card-header { padding: 10px 14px; background: #f8f9fc; border-bottom: 1px solid #dde1ea; font-size: 12px; font-weight: 700; color: #3a4060; text-transform: uppercase; letter-spacing: 1px; }
.card-body { padding: 14px; }

/* ── TIMELINE ── */
.tl-step { display: flex; align-items: flex-start; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f0f2f5; }
.tl-step:last-child { border-bottom: none; }
.tl-icon { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 15px; flex-shrink: 0; background: #f0f2f5; border: 2px solid #dde1ea; }
.tl-icon.active   { background: #eaf4ff; border-color: #3a86ff; }
.tl-icon.critical { background: #fff1f0; border-color: #e63946; }
.tl-icon.pending  { opacity: 0.45; }
.tl-body { flex: 1; padding-top: 2px; }
.tl-title { font-size: 14px; font-weight: 600; color: #1a1a2e; }
.tl-title.pending { color: #9aa0b0; font-weight: 400; }
.tl-detail { font-size: 12px; color: #7a8299; margin-top: 2px; }

/* ── LOG ── */
.log-wrap { background: #1a1a2e; border-radius: 8px; padding: 12px 16px; font-family: 'Courier New', monospace; font-size: 12px; line-height: 2; max-height: 150px; overflow-y: auto; border: 1px solid #dde1ea; }
.log-time { color: #6a7490; margin-right: 10px; }
.log-ok   { color: #2ecc71; }
.log-warn { color: #f39c12; }
.log-crit { color: #e74c3c; }

[data-testid="stHorizontalBlock"] { gap: 20px !important; padding: 0 24px 20px !important; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOAD ---
def load_state():
    if os.path.exists("state.json"):
        try:
            with open("state.json", "r") as f:
                return json.load(f)
        except:
            pass
    return {"accident": False}

state = load_state()
accident = state.get("accident", False)
severity = state.get("severity", 0)
dispatch_time = state.get("dispatch_time", "")
crash_timestamp = state.get("crash_timestamp", 0)

now = datetime.now()
ts = now.strftime("%H:%M:%S")
date_str = now.strftime("%d %b %Y")

# ── NAV BAR ──
st.markdown(f"""
<div class="nav-bar">
  <div class="nav-logo">Guardian<span>AI</span> &nbsp;·&nbsp; Flink Emergency Response</div>
  <div class="nav-right">
    <span>CAM_001 · CUSAT Main Gate</span>
    <span>{date_str} &nbsp; {ts}</span>
    <div class="live-pill"><div class="live-dot"></div> LIVE</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── STATUS BANNER ──
if accident:
    st.markdown(f"""
    <div class="banner-alert">
      <span class="banner-tag">⚠ COLLISION DETECTED</span>
      Accident confirmed on CAM_001 &nbsp;|&nbsp; Severity: <strong>{severity}%</strong> &nbsp;|&nbsp; Dispatch initiated at {dispatch_time}
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="banner-ok">
      <span class="banner-tag-ok">✓ ALL CLEAR</span>
      All systems operational &nbsp;·&nbsp; No incidents detected &nbsp;·&nbsp; Monitoring active
    </div>""", unsafe_allow_html=True)

# ── COLUMNS ──
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="section-title">Live Camera Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="video-wrap">', unsafe_allow_html=True)
    st.image("https://via.placeholder.com/960x480/1a1a2e/4a6090?text=Camera+Feed+—+CAM_001", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    det_val = f"{severity}%" if accident else "Clear"
    det_cls = "warn" if accident else "green"
    det_sub = "Collision confirmed" if accident else "No incident detected"
    eta_val = "5 min" if accident else "—"
    eta_sub = "Ambulance ETA" if accident else "Standby"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card {det_cls}">
        <div class="metric-label">Detection Status</div>
        <div class="metric-value {'red' if accident else ''}">{det_val}</div>
        <div class="metric-sub">{det_sub}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Cameras Online</div>
        <div class="metric-value">4 / 4</div>
        <div class="metric-sub">Zone NH-66 active</div>
      </div>
      <div class="metric-card orange">
        <div class="metric-label">Response ETA</div>
        <div class="metric-value">{eta_val}</div>
        <div class="metric-sub">{eta_sub}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Frames Analysed</div>
        <div class="metric-value">2,418</div>
        <div class="metric-sub">This session</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:22px;">Incident Log</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="log-wrap">
      <div><span class="log-time">08:12:03</span><span class="log-ok">SYSTEM BOOT — All modules nominal</span></div>
      <div><span class="log-time">09:34:17</span><span class="log-ok">CAM_002 reconnected — signal restored</span></div>
      <div><span class="log-time">11:05:44</span><span class="log-warn">ALERT — High vehicle density on NH-66</span></div>
      <div><span class="log-time">13:22:09</span><span class="log-ok">Routine scan — no incidents detected</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-title">Live Map Radar</div>', unsafe_allow_html=True)
    
    # Coordinates mapping
   	# Coordinates mapping
    cam_lat, cam_lon = 10.0415, 76.3243 # CUSAT Main Gate
    hosp_lat, hosp_lon = 10.0403, 76.3214 # Kinder Hospital
    amb_lat, amb_lon = 10.0480, 76.3180 # Nearby Ambulance
    
    if accident:
        elapsed_time = time.time() - crash_timestamp
        countdown = max(0, 10 - int(elapsed_time))
        
        if elapsed_time >= 10:
            data = {
                'lat': [cam_lat, hosp_lat, amb_lat],
                'lon': [cam_lon, hosp_lon, amb_lon],
                'color': ['#ff0000', '#0000ff', '#f39c12'], 
                'size': [70, 70, 70]
            }
            map_data = pd.DataFrame(data)
            st.map(map_data, latitude='lat', longitude='lon', color='color', size='size', zoom=12, height=220)
            st.markdown("<div style='font-size:11px; text-align:center; margin-top:5px;'>🔴 Crash Site &nbsp;|&nbsp; 🔵 Kinder Hospital &nbsp;|&nbsp; 🟠 Dispatched Ambulance</div>", unsafe_allow_html=True)
        else:
            map_data = pd.DataFrame({'lat': [cam_lat], 'lon': [cam_lon], 'color': ['#ff0000'], 'size': [50]})
            st.map(map_data, latitude='lat', longitude='lon', color='color', size='size', zoom=15, height=220)
            st.markdown(f"<div style='font-size:11px; text-align:center; margin-top:5px; color:#e63946;'><strong>Routing Response Units... ({countdown}s)</strong></div>", unsafe_allow_html=True)
    else:
        map_data = pd.DataFrame({'lat': [cam_lat], 'lon': [cam_lon], 'color': ['#27ae60'], 'size': [20]})
        st.map(map_data, latitude='lat', longitude='lon', color='color', size='size', zoom=14, height=220)

    st.markdown('<div class="section-title">Agent Dispatch</div>', unsafe_allow_html=True)

    if accident:
        if elapsed_time >= 10:
            steps = [
                ("🔍", "active",   "Vision Agent",   "Accident classified · 97% confidence"),
                ("🚑", "active",   "Ambulance",       "Dispatched · ETA 5 min"),
                ("🏥", "active",   "Hospital Alert",  "Bed reserved @ Kinder Hospital"),
                ("📱", "critical", "Family Notified", "WhatsApp Alert Sent Successfully"),
                ("🚔", "active",   "Traffic Mgmt",    "Signal override active"),
            ]
        else:
            steps = [
                ("🔍", "active",   "Vision Agent",   "Accident classified · 97% confidence"),
                ("⏳", "pending",  "Orchestration",   f"Calculating optimal route... ({countdown}s)"),
                ("🚑", "pending",  "Ambulance",       "Standby"),
                ("🏥", "pending",  "Hospital Alert",  "Standby"),
                ("📱", "pending",  "Family Notify",   "Standby"),
            ]
    else:
        steps = [
            ("🔍", "active",  "Vision Agent",   "Scanning · no incidents"),
            ("🚑", "pending", "Ambulance",       "Standby"),
            ("🏥", "pending", "Hospital Alert",  "Standby"),
            ("📱", "pending", "Family Notify",   "Standby"),
            ("🚔", "pending", "Traffic Mgmt",    "Standby"),
        ]

    tl = '<div class="card"><div class="card-header">Response Chain</div><div class="card-body">'
    for icon, s, title, detail in steps:
        tl += f"""
        <div class="tl-step">
          <div class="tl-icon {s}">{icon}</div>
          <div class="tl-body">
            <div class="tl-title {'pending' if s=='pending' else ''}">{title}</div>
            <div class="tl-detail">{detail}</div>
          </div>
        </div>"""
    tl += "</div></div>"
    st.markdown(tl, unsafe_allow_html=True)

# ── AUTO-REFRESH ──
time.sleep(2)

st.rerun()
