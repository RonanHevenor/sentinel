"""
The Sentinel Program, a light-based heating zone usage tracker. 
Copyright (C) 2025  Ronan Hevenor

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""


import cv2
import time
import os
import sqlite3
import csv
import io
import json
import logging
import threading
import psutil
from datetime import datetime
from flask import Flask, Response, request, jsonify, redirect, url_for

# --- Configuration ---
CAMERA_INDEX = 0
RESOLUTION = (640, 480)
FRAME_RATE = 1.0 

# Storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "monitor.db")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
QC_DIR = os.path.join(BASE_DIR, "qc")
QC_RETENTION_COUNT = 10

# Web
WEB_PORT = 8000

# --- Global State ---
SERVER_START_TIME = datetime.now()
current_frame = None
latest_result = {
    "state_int": 0,
    "state_bin": "000000",
    "timestamp": "",
    "boxes": [],
    "power_on": False,
    "power_brightness": 0
}
lock = threading.Lock()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Config Management ---
DEFAULT_CONFIG = {
    "device_id": "RPI-01",
    "location": "MAIN_LAB",
    "sections": ["Section 1", "Section 2", "Section 3", "Section 4", "Section 5", "Section 6"],
    "box_size": 64,
    "box_spacing": 58,
    "box_offset_x": 0,
    "box_offset_y": 0,
    "led_positions": None
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
            for key in DEFAULT_CONFIG:
                if key not in cfg:
                    cfg[key] = DEFAULT_CONFIG[key]
            return cfg
    except:
        return DEFAULT_CONFIG

def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=4)

sys_config = load_config()

# --- Database Management ---

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS logs 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      timestamp TEXT, 
                      state_int INTEGER,
                      power_on INTEGER DEFAULT 1)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp ON logs (timestamp)''')
        c.execute('''CREATE TABLE IF NOT EXISTS monthly_stats
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      year INTEGER,
                      month INTEGER,
                      total_hours REAL,
                      section_hours TEXT,
                      relative_share TEXT,
                      archived_at TEXT,
                      UNIQUE(year, month))''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Init Error: {e}")

def db_insert_state(state_int, power_on):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO logs (timestamp, state_int, power_on) VALUES (?, ?, ?)", 
                  (timestamp_str, state_int, 1 if power_on else 0))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Write Error: {e}")

def db_get_month_data(month_str):
    rows = []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp, state_int, COALESCE(power_on, 1) FROM logs WHERE timestamp LIKE ? ORDER BY timestamp ASC", (f"{month_str}%",))
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"DB Read Error: {e}")
    return rows

def db_get_recent_logs(limit=500):
    rows = []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp, state_int, COALESCE(power_on, 1) FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"DB Recent Read Error: {e}")
    return rows

def db_archive_month(year, month, stats):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO monthly_stats 
                     (year, month, total_hours, section_hours, relative_share, archived_at)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (year, month, stats['total_duration_hours'],
                   json.dumps(stats['section_hours']),
                   json.dumps(stats['relative_share']),
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        logger.info(f"Archived stats for {year}-{month:02d}")
    except Exception as e:
        logger.error(f"DB Archive Error: {e}")

def db_get_archived_months():
    rows = []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT DISTINCT year, month FROM monthly_stats ORDER BY year DESC, month DESC")
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"DB Get Archives Error: {e}")
    return rows

def db_get_archived_stats(year, month):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT total_hours, section_hours, relative_share FROM monthly_stats WHERE year=? AND month=?", (year, month))
        row = c.fetchone()
        conn.close()
        if row:
            return {
                'total_duration_hours': row[0],
                'section_hours': json.loads(row[1]),
                'relative_share': json.loads(row[2]),
                'is_archived': True
            }
    except Exception as e:
        logger.error(f"DB Get Archived Stats Error: {e}")
    return None

def db_get_available_years():
    years = set()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT DISTINCT year FROM monthly_stats")
        for row in c.fetchall():
            years.add(row[0])
        c.execute("SELECT DISTINCT substr(timestamp, 1, 4) FROM logs")
        for row in c.fetchall():
            try:
                years.add(int(row[0]))
            except:
                pass
        conn.close()
    except Exception as e:
        logger.error(f"DB Get Years Error: {e}")
    years.add(datetime.now().year)
    return sorted(list(years), reverse=True)

def check_and_archive_previous_month():
    now = datetime.now()
    if now.month == 1:
        prev_year, prev_month = now.year - 1, 12
    else:
        prev_year, prev_month = now.year, now.month - 1
     
    existing = db_get_archived_stats(prev_year, prev_month)
    if existing:
        return
     
    month_str = f"{prev_year}-{prev_month:02d}"
    rows = db_get_month_data(month_str)
     
    if len(rows) > 0:
        stats = calculate_month_stats(prev_year, prev_month)
        if stats['total_duration_hours'] > 0:
            db_archive_month(prev_year, prev_month, stats)

def calculate_month_stats(year, month):
    month_str = f"{year}-{month:02d}"
    rows = db_get_month_data(month_str)
     
    total_on_time = [0] * 6
    total_powered_duration = 0
    last_dt, last_state, last_power = None, 0, 1
     
    for row in rows:
        try:
            dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            state = int(row[1])
            power = int(row[2]) if len(row) > 2 else 1
             
            if last_dt:
                diff = (dt - last_dt).total_seconds()
                if last_power:
                    total_powered_duration += diff
                    for i in range(6):
                        if (last_state >> i) & 1:
                            total_on_time[i] += diff
            last_dt, last_state, last_power = dt, state, power
        except ValueError:
            continue

    # --- CRITICAL FIX: Account for the current "live" spree ---
    # If we are looking at the current month, we must add the time 
    # from the LAST log entry until NOW.
    now = datetime.now()
    if last_dt and now.year == year and now.month == month:
        diff_live = (now - last_dt).total_seconds()
        if diff_live > 0 and last_power:
            total_powered_duration += diff_live
            for i in range(6):
                if (last_state >> i) & 1:
                    total_on_time[i] += diff_live

    if total_powered_duration == 0:
        total_powered_duration = 1
     
    uptimes = [(t / total_powered_duration * 100) for t in total_on_time]
    sum_on_time = sum(total_on_time)
    relative_share = [(t / sum_on_time * 100) for t in total_on_time] if sum_on_time > 0 else [0] * 6

    return {
        "month": month_str,
        "total_duration_hours": total_powered_duration / 3600,
        "section_hours": [t / 3600 for t in total_on_time],
        "uptimes": uptimes,
        "relative_share": relative_share
    }

def get_monthly_stats():
    now = datetime.now()
    return calculate_month_stats(now.year, now.month)

def get_alltime_stats():
    """Calculate all-time cumulative stats."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get all logs ordered by time
        c.execute("SELECT timestamp, state_int, COALESCE(power_on, 1) FROM logs ORDER BY timestamp ASC")
        rows = c.fetchall()
        
        # Get first timestamp and count
        c.execute("SELECT MIN(timestamp), COUNT(*) FROM logs")
        time_range = c.fetchone()
        
        conn.close()
        
        if not rows or time_range[1] == 0:
            return {
                "total_duration_hours": 0,
                "section_hours": [0] * 6,
                "relative_share": [0] * 6,
                "first_log": None,
                "total_records": 0,
                "days_tracked": 0,
                "peak_section": None,
                "peak_section_hours": 0,
                "avg_daily_hours": 0
            }
        
        total_on_time = [0] * 6
        total_powered_duration = 0
        last_dt, last_state, last_power = None, 0, 1
        
        for row in rows:
            try:
                dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                state = int(row[1])
                power = int(row[2]) if len(row) > 2 else 1
                
                if last_dt:
                    diff = (dt - last_dt).total_seconds()
                    if last_power:
                        total_powered_duration += diff
                        for i in range(6):
                            if (last_state >> i) & 1:
                                total_on_time[i] += diff
                last_dt, last_state, last_power = dt, state, power
            except ValueError:
                continue

        # --- CRITICAL FIX: Account for the current "live" spree for all-time ---
        # We must add the time from the LAST log entry until NOW.
        if last_dt:
            now = datetime.now()
            diff_live = (now - last_dt).total_seconds()
            if diff_live > 0 and last_power:
                total_powered_duration += diff_live
                for i in range(6):
                    if (last_state >> i) & 1:
                        total_on_time[i] += diff_live
        
        if total_powered_duration == 0:
            total_powered_duration = 1
        
        sum_on_time = sum(total_on_time)
        relative_share = [(t / sum_on_time * 100) for t in total_on_time] if sum_on_time > 0 else [0] * 6
        section_hours = [t / 3600 for t in total_on_time]
        
        # Find peak section
        peak_idx = section_hours.index(max(section_hours)) if max(section_hours) > 0 else None
        
        # Calculate days tracked from first record to NOW (real-time)
        first_dt = datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S") if time_range[0] else None
        now = datetime.now()
        
        if first_dt:
            delta = now - first_dt
            days_tracked = delta.total_seconds() / 86400  # Fractional days
        else:
            days_tracked = 0
        
        return {
            "total_duration_hours": total_powered_duration / 3600,
            "section_hours": section_hours,
            "relative_share": relative_share,
            "first_log": time_range[0],
            "total_records": time_range[1],
            "days_tracked": days_tracked,
            "peak_section": peak_idx,
            "peak_section_hours": max(section_hours) if section_hours else 0,
            "avg_daily_hours": (total_powered_duration / 3600) / days_tracked if days_tracked > 0 else 0
        }
    except Exception as e:
        logger.error(f"All-time stats error: {e}")
        return {
            "total_duration_hours": 0,
            "section_hours": [0] * 6,
            "relative_share": [0] * 6,
            "first_log": None,
            "total_records": 0,
            "days_tracked": 0,
            "peak_section": None,
            "peak_section_hours": 0,
            "avg_daily_hours": 0
        }

# --- Helper Functions ---

def calculate_boxes(width, height):
    box_size = sys_config.get('box_size', 64)
    box_spacing = sys_config.get('box_spacing', 58)
    offset_x = sys_config.get('box_offset_x', 0)
    offset_y = sys_config.get('box_offset_y', 0)
    half_box = box_size // 2
     
    if sys_config.get('led_positions'):
        positions = sys_config['led_positions']
        boxes = []
        for (cx, cy) in positions:
            boxes.append((cx - half_box, cy - half_box, cx + half_box, cy + half_box))
        return boxes
     
    cy = (height // 2) + offset_y
    total_span = box_spacing * 6
    x_start = ((width - total_span) // 2) + offset_x
     
    boxes = []
    for i in range(7):
        center_x = x_start + (i * box_spacing)
        x1 = center_x - half_box
        y1 = cy - half_box
        x2 = center_x + half_box
        y2 = cy + half_box
        boxes.append((x1, y1, x2, y2))
    return boxes

def cleanup_qc_images():
    try:
        files = [os.path.join(QC_DIR, f) for f in os.listdir(QC_DIR) if f.endswith('.jpg')]
        files.sort(key=os.path.getmtime)
        while len(files) > QC_RETENTION_COUNT:
            os.remove(files[0])
            files.pop(0)
    except Exception as e:
        logger.error(f"QC Cleanup Error: {e}")

def draw_overlays(frame, box_data, power_data=None):
    if power_data:
        color = (0, 255, 0) if power_data['on'] else (0, 0, 255)
        x1, y1, x2, y2 = power_data['coords']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "Power", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, str(int(power_data['brightness'])), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
     
    for i, b in enumerate(box_data):
        color = (0, 255, 0) if b['on'] else (0, 0, 255)
        x1, y1, x2, y2 = b['coords']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = sys_config['sections'][i] if i < len(sys_config['sections']) else f"Unit {i+1}"
        short_label = (label[:8] + '..') if len(label) > 10 else label
        cv2.putText(frame, short_label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, str(int(b['brightness'])), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def handle_state_change(state_int, state_bin, power_on, frame_to_save, box_data, power_data):
    db_insert_state(state_int, power_on)
    power_str = "PWR:ON" if power_on else "PWR:OFF"
    logger.info(f"Logged State: {state_bin} ({state_int}) {power_str}")

    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    pwr_char = "P" if power_on else "p"
    qc_filename = f"{now_str}_{pwr_char}_{state_bin}.jpg"
    qc_path = os.path.join(QC_DIR, qc_filename)
     
    try:
        annotated_frame = frame_to_save.copy()
        draw_overlays(annotated_frame, box_data, power_data)
        cv2.imwrite(qc_path, annotated_frame)
        cleanup_qc_images()
    except Exception as e:
        logger.error(f"QC Image Save Error: {e}")

def analyze_frame(frame):
    h, w, _ = frame.shape
    boxes = calculate_boxes(w, h)
     
    power_box = boxes[0]
    section_boxes = boxes[1:7]
     
    x1, y1, x2, y2 = power_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
     
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        power_brightness = cv2.mean(gray)[0]
        power_on = power_brightness >= 150
    else:
        power_brightness, power_on = 0, False
     
    power_data = {'brightness': power_brightness, 'on': power_on, 'coords': (x1, y1, x2, y2)}
     
    state_int = 0
    box_results = []
     
    for i, (x1, y1, x2, y2) in enumerate(section_boxes):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
         
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            box_results.append({'brightness': 0, 'on': False, 'coords': (x1, y1, x2, y2)})
            continue
         
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray)[0]
        is_on = mean_brightness >= 150
         
        if is_on:
            state_int |= (1 << i)
         
        box_results.append({'brightness': mean_brightness, 'on': is_on, 'coords': (x1, y1, x2, y2)})
     
    return state_int, box_results, power_on, power_data

# --- Main Camera Loop ---

def camera_loop():
    global current_frame, latest_result
     
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
     
    time.sleep(2)
    previous_state, previous_power = -1, None
     
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera fail. Retrying...")
            time.sleep(1)
            continue
         
        with lock:
            current_frame = frame.copy()
         
        state_int, box_data, power_on, power_data = analyze_frame(frame)
        state_bin = "".join(["1" if (state_int >> i) & 1 else "0" for i in range(6)])
         
        latest_result = {
            "state_int": state_int,
            "state_bin": state_bin,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "boxes": box_data,
            "power_on": power_on,
            "power_brightness": power_data['brightness'],
            "power_coords": power_data['coords']
        }
         
        if state_int != previous_state or power_on != previous_power:
            handle_state_change(state_int, state_bin, power_on, frame, box_data, power_data)
            previous_state, previous_power = state_int, power_on
         
        time.sleep(FRAME_RATE)

# --- Web Server ---
app = Flask(__name__)

@app.route('/api/status')
def api_status():
    check_and_archive_previous_month()
    stats = get_monthly_stats()
     
    delta = datetime.now() - SERVER_START_TIME
    d = delta.days
    h, rem = divmod(delta.seconds, 3600)
    m, s = divmod(rem, 60)
     
    if d > 0: uptime_str = f"{d}d {h}h"
    elif h > 0: uptime_str = f"{h}h {m}m"
    else: uptime_str = f"{m}m {s}s"

    return jsonify({
        "current": latest_result,
        "stats": stats,
        "config": sys_config,
        "system": {
            "uptime": uptime_str,
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        },
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/historical_stats')
def api_historical_stats():
    year = request.args.get('year', type=int, default=datetime.now().year)
    month = request.args.get('month', type=int, default=datetime.now().month)
     
    now = datetime.now()
    is_current_month = (year == now.year and month == now.month)
     
    if not is_current_month:
        archived = db_get_archived_stats(year, month)
        if archived:
            archived['year'] = year
            archived['month'] = month
            archived['is_current'] = False
            return jsonify(archived)
     
    stats = calculate_month_stats(year, month)
    stats['year'] = year
    stats['month'] = month
    stats['is_current'] = is_current_month
    stats['is_archived'] = False
    return jsonify(stats)

@app.route('/api/available_periods')
def api_available_periods():
    years = db_get_available_years()
    archived = db_get_archived_months()
    return jsonify({
        "years": years,
        "archived_months": [{"year": y, "month": m} for y, m in archived],
        "current_year": datetime.now().year,
        "current_month": datetime.now().month
    })

@app.route('/api/alltime_stats')
def api_alltime_stats():
    stats = get_alltime_stats()
    return jsonify(stats)

@app.route('/api/save_settings', methods=['POST'])
def save_settings_route():
    global sys_config
    data = request.form
     
    new_config = {
        "device_id": data.get('device_id'),
        "location": data.get('location'),
        "sections": [data.get(f'section_{i}') for i in range(6)],
        "box_size": int(data.get('box_size', 64)),
        "box_spacing": int(data.get('box_spacing', 58)),
        "box_offset_x": int(data.get('box_offset_x', 0)),
        "box_offset_y": int(data.get('box_offset_y', 0)),
        "led_positions": sys_config.get('led_positions')
    }
    save_config(new_config)
    sys_config = new_config
    return redirect(url_for('dashboard'))

@app.route('/settings')
def settings_page():
    def get_sec(idx):
        try: return sys_config['sections'][idx]
        except: return f"Section {idx+1}"
     
    section_inputs = "".join([f'''
        <div class="form-group">
            <label>Section {i+1} Name</label>
            <input type="text" name="section_{i}" value="{get_sec(i)}" required>
        </div>
    ''' for i in range(6)])

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Settings | Sentinel</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {{ --bg: #000; --card: #111; --text: #e0e0e0; --accent: #00e676; }}
            body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 40px 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            h1 {{ font-weight: 300; letter-spacing: 2px; border-bottom: 1px solid #333; padding-bottom: 20px; }}
            h2 {{ font-weight: 300; letter-spacing: 1px; font-size: 1em; color: #888; margin: 0 0 15px 0; }}
            .card {{ background: var(--card); border: 1px solid #333; border-radius: 8px; padding: 24px; margin-bottom: 24px; }}
            label {{ display: block; margin-bottom: 8px; color: #888; font-size: 0.85rem; }}
            input[type="text"], input[type="number"] {{ width: 100%; padding: 10px; background: #222; border: 1px solid #444; color: #fff; border-radius: 4px; box-sizing: border-box; }}
            .input-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; }}
            .hint {{ font-size: 0.75rem; color: #555; margin-top: 4px; }}
            .btn {{ display: block; width: 100%; padding: 12px; background: var(--accent); color: #000; font-weight: bold; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; text-align: center; }}
            .tools-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }}
            .tool-item {{ background: #222; padding: 15px; border-radius: 4px; text-decoration: none; color: #ccc; border: 1px solid #333; text-align: center; }}
            .tool-item:hover {{ border-color: var(--accent); color: #fff; }}
            .preview-link {{ display: inline-block; margin-top: 10px; color: var(--accent); text-decoration: none; font-size: 0.85rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CONFIGURATION</h1>
            <form action="/api/save_settings" method="POST">
                <div class="card">
                    <h2>DEVICE INFO</h2>
                    <div class="input-row">
                        <div><label>Device ID</label><input type="text" name="device_id" value="{sys_config.get('device_id', '')}" required></div>
                        <div><label>Location</label><input type="text" name="location" value="{sys_config.get('location', '')}" required></div>
                    </div>
                </div>
                <div class="card">
                    <h2>DETECTION BOX CALIBRATION</h2>
                    <div class="input-row">
                        <div><label>Box Size (px)</label><input type="number" name="box_size" value="{sys_config.get('box_size', 64)}" min="20" max="200" required><div class="hint">Size of each detection square</div></div>
                        <div><label>LED Spacing (px)</label><input type="number" name="box_spacing" value="{sys_config.get('box_spacing', 58)}" min="20" max="200" required><div class="hint">Distance between LED centers</div></div>
                    </div>
                    <div class="input-row">
                        <div><label>X Offset (px)</label><input type="number" name="box_offset_x" value="{sys_config.get('box_offset_x', 0)}" min="-300" max="300" required><div class="hint">Shift left (-) or right (+)</div></div>
                        <div><label>Y Offset (px)</label><input type="number" name="box_offset_y" value="{sys_config.get('box_offset_y', 0)}" min="-300" max="300" required><div class="hint">Shift up (-) or down (+)</div></div>
                    </div>
                    <a href="/debug" class="preview-link" target="_blank">↗ Open Live Feed to preview box positions</a>
                </div>
                <div class="card">
                    <h2>HEATING ZONE NAMES</h2>
                    {section_inputs}
                </div>
                <button type="submit" class="btn">SAVE SETTINGS</button>
            </form>
            <div class="card">
                <div class="tools-grid">
                    <a href="/logs_pretty" class="tool-item">Activity Logs</a>
                    <a href="/debug" class="tool-item">Live Feed</a>
                    <a href="/qc" class="tool-item">QC Gallery</a>
                    <a href="/system" class="tool-item">JSON API</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

DASHBOARD_HTML = r'''
<!DOCTYPE html>
<html>
<head>
    <title>SENTINEL</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
    <style>
        :root { --bg: #050505; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: #e0e0e0; margin: 0; padding: 40px 20px; padding-bottom: 100px; min-height: 100vh; }
        .container { max-width: 1100px; margin: 0 auto; }
        h1, h2, h3 { font-weight: 300; letter-spacing: 4px; text-transform: uppercase; margin: 0; }
        .meta { font-family: monospace; color: #666; font-size: 0.9em; margin-top: 5px; }
        .header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 40px; }
        .time-chip { font-family: monospace; font-size: 1.4em; color: #00e676; border-bottom: 2px solid #00e676; padding-bottom: 5px; text-shadow: 0 0 10px rgba(0,230,118,0.4); }
        .glass-card { backdrop-filter: blur(5px); border-radius: 12px; padding: 25px; transition: transform 0.2s ease; margin-bottom: 20px; }
        .glass-card:hover { transform: translateY(-2px); }
        .tint-cyan { background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); }
        .tint-cyan h3 { color: #00e5ff; font-size: 0.8em; opacity: 0.8; }
        .tint-green { background: rgba(0, 230, 118, 0.05); border: 1px solid rgba(0, 230, 118, 0.2); }
        .tint-green .val { color: #00e676; text-shadow: 0 0 15px rgba(0, 230, 118, 0.3); }
        .tint-green h3 { color: #69f0ae; margin-bottom: 15px; font-size: 0.9em; }
        .tint-purple { background: rgba(224, 64, 251, 0.05); border: 1px solid rgba(224, 64, 251, 0.2); }
        .tint-purple .val { color: #e040fb; text-shadow: 0 0 15px rgba(224, 64, 251, 0.3); }
        .tint-purple h3 { color: #ea80fc; margin-bottom: 15px; font-size: 0.9em; }
        .tint-orange { background: rgba(255, 145, 0, 0.05); border: 1px solid rgba(255, 145, 0, 0.2); }
        .tint-orange .val { color: #ff9100; text-shadow: 0 0 15px rgba(255, 145, 0, 0.3); }
        .tint-orange h3 { color: #ffb74d; margin-bottom: 15px; font-size: 0.9em; }
        .tint-blue { background: rgba(41, 121, 255, 0.05); border: 1px solid rgba(41, 121, 255, 0.2); }
        .tint-blue h3 { color: #448aff; margin-bottom: 15px; font-size: 0.9em; }
        .tint-pink { background: rgba(255, 23, 68, 0.05); border: 1px solid rgba(255, 23, 68, 0.2); }
        .tint-pink h3 { color: #ff5252; margin-bottom: 15px; font-size: 0.9em; }
        .tint-teal { background: rgba(0, 150, 136, 0.05); border: 1px solid rgba(0, 150, 136, 0.2); }
        .tint-teal h3 { color: #26a69a; margin-bottom: 15px; font-size: 0.9em; }
        .tint-amber { background: rgba(255, 193, 7, 0.05); border: 1px solid rgba(255, 193, 7, 0.2); }
        .tint-amber h3 { color: #ffc107; margin-bottom: 15px; font-size: 0.9em; }
        .tint-amber .val { color: #ffc107; text-shadow: 0 0 15px rgba(255, 193, 7, 0.3); }
        .tint-indigo { background: rgba(63, 81, 181, 0.05); border: 1px solid rgba(63, 81, 181, 0.2); }
        .tint-indigo h3 { color: #7986cb; margin-bottom: 15px; font-size: 0.9em; }
        .tint-indigo .val { color: #7986cb; text-shadow: 0 0 15px rgba(63, 81, 181, 0.3); }
        .tint-lime { background: rgba(205, 220, 57, 0.05); border: 1px solid rgba(205, 220, 57, 0.2); }
        .tint-lime h3 { color: #cddc39; margin-bottom: 15px; font-size: 0.9em; }
        .tint-lime .val { color: #cddc39; text-shadow: 0 0 15px rgba(205, 220, 57, 0.3); }
        .tint-red { background: rgba(244, 67, 54, 0.05); border: 1px solid rgba(244, 67, 54, 0.2); }
        .tint-red h3 { color: #f44336; margin-bottom: 15px; font-size: 0.9em; }
        .tint-red .val { color: #f44336; text-shadow: 0 0 15px rgba(244, 67, 54, 0.3); }
        .viz-panel { text-align: center; }
        .bin-readout { font-family: monospace; font-size: 3em; letter-spacing: 10px; color: #00e5ff; margin-bottom: 15px; display: block; text-shadow: 0 0 20px rgba(0,229,255,0.4); }
        .led-row { display: flex; justify-content: center; gap: 20px; }
        .led { width: 15px; height: 15px; background: #333; border-radius: 50%; box-shadow: inset 0 0 5px #000; }
        .led.on { background: #00e5ff; box-shadow: 0 0 15px #00e5ff; }
        .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px; }
        .val { font-size: 2.2em; font-weight: 300; margin-top: 5px; }
        .val-sm { font-size: 1.6em; font-weight: 300; margin-top: 5px; }
        .label { font-size: 0.7em; letter-spacing: 2px; text-transform: uppercase; opacity: 0.7; }
        .split-view { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; }
        .split-inner { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        table { width: 100%; border-collapse: collapse; }
        td { padding: 10px 5px; border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 0.9em; }
        tr:last-child td { border: none; }
        .bar-bg { background: rgba(255,255,255,0.1); height: 4px; width: 100%; border-radius: 2px; }
        .bar-fill { height: 100%; border-radius: 2px; }
        .history-panel { margin-bottom: 20px; }
        .history-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px; }
        .history-title { display: flex; align-items: center; gap: 15px; }
        .period-badge { font-size: 0.7em; padding: 4px 10px; border-radius: 12px; background: rgba(0, 150, 136, 0.2); color: #4db6ac; letter-spacing: 1px; }
        .period-badge.current { background: rgba(0, 230, 118, 0.2); color: #69f0ae; }
        .period-badge.archived { background: rgba(224, 64, 251, 0.2); color: #ea80fc; }
        .year-select { background: #111; border: 1px solid #333; color: #fff; padding: 8px 30px 8px 12px; border-radius: 6px; font-size: 0.85em; cursor: pointer; appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%23888' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 10px center; }
        .year-select:hover, .year-select:focus { border-color: #26a69a; outline: none; }
        .month-row { display: flex; justify-content: space-between; background: #0a0a0a; border-radius: 8px; padding: 4px; margin-bottom: 20px; }
        .month-btn { flex: 1; padding: 10px 0; text-align: center; font-size: 0.75em; letter-spacing: 1px; color: #666; cursor: pointer; border-radius: 6px; transition: all 0.2s; text-transform: uppercase; }
        .month-btn:hover { color: #aaa; background: rgba(255,255,255,0.03); }
        .month-btn.active { background: #26a69a; color: #000; font-weight: 600; }
        .month-btn.disabled { color: #333; cursor: not-allowed; }
        .month-btn.disabled:hover { background: transparent; color: #333; }
        .history-stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 20px; }
        .history-stat { text-align: center; padding: 15px 10px; background: rgba(0,0,0,0.3); border-radius: 8px; }
        .history-stat .section-name { font-size: 0.7em; color: #666; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .history-stat .hours { font-size: 1.4em; font-weight: 300; margin-bottom: 4px; }
        .history-stat .share { font-size: 0.9em; font-weight: 600; }
        .history-total { text-align: center; margin-bottom: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px; font-size: 0.9em; color: #888; }
        .history-total strong { color: #26a69a; font-size: 1.2em; }
        .monthly-details { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 20px; }
        .monthly-details h4 { color: #4db6ac; font-size: 0.75em; letter-spacing: 2px; margin: 0 0 15px 0; font-weight: 400; text-transform: uppercase; }
        .section-title { font-size: 0.85em; letter-spacing: 3px; color: #666; margin: 30px 0 20px 0; text-transform: uppercase; border-bottom: 1px solid #222; padding-bottom: 10px; }
        .alltime-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .alltime-sections { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
        .alltime-section { text-align: center; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 8px; }
        .alltime-section .name { font-size: 0.7em; color: #666; margin-bottom: 8px; }
        .alltime-section .hrs { font-size: 1.2em; font-weight: 300; }
        .alltime-section .pct { font-size: 0.8em; opacity: 0.8; }
        .dock { position: fixed; bottom: 30px; width: 100%; display: flex; justify-content: center; gap: 15px; pointer-events: none; z-index: 100; }
        .dock-btn { pointer-events: auto; background: rgba(0,0,0,0.8); border: 1px solid #333; color: #888; padding: 10px 30px; border-radius: 50px; text-decoration: none; font-size: 0.8em; letter-spacing: 1px; transition: all 0.3s; text-transform: uppercase; }
        .dock-btn:hover { border-color: #00e676; color: #fff; box-shadow: 0 0 15px rgba(0,230,118,0.2); }
        .power-chip { pointer-events: auto; background: rgba(0,0,0,0.9); border: 1px solid #333; padding: 8px 20px; border-radius: 50px; font-size: 0.75em; letter-spacing: 1px; text-transform: uppercase; display: flex; align-items: center; gap: 8px; transition: all 0.3s; }
        .power-chip.on { border-color: #00e676; color: #00e676; box-shadow: 0 0 10px rgba(0,230,118,0.2); }
        .power-chip.off { border-color: #ff1744; color: #ff1744; box-shadow: 0 0 10px rgba(255,23,68,0.2); }
        .power-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }
        @media (max-width: 800px) { 
            .metrics-grid, .split-view, .monthly-details { grid-template-columns: 1fr; } 
            .history-stats, .alltime-sections { grid-template-columns: repeat(3, 1fr); } 
            .alltime-grid { grid-template-columns: repeat(2, 1fr); }
            .month-btn { font-size: 0.65em; padding: 8px 0; } 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div><h1>Sentinel</h1><div class="meta" id="sysInfo">INITIALIZING SYSTEM...</div></div>
            <div class="time-chip" id="ts">--:--:--</div>
        </div>
        
        <div class="glass-card tint-cyan viz-panel">
            <h3>OPTICAL ARRAY STATUS</h3>
            <span class="bin-readout" id="binStr">------</span>
            <div class="led-row" id="ledRow"></div>
        </div>
        
        <div class="metrics-grid">
            <div class="glass-card tint-green"><div class="label">System Uptime</div><div class="val" id="serverUptime">--</div></div>
            <div class="glass-card tint-purple"><div class="label">Logged Hours (This Month)</div><div class="val" id="totalHrs">--</div></div>
            <div class="glass-card tint-orange"><div class="label">CPU Load</div><div class="val" id="cpuLoad">--%</div></div>
        </div>
        
        <div class="glass-card tint-teal">
            <div class="history-header">
                <div class="history-title"><h3>MONTHLY BILLING</h3><span class="period-badge current" id="periodBadge">MONTH TO DATE</span></div>
                <select class="year-select" id="yearSelect"></select>
            </div>
            <div class="month-row" id="monthRow"></div>
            <div class="history-stats" id="historyStats"></div>
            <div class="history-total" id="historyTotal"></div>
            <div class="monthly-details">
                <div>
                    <h4>Section Activity</h4>
                    <table id="uptimeTable"></table>
                </div>
                <div>
                    <h4>Bill Distribution</h4>
                    <div style="position: relative; height: 220px;"><canvas id="shareChart"></canvas></div>
                </div>
            </div>
        </div>
        
        <div class="section-title">All-Time Statistics</div>
        
        <div class="alltime-grid">
            <div class="glass-card tint-amber">
                <div class="label">Total Heating Hours</div>
                <div class="val-sm" id="alltimeHours">--</div>
            </div>
            <div class="glass-card tint-indigo">
                <div class="label">Days Tracked</div>
                <div class="val-sm" id="alltimeDays">--</div>
            </div>
            <div class="glass-card tint-lime">
                <div class="label">Avg Daily Hours</div>
                <div class="val-sm" id="alltimeAvg">--</div>
            </div>
            <div class="glass-card tint-red">
                <div class="label">Total Records</div>
                <div class="val-sm" id="alltimeRecords">--</div>
            </div>
        </div>
        
        <div class="split-view">
            <div class="glass-card tint-purple">
                <h3>All-Time Section Totals</h3>
                <div class="alltime-sections" id="alltimeSections"></div>
            </div>
            <div class="glass-card tint-pink">
                <h3>All-Time Distribution</h3>
                <div style="position: relative; height: 220px;"><canvas id="alltimeChart"></canvas></div>
            </div>
        </div>
        
        <div class="glass-card tint-green" style="margin-top: 20px;">
            <h3>Tracking Period</h3>
            <div class="split-inner" style="margin-top: 15px;">
                <div><div class="label">First Record</div><div style="font-family: monospace; margin-top: 5px;" id="firstRecord">--</div></div>
                <div><div class="label">System Depth</div><div style="font-family: monospace; margin-top: 5px;" id="systemDepth">--</div></div>
            </div>
        </div>
    </div>
    
    <div class="dock">
        <div class="power-chip off" id="powerChip"><span class="power-dot"></span><span id="powerLabel">Heating System Power</span></div>
        <a href="/settings" class="dock-btn">Configure System</a>
    </div>
    
    <script>
        Chart.register(ChartDataLabels);
        const THEME_COLORS = ['#00e5ff', '#00e676', '#e040fb', '#ff9100', '#2979ff', '#ff1744'];
        const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        let selectedYear = new Date().getFullYear();
        let selectedMonth = new Date().getMonth() + 1;
        let availablePeriods = { years: [], archived_months: [], current_year: 0, current_month: 0 };
        let sectionNames = [];

        const chartOptions = {
            responsive: true, 
            maintainAspectRatio: false, 
            cutout: '55%', 
            plugins: { 
                legend: { display: false }, 
                datalabels: { 
                    color: '#fff', 
                    font: { size: 10, weight: 'bold' }, 
                    formatter: (v) => v < 3 ? '' : v.toFixed(1) + '%', 
                    anchor: 'end', 
                    align: 'start', 
                    offset: -5 
                } 
            }
        };

        const shareChart = new Chart(document.getElementById('shareChart').getContext('2d'), {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: THEME_COLORS, borderColor: '#000', borderWidth: 2, hoverOffset: 5 }] },
            options: chartOptions
        });
        
        const alltimeChart = new Chart(document.getElementById('alltimeChart').getContext('2d'), {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: THEME_COLORS, borderColor: '#000', borderWidth: 2, hoverOffset: 5 }] },
            options: chartOptions
        });

        function updateClock() { document.getElementById('ts').innerText = new Date().toTimeString().split(' ')[0]; }
        setInterval(updateClock, 1000); updateClock();

        function isMonthArchived(y, m) { return availablePeriods.archived_months.some(a => a.year === y && a.month === m); }
        function isCurrentMonth(y, m) { return y === availablePeriods.current_year && m === availablePeriods.current_month; }
        function hasDataForMonth(y, m) { if (isCurrentMonth(y, m)) return true; if (new Date(y, m - 1, 1) > new Date()) return false; return isMonthArchived(y, m); }

        function renderMonthRow() {
            const c = document.getElementById('monthRow');
            c.innerHTML = MONTH_NAMES.map((n, i) => {
                const m = i + 1, isA = m === selectedMonth, hasD = hasDataForMonth(selectedYear, m), isFut = new Date(selectedYear, m - 1, 1) > new Date();
                let cls = 'month-btn'; if (isA) cls += ' active'; if (!hasD || isFut) cls += ' disabled';
                return `<div class="${cls}" data-month="${m}">${n}</div>`;
            }).join('');
            c.querySelectorAll('.month-btn:not(.disabled)').forEach(b => b.addEventListener('click', () => { selectedMonth = parseInt(b.dataset.month); renderMonthRow(); loadHistoricalStats(); }));
        }

        function renderYearSelect() {
            const s = document.getElementById('yearSelect');
            s.innerHTML = availablePeriods.years.map(y => `<option value="${y}" ${y === selectedYear ? 'selected' : ''}>${y}</option>`).join('');
            s.addEventListener('change', () => { selectedYear = parseInt(s.value); selectedMonth = selectedYear < availablePeriods.current_year ? 12 : availablePeriods.current_month; renderMonthRow(); loadHistoricalStats(); });
        }

        function loadHistoricalStats() {
            fetch(`/api/historical_stats?year=${selectedYear}&month=${selectedMonth}`).then(r => r.json()).then(d => {
                const b = document.getElementById('periodBadge');
                b.className = 'period-badge ' + (d.is_current ? 'current' : d.is_archived ? 'archived' : '');
                b.innerText = d.is_current ? 'MONTH TO DATE' : d.is_archived ? 'ARCHIVED' : 'HISTORICAL';
                
                document.getElementById('historyStats').innerHTML = (d.section_hours || []).map((h, i) => 
                    `<div class="history-stat"><div class="section-name">${sectionNames[i] || 'Section '+(i+1)}</div><div class="hours" style="color:${THEME_COLORS[i]}">${h.toFixed(1)}h</div><div class="share" style="color:${THEME_COLORS[i]}">${(d.relative_share[i]||0).toFixed(1)}%</div></div>`
                ).join('');
                
                document.getElementById('historyTotal').innerHTML = `Total Heating Time: <strong>${(d.total_duration_hours||0).toFixed(1)} hours</strong>`;
                
                // Update the synced table and chart
                document.getElementById('uptimeTable').innerHTML = (d.uptimes || d.relative_share || []).map((v, i) => 
                    `<tr><td>${sectionNames[i] || 'Section '+(i+1)}</td><td style="font-family:monospace;color:#888;font-size:0.8em;">${(d.section_hours[i]||0).toFixed(1)}h</td><td width="30%"><div class="bar-bg"><div class="bar-fill" style="width:${d.relative_share[i]||0}%;background:${THEME_COLORS[i]};box-shadow:0 0 5px ${THEME_COLORS[i]}"></div></div></td><td align="right" style="font-family:monospace;color:${THEME_COLORS[i]}">${(d.relative_share[i]||0).toFixed(1)}%</td></tr>`
                ).join('');
                
                shareChart.data.labels = sectionNames;
                shareChart.data.datasets[0].data = d.relative_share || [];
                shareChart.update();
            });
        }
        
        function loadAlltimeStats() {
            fetch('/api/alltime_stats').then(r => r.json()).then(d => {
                document.getElementById('alltimeHours').innerText = (d.total_duration_hours || 0).toFixed(1);
                document.getElementById('alltimeDays').innerText = (d.days_tracked || 0).toFixed(2);
                document.getElementById('alltimeAvg').innerText = (d.avg_daily_hours || 0).toFixed(2);
                document.getElementById('alltimeRecords').innerText = (d.total_records || 0).toLocaleString();
                document.getElementById('firstRecord').innerText = d.first_log || '--';
                
                // Calculate real-time system depth
                if (d.first_log) {
                    const firstDate = new Date(d.first_log.replace(' ', 'T'));
                    const updateDepth = () => {
                        const now = new Date();
                        const diffMs = now - firstDate;
                        const diffDays = diffMs / (1000 * 60 * 60 * 24);
                        if (diffDays < 1) {
                            const hours = diffMs / (1000 * 60 * 60);
                            document.getElementById('systemDepth').innerText = hours.toFixed(2) + ' hours';
                        } else {
                            document.getElementById('systemDepth').innerText = diffDays.toFixed(4) + ' days';
                        }
                        document.getElementById('alltimeDays').innerText = diffDays.toFixed(2);
                    };
                    updateDepth();
                    if (!window.depthInterval) {
                        window.depthInterval = setInterval(updateDepth, 1000);
                    }
                } else {
                    document.getElementById('systemDepth').innerText = '--';
                }
                
                document.getElementById('alltimeSections').innerHTML = (d.section_hours || []).map((h, i) => 
                    `<div class="alltime-section"><div class="name">${sectionNames[i] || 'Section '+(i+1)}</div><div class="hrs" style="color:${THEME_COLORS[i]}">${h.toFixed(1)}h</div><div class="pct" style="color:${THEME_COLORS[i]}">${(d.relative_share[i]||0).toFixed(1)}%</div></div>`
                ).join('');
                
                alltimeChart.data.labels = sectionNames;
                alltimeChart.data.datasets[0].data = d.relative_share || [];
                alltimeChart.update();
            });
        }

        function loadAvailablePeriods() { 
            fetch('/api/available_periods').then(r => r.json()).then(d => { 
                availablePeriods = d; 
                selectedYear = d.current_year; 
                selectedMonth = d.current_month; 
                renderYearSelect(); 
                renderMonthRow(); 
                loadHistoricalStats(); 
            }); 
        }

        function update() {
            fetch('/api/status').then(r => r.json()).then(d => {
                const cur = d.current, stats = d.stats, cfg = d.config;
                sectionNames = cfg.sections;
                document.getElementById('sysInfo').innerText = `ID: ${cfg.device_id} // LOC: ${cfg.location}`;
                document.getElementById('binStr').innerText = cur.state_bin;
                document.getElementById('serverUptime').innerText = d.system.uptime;
                document.getElementById('totalHrs').innerText = stats.total_duration_hours.toFixed(1);
                document.getElementById('cpuLoad').innerText = d.system.cpu + "%";
                const pc = document.getElementById('powerChip'), pl = document.getElementById('powerLabel');
                pc.className = 'power-chip ' + (cur.power_on ? 'on' : 'off');
                pl.innerText = 'Heating System Power: ' + (cur.power_on ? 'ON' : 'OFF');
                const lc = document.getElementById('ledRow'); lc.innerHTML = '';
                for (let c of cur.state_bin) { const l = document.createElement('div'); l.className = 'led ' + (c === '1' ? 'on' : ''); lc.appendChild(l); }
            });
        }
        
        loadAvailablePeriods(); 
        loadAlltimeStats();
        update(); 
        setInterval(update, 1000);
        setInterval(loadAlltimeStats, 30000); // Refresh all-time stats every 30s
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    return DASHBOARD_HTML

@app.route('/logs_pretty')
def logs_pretty():
    def get_col_header(i):
        try: return sys_config['sections'][i]
        except: return f"LED {i+1}"

    rows = db_get_recent_logs(500)
    rows_html = ""
    for row in rows:
        ts, val = row[0], int(row[1])
        power = int(row[2]) if len(row) > 2 else 1
        pwr_style = "color:#00e676;" if power else "color:#ff1744;"
        pwr_text = "ON" if power else "OFF"
        cols = f"<td style='text-align:center;{pwr_style}'>{pwr_text}</td>"
        for i in range(6):
            is_on = (val >> i) & 1
            style = "color:#00e676;font-weight:bold;" if is_on else "color:#444;"
            cols += f"<td style='text-align:center;{style}'>{is_on}</td>"
        rows_html += f"<tr><td>{ts}</td>{cols}</tr>"

    headers_html = "<th>Power</th>" + "".join([f"<th>{get_col_header(i)}</th>" for i in range(6)])
    return f'''<!DOCTYPE html><html><head><title>Detailed Logs</title><style>body{{font-family:monospace;background:#000;color:#ccc;padding:20px}}h2{{color:#eee;font-family:'Segoe UI';font-weight:300;letter-spacing:2px}}table{{border-collapse:collapse;width:100%;max-width:1100px;margin-top:20px;border:1px solid #333}}th,td{{border:1px solid #333;padding:8px;font-size:0.9em}}th{{background:#111;color:#00e676;text-align:left}}tr:nth-child(even){{background:#0a0a0a}}.btn{{padding:10px 20px;background:#00e676;color:#000;text-decoration:none;border-radius:4px;font-weight:bold;font-family:sans-serif}}</style></head><body><div style="display:flex;justify-content:space-between;align-items:center;"><h2>Recent Activity (Last 500)</h2><a href="/download_csv" class="btn">Download CSV</a></div><table><tr><th>Timestamp</th>{headers_html}</tr>{rows_html}</table></body></html>'''

@app.route('/download_csv')
def download_csv():
    month = datetime.now().strftime("%Y-%m")
    rows = db_get_month_data(month)
    si = io.StringIO()
    cw = csv.writer(si)
    stats = get_monthly_stats()
    headers = ['Timestamp', 'State_Int', 'Power'] + sys_config['sections']
    cw.writerow(headers)
    for row in rows:
        ts, val = row[0], int(row[1])
        power = int(row[2]) if len(row) > 2 else 1
        bits = [(val >> i) & 1 for i in range(6)]
        cw.writerow([ts, val, power] + bits)
    cw.writerow([])
    cw.writerow(['MONTHLY SUMMARY', '', ''] + [f'{h:.2f} hrs' for h in stats['section_hours']])
    cw.writerow(['BILL SHARE %', '', ''] + [f'{s:.1f}%' for s in stats['relative_share']])
    return Response(si.getvalue(), mimetype="text/csv", headers={"Content-disposition": f"attachment; filename={month}_heating_log.csv"})

@app.route('/debug')
def debug_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        with lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            vis = current_frame.copy()
            power_data = None
            if latest_result.get('power_coords'):
                power_data = {'on': latest_result['power_on'], 'brightness': latest_result['power_brightness'], 'coords': latest_result['power_coords']}
            vis = draw_overlays(vis, latest_result['boxes'], power_data)
            ret, buf = cv2.imencode('.jpg', vis)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.route('/qc')
def qc_browser():
    imgs = sorted([f for f in os.listdir(QC_DIR) if f.endswith('.jpg')], reverse=True)
    return "".join([f"<a href='/qc_img?f={i}'><img src='/qc_img?f={i}' width=300 style='margin:5px;border:2px solid #333;'></a>" for i in imgs])

@app.route('/qc_img')
def serve_img():
    f = request.args.get('f', '')
    if '..' in f or '/' in f: return "Bad filename", 400
    p = os.path.join(QC_DIR, f)
    if os.path.exists(p):
        with open(p, 'rb') as x: return Response(x.read(), mimetype='image/jpeg')
    return "Not found", 404

@app.route('/system')
def health():
    return jsonify({"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent, "disk": psutil.disk_usage('/').percent})

if __name__ == '__main__':
    init_db()
    os.makedirs(QC_DIR, exist_ok=True)
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)
