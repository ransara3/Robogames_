#!/usr/bin/env python3
"""
PHASE 2: DRONE AIRPORT NAVIGATION
===================================
Task: Navigate the drone following yellow guiding lines (CurveStrips),
      detect AprilTags on landing pads to identify airports and landing status,
      visit specified airports in the Airport list, and land on the correct pad.

World layout (from iris_Task_2.wbt):
  - Pad 1 (HOME):  (-3.18, 2.84) — no texture, drone start position
  - Pad 2:         ( 0.19, 1.64) — texture 103.png
  - Pad 3:         ( 2.84, 1.67) — texture 203.png
  - Pad 4:         (-2.96,-0.31) — texture 112.png
  - Pad 5:         (-1.31,-2.50) — texture 213.png
  - Pad 6:         ( 2.82,-2.49) — texture 302.png

Yellow paths: CurveStrips (curved yellow lines) and straight Strips
connecting pads in the arena.

The drone must:
  1. Take off from Pad 1
  2. Follow the yellow guiding line network
  3. At each pad, read the AprilTag to identify airport ID
  4. Visit the airports listed in the Airport variable
  5. Land on the designated pad

Texture naming: ABC.png -> first digit = airport number, last two = status code
"""

import cv2
import numpy as np
import socket
import struct
import threading
import time
import math
from collections import deque
from pymavlink import mavutil

# AprilTag library
try:
    from pupil_apriltags import Detector as AprilTagDetector
    at_detector = AprilTagDetector(
        families="tag36h11", nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.8,
        refine_edges=1,
        decode_sharpening=0.25)
    print("AprilTag: using pupil-apriltags")
except ImportError:
    try:
        import apriltag
        at_detector = apriltag.Detector(
            apriltag.DetectorOptions(families="tag36h11"))
        print("AprilTag: using apriltag")
    except ImportError:
        at_detector = None
        print("WARNING: No AprilTag library. pip install pupil-apriltags")

# ================================================================
# AIRPORT LIST — Evaluators will change this when testing
# ================================================================
Airport = [1, 2]

# ================================================================
# CONFIG
# ================================================================
TARGET_ALT = 2.5

# Yellow line HSV — camera streams grayscale, so yellow appears as
# bright white after GRAY2BGR conversion (H≈0, S≈0, V=high).
# Match your Phase 1 values that worked.
YELLOW_H_MIN = 0
YELLOW_H_MAX = 38
YELLOW_S_MIN = 0
YELLOW_S_MAX = 82
YELLOW_V_MIN = 153
YELLOW_V_MAX = 255

# Line detection
LINE_ROI_START       = 0.55
LINE_MIN_AREA        = 400

# Narrow center strip to filter side lines at junctions
# Only contours whose centroid falls within this horizontal band are considered
LINE_CENTER_MARGIN   = 0.35   # fraction of width from center (0.35 = middle 70%)

# PD controller for line following
LINE_KP              = 0.002
LINE_KD              = 0.001
MAX_LATERAL          = 0.25

# Speeds (m/s)
LINE_SPEED           = 0.15
LINE_SPEED_SLOW      = 0.05
FWD_SPEED            = 0.15
SEARCH_SPEED         = 0.10

# Pad clearing — longer duration to get past line junctions
CLEAR_SPEED          = 0.15
CLEAR_DURATION       = 4.0

# Line lost
LINE_LOST_TIMEOUT    = 30.0
LINE_MIN_FOLLOW_TIME = 2.0

# Black pad detection
PAD_V_MAX              = 60
PAD_ROI_TOP            = 0.20
PAD_ROI_BOT            = 0.80
PAD_ROI_LEFT           = 0.15
PAD_ROI_RIGHT          = 0.85
PAD_FILL_RATIO         = 0.18
PAD_SLOWDOWN_THRESHOLD = 0.12

# AprilTag voting
VOTE_WINDOW      = 25
VOTE_THRESHOLD   = 0.60
APRILTAG_TIMEOUT = 25

# Turn
TURN_SPEED   = 30
TURN_TIMEOUT = 15
TURN_MARGIN  = 5

# Search / exploration
SEARCH_TIMEOUT  = 60
HOVER_TIME      = 1.0

CAM_HOST    = "127.0.0.1"
CAM_PORT    = 5599
MAVLINK_URI = "udp:0.0.0.0:14550"

# ================================================================
# CAMERA (COLOR: 3 bytes per pixel in Phase 2)
# ================================================================
print(f"Connecting to Webots camera at {CAM_HOST}:{CAM_PORT}...")
cam_socket = None
while cam_socket is None:
    try:
        cam_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cam_socket.connect((CAM_HOST, CAM_PORT))
        print("Camera connected!")
    except ConnectionRefusedError:
        print("  Not ready - retrying in 2s...")
        cam_socket = None
        time.sleep(2)

frame       = None
frame_lock  = threading.Lock()
running     = True
header_size = struct.calcsize("=HH")

def camera_thread():
    """Read grayscale frames from TCP stream: 4-byte header + width*height payload."""
    global frame, running
    while running:
        try:
            # Read header
            header = b""
            while len(header) < header_size:
                chunk = cam_socket.recv(header_size - len(header))
                if not chunk:
                    running = False
                    return
                header += chunk
            width, height = struct.unpack("=HH", header)

            # Read pixel payload (1 byte per pixel — grayscale stream)
            payload_size = width * height
            img_bytes = b""
            while len(img_bytes) < payload_size:
                chunk = cam_socket.recv(min(payload_size - len(img_bytes), 65536))
                if not chunk:
                    running = False
                    return
                img_bytes += chunk

            img = np.frombuffer(img_bytes, np.uint8).reshape((height, width))
            # Convert grayscale to BGR so HSV/color detection still works
            with frame_lock:
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            if running:
                print(f"\nCamera error: {e}")
            running = False
            return

threading.Thread(target=camera_thread, daemon=True).start()
print("Waiting for first frame...")
deadline = time.time() + 15
while True:
    with frame_lock:
        has_frame = frame is not None
    if has_frame:
        break
    if time.time() > deadline:
        print("ERROR: No frames after 15s. Is Webots running?")
        running = False
        exit(1)
    time.sleep(0.1)
print("Camera live!\n")

def get_frame():
    with frame_lock:
        return frame.copy() if frame is not None else None

# ================================================================
# MAVLINK
# ================================================================
print("Connecting to ArduPilot SITL...")
master = mavutil.mavlink_connection(MAVLINK_URI)
master.wait_heartbeat()
print(f"Connected - sys={master.target_system} comp={master.target_component}\n")

# ================================================================
# FLIGHT HELPERS
# ================================================================
def set_mode(mode_name):
    print(f"Setting mode: {mode_name}")
    mode_id = master.mode_mapping()[mode_name]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id)
    time.sleep(1)

def arm_drone():
    print("Arming...")
    master.arducopter_arm()
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)
    time.sleep(2)
    print("Armed!")

def takeoff(altitude):
    print(f"Taking off to {altitude}m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude)
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            alt = msg.relative_alt / 1000.0
            print(f"\r  Altitude: {alt:.2f}m / {altitude}m", end="", flush=True)
            if alt >= altitude * 0.95:
                print(f"\nReached {alt:.2f}m!")
                break
        time.sleep(0.05)

def send_velocity(vx, vy, vz=0):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0, 0, 0)

def stop_drone():
    for _ in range(10):
        send_velocity(0, 0, 0)
        time.sleep(0.05)

def get_heading():
    """Get current heading in degrees (0-360)."""
    msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=3)
    if msg:
        return msg.heading
    return None

def get_altitude():
    """Get current relative altitude in meters."""
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
    if msg:
        return msg.relative_alt / 1000.0
    return None

def show_frame(status=""):
    img = get_frame()
    if img is None:
        return
    cv2.putText(img, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Drone Mission", img)
    cv2.waitKey(1)

def fly_forward_timed(speed, duration, status="FLYING FORWARD"):
    print(f"{status} ({duration}s @ {speed}m/s)")
    start = time.time()
    while time.time() - start < duration:
        send_velocity(speed, 0, 0)
        elapsed = time.time() - start
        show_frame(f"{status} | {elapsed:.1f}s / {duration}s")
        time.sleep(0.05)
    stop_drone()
    time.sleep(0.5)

def turn_to_heading(target_heading):
    """Turn to an absolute heading (0-360)."""
    current = get_heading()
    if current is None:
        print("  No heading data - using timed turn")
        time.sleep(3)
        return

    # Calculate shortest turn direction
    diff = (target_heading - current) % 360
    if diff > 180:
        diff -= 360

    direction = 1 if diff >= 0 else -1  # 1=CW, -1=CCW
    degrees = abs(diff)

    print(f"  Turning {degrees:.0f}° {'CW' if direction == 1 else 'CCW'} "
          f"from {current}° to {target_heading}°")

    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0, degrees, TURN_SPEED, direction, 0, 0, 0, 0)

    timeout = time.time() + TURN_TIMEOUT
    while time.time() < timeout:
        m = master.recv_match(type='VFR_HUD', blocking=False)
        if m:
            d = abs(m.heading - target_heading)
            if d > 180:
                d = 360 - d
            show_frame(f"TURNING | {m.heading}° -> {target_heading}° diff={d:.1f}°")
            if d < TURN_MARGIN:
                print(f"\n  Turn complete at {m.heading}°")
                break
        time.sleep(0.05)
    time.sleep(0.5)

def turn_degrees(degrees, clockwise=True):
    """Turn by a relative number of degrees."""
    current = get_heading()
    if current is None:
        print("  No heading data")
        return
    if clockwise:
        target = (current + degrees) % 360
    else:
        target = (current - degrees) % 360
    turn_to_heading(target)

def land_and_disarm():
    print("\nLanding...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            alt = msg.relative_alt / 1000.0
            show_frame(f"LANDING | {alt:.2f}m")
            print(f"\r  Altitude: {alt:.2f}m", end="", flush=True)
            if alt < 0.15:
                print("\nLanded!")
                break
        time.sleep(0.05)
    print("Disarming...")
    master.arducopter_disarm()
    time.sleep(1)
    print("Disarmed!")

# ================================================================
# CV — LINE + PAD DETECTION (color-aware for Phase 2)
# ================================================================
def detect_line_and_pad(img):
    """
    Detect yellow line and black landing pad in a color image.
    Returns: (error, line_found, end_of_line, pad_ratio, display)
    """
    h, w = img.shape[:2]
    display = img.copy()

    # ---- Black pad detection (centre ROI) ----
    py1 = int(h * PAD_ROI_TOP);  py2 = int(h * PAD_ROI_BOT)
    px1 = int(w * PAD_ROI_LEFT); px2 = int(w * PAD_ROI_RIGHT)
    pad_roi  = img[py1:py2, px1:px2]
    pad_hsv  = cv2.cvtColor(pad_roi, cv2.COLOR_BGR2HSV)
    pad_mask = cv2.inRange(pad_hsv,
                           np.array([0, 0, 0]),
                           np.array([179, 255, PAD_V_MAX]))
    pad_ratio   = np.sum(pad_mask > 0) / pad_mask.size
    end_of_line = pad_ratio >= PAD_FILL_RATIO

    if end_of_line:
        box_color = (0, 0, 255)
    elif pad_ratio > PAD_SLOWDOWN_THRESHOLD:
        box_color = (0, 165, 255)
    else:
        box_color = (100, 100, 100)

    cv2.rectangle(display, (px1, py1), (px2, py2), box_color, 2)
    cv2.putText(display, f"pad={pad_ratio:.2f}",
                (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # ---- Yellow line detection (bottom ROI) ----
    roi_y = int(h * LINE_ROI_START)
    roi   = img[roi_y:h, :]
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    raw_mask = cv2.inRange(hsv,
                           np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN]),
                           np.array([YELLOW_H_MAX, YELLOW_S_MAX, YELLOW_V_MAX]))

    k          = np.ones((7, 7), np.uint8)
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN,  k)

    cv2.line(display, (0, roi_y), (w, roi_y), (0, 255, 255), 2)

    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, False, end_of_line, pad_ratio, display

    # --- SMART LINE SELECTION ---
    # At junctions near pads, multiple lines are visible.
    # Strategy: prefer the contour whose centroid is closest to the
    # bottom-center of the ROI (i.e. the line directly ahead).
    roi_h = h - roi_y
    center_x = w // 2
    margin_px = int(w * LINE_CENTER_MARGIN)
    left_bound = center_x - margin_px
    right_bound = center_x + margin_px

    best_contour = None
    best_score = float('inf')  # lower = better (distance to bottom-center)
    best_cx = 0
    best_cy = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < LINE_MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_c = int(M["m10"] / M["m00"])
        cy_c = int(M["m01"] / M["m00"])

        # Skip contours far off to the side (not ahead)
        if cx_c < left_bound or cx_c > right_bound:
            # Draw rejected contours in red for debug
            display_roi = display[roi_y:h, :]
            cv2.drawContours(display_roi, [cnt], -1, (0, 0, 180), 1)
            continue

        # Score: distance from bottom-center of ROI
        # Prefer bottom (large cy) and center (small dx)
        dx = abs(cx_c - center_x)
        dy = roi_h - cy_c  # lower = closer to bottom
        score = dx + dy * 0.5  # weight bottom-ness
        if score < best_score:
            best_score = score
            best_contour = cnt
            best_cx = cx_c
            best_cy = cy_c

    if best_contour is None:
        return 0, False, end_of_line, pad_ratio, display

    error = best_cx - center_x

    display_roi = display[roi_y:h, :]
    cv2.drawContours(display_roi, [best_contour], -1, (0, 255, 0), 2)
    cv2.circle(display_roi, (best_cx, best_cy), 10, (0, 0, 255), -1)
    cv2.line(display, (center_x, roi_y), (center_x, h), (255, 0, 0), 2)
    # Draw center margin bounds for debug
    cv2.line(display, (left_bound, roi_y), (left_bound, h), (255, 255, 0), 1)
    cv2.line(display, (right_bound, roi_y), (right_bound, h), (255, 255, 0), 1)

    return error, True, end_of_line, pad_ratio, display

def detect_yellow_line_direction(img):
    """
    Detect where the yellow line is in the full frame.
    Returns angle in degrees from center (used for finding line after turns).
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN]),
                       np.array([YELLOW_H_MAX, YELLOW_S_MAX, YELLOW_V_MAX]))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200:
        return None, 0

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, 0

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Angle from image center
    dx = cx - w // 2
    dy = cy - h // 2
    angle = math.degrees(math.atan2(dx, -dy))  # 0=forward, +ve=right
    return angle, area

# ================================================================
# APRILTAG — VOTING CONFIRMATION
# ================================================================
def read_apriltag(pad_label="PAD"):
    """Read and confirm AprilTag ID using voting window."""
    if at_detector is None:
        print("No AprilTag library - skipping")
        return None

    print(f"\nReading AprilTag on {pad_label}...")
    vote_window  = deque(maxlen=VOTE_WINDOW)
    deadline     = time.time() + APRILTAG_TIMEOUT
    confirmed_id = None

    while time.time() < deadline:
        img = get_frame()
        if img is None:
            time.sleep(0.03)
            continue

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        try:
            detections = at_detector.detect(gray)
        except Exception as e:
            print(f"AprilTag error: {e}")
            detections = []

        ann = img.copy()

        if detections:
            best   = max(detections,
                         key=lambda d: d.decision_margin
                         if hasattr(d, 'decision_margin') else 1)
            tid    = best.tag_id if hasattr(best, 'tag_id') else best.id
            pts    = best.corners.astype(int)
            cx_tag = int(np.mean(pts[:, 0]))
            cy_tag = int(np.mean(pts[:, 1]))
            vote_window.append(tid)
            cv2.polylines(ann, [pts], True, (0, 255, 0), 3)
            cv2.putText(ann, f"ID:{tid}",
                        (cx_tag - 30, cy_tag),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            vote_window.append(None)

        total  = len(vote_window)
        counts = {}
        for v in vote_window:
            if v is not None:
                counts[v] = counts.get(v, 0) + 1

        best_count   = 0
        confirmed_id = None
        for tid, count in counts.items():
            if count > best_count:
                best_count   = count
                confirmed_id = tid

        vote_ratio = best_count / total if total > 0 else 0

        if confirmed_id is not None:
            bar_x1, bar_y1 = 10, 60
            bar_x2, bar_y2 = 310, 80
            cv2.rectangle(ann, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
            fill_x    = bar_x1 + int(300 * vote_ratio)
            bar_color = (0, 255, 0) if vote_ratio >= VOTE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(ann, (bar_x1, bar_y1), (fill_x, bar_y2), bar_color, -1)
            cv2.rectangle(ann, (bar_x1, bar_y1), (bar_x2, bar_y2), (200, 200, 200), 1)
            cv2.putText(ann,
                        f"VOTING {pad_label}: ID={confirmed_id} "
                        f"{best_count}/{total} ({vote_ratio:.0%})",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"\r  ID={confirmed_id}  {best_count}/{total} ({vote_ratio:.0%})",
                  end="", flush=True)

            if vote_ratio >= VOTE_THRESHOLD:
                print(f"\n\n{'='*55}")
                print(f"  {pad_label}  AprilTag ID = {confirmed_id}")
                print(f"  ({best_count}/{total} frames agreed)")
                print(f"{'='*55}\n")
                cv2.putText(ann, f"CONFIRMED!  ID = {confirmed_id}",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 255, 0), 3)
                cv2.imshow("Drone Mission", ann)
                cv2.waitKey(1500)
                return confirmed_id
        else:
            cv2.putText(ann, f"Searching {pad_label} AprilTag...",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("Drone Mission", ann)
        cv2.waitKey(1)
        time.sleep(0.03)

    print(f"\nWARNING: {pad_label} tag not confirmed in {APRILTAG_TIMEOUT}s")
    return confirmed_id

# ================================================================
# LINE FOLLOWING (handles curves via CurveStrips)
# ================================================================
def follow_line_to_pad(label="LINE"):
    """
    Follow yellow line until a black landing pad is detected.
    Returns True if pad reached, False on timeout/error.
    """
    print(f"\n{'='*55}")
    print(f"  FOLLOWING {label}")
    print(f"{'='*55}\n")

    line_lost_since  = None
    line_seen_once   = False
    line_seen_since  = None
    prev_error       = 0
    last_time        = time.time()

    while running:
        img = get_frame()
        if img is None:
            time.sleep(0.05)
            continue

        error, line_found, end_of_line, pad_ratio, display = detect_line_and_pad(img)

        if line_found and not line_seen_once:
            line_seen_once  = True
            line_seen_since = time.time()

        line_followed_enough = (
            line_seen_once and
            (time.time() - line_seen_since) >= LINE_MIN_FOLLOW_TIME
        )

        # PAD REACHED (line still visible)
        if end_of_line and line_followed_enough and line_found:
            stop_drone()
            print(f"\n\nPAD REACHED on {label}!")
            cv2.putText(display, f"PAD REACHED - {label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow("Drone Mission", display)
            cv2.waitKey(1000)
            return True

        # LINE LOST
        if not line_found:
            if line_lost_since is None:
                line_lost_since = time.time()
            lost = time.time() - line_lost_since

            # Check pad even when line is lost
            if end_of_line and line_followed_enough:
                stop_drone()
                print(f"\n\nPAD REACHED on {label} (pad detected while line lost)!")
                cv2.putText(display, f"PAD REACHED - {label}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("Drone Mission", display)
                cv2.waitKey(1000)
                return True

            if lost > LINE_LOST_TIMEOUT:
                stop_drone()
                print(f"\nLine lost {lost:.1f}s - aborting")
                return False

            # Drift forward slowly when line lost
            send_velocity(0.05, 0, 0)
            cv2.putText(display,
                        f"LINE LOST pad={pad_ratio:.2f} | {lost:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Drone Mission", display)
            time.sleep(0.05)
            continue

        # NORMAL LINE FOLLOWING
        line_lost_since = None
        now     = time.time()
        dt      = max(now - last_time, 0.02)
        d_term  = LINE_KD * (error - prev_error) / dt
        lateral = max(-MAX_LATERAL, min(MAX_LATERAL, LINE_KP * error + d_term))
        prev_error = error
        last_time  = now

        # Dynamic speed — slow near pad
        if pad_ratio > PAD_SLOWDOWN_THRESHOLD:
            fwd        = LINE_SPEED_SLOW
            spd_label  = f"SLOWING pad={pad_ratio:.2f}"
            status_col = (0, 165, 255)
        else:
            fwd        = LINE_SPEED
            spd_label  = f"fwd={fwd:.2f}"
            status_col = (0, 255, 0)

        send_velocity(fwd, lateral, 0)

        msg     = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        alt_str = f"{msg.relative_alt/1000:.1f}m" if msg else "---"

        cv2.putText(display,
                    f"{label} | alt={alt_str} err={error:+d}px "
                    f"lat={lateral:+.3f} {spd_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        cv2.imshow("Drone Mission", display)
        cv2.waitKey(1)

        print(f"\r{label} | alt={alt_str:5s} | err={error:+4d}px | "
              f"lat={lateral:+.3f} | {spd_label}",
              end="", flush=True)

        time.sleep(0.02)

    return False

def search_for_line(timeout=SEARCH_TIMEOUT, label="SEARCHING"):
    """
    Slowly rotate / drift trying to re-acquire the yellow line.
    Returns True once line is found in view.
    """
    print(f"\n{label}: Looking for yellow line...")
    start = time.time()
    while running and (time.time() - start) < timeout:
        img = get_frame()
        if img is None:
            time.sleep(0.05)
            continue

        angle, area = detect_yellow_line_direction(img)
        if angle is not None and area > 500:
            print(f"\n  Line found at angle {angle:.1f}° area={area}")
            stop_drone()
            return True

        # Slow rotation to search
        send_velocity(0, 0, 0)
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0, 10, 15, 1, 1, 0, 0, 0)

        show_frame(f"{label} | searching... {time.time()-start:.1f}s")
        time.sleep(0.5)

    print(f"\n  Line not found after {timeout}s")
    return False

# ================================================================
# MAIN MISSION
# ================================================================
detected_tags = {}  # pad_label -> tag_id

try:
    print("\n" + "="*55)
    print("  PHASE 2: DRONE AIRPORT NAVIGATION")
    print(f"  Airports to visit: {Airport}")
    print("="*55 + "\n")

    # ---- STEP 1: ARM + TAKEOFF ----
    set_mode("GUIDED")
    arm_drone()
    takeoff(TARGET_ALT)
    time.sleep(1)
    show_frame(f"HOVERING {TARGET_ALT}m - READY")
    time.sleep(1)

    # ---- STEP 2: FLY FORWARD to get over first yellow line ----
    fly_forward_timed(FWD_SPEED, 5.0, "FINDING FIRST LINE")

    # ---- STEP 3: FOLLOW LINE NETWORK ----
    # The drone navigates by line-following + AprilTag reading at each pad.
    # At each pad:
    #   - Stop and read AprilTag
    #   - Print the detected tag ID
    #   - Decide whether this is a target airport
    #   - If target: land; otherwise clear pad, find next line, continue

    pad_count = 0
    airports_visited = []
    mission_done = False

    while running and not mission_done:
        pad_count += 1
        pad_label = f"PAD {pad_count}"

        # Follow line to next pad
        reached = follow_line_to_pad(f"LINE -> {pad_label}")
        if not reached or not running:
            print(f"\nFailed to reach {pad_label} — trying to search for line...")
            found = search_for_line(label=f"RE-SEARCH {pad_label}")
            if found:
                reached = follow_line_to_pad(f"LINE -> {pad_label} (retry)")
            if not reached:
                print(f"Cannot reach {pad_label} — aborting mission")
                break

        # Read AprilTag at this pad
        tag_id = read_apriltag(pad_label)
        detected_tags[pad_label] = tag_id
        print(f">>> {pad_label}  AprilTag ID: {tag_id}")

        # Check if this is one of our target airports
        # Tag ID might directly correspond to airport number,
        # or we may need to decode from texture naming convention
        if tag_id is not None:
            airports_visited.append(tag_id)
            print(f"  Airports visited so far (tag IDs): {airports_visited}")

        # Check if we've visited all required airports
        # The Airport list contains airport numbers to visit
        # We check if all airports have been seen
        all_found = True
        for apt in Airport:
            if apt not in airports_visited:
                all_found = False
                break

        if all_found:
            print(f"\n  All airports {Airport} found! Landing here.")
            mission_done = True
            break

        # Not done yet — clear this pad and continue to next line
        # Lock current heading so we fly STRAIGHT past the junction
        heading_before = get_heading()
        print(f"\n  Not all airports visited yet. Clearing {pad_label}...")
        print(f"  Heading lock: {heading_before}°")
        fly_forward_timed(CLEAR_SPEED, CLEAR_DURATION, f"CLEARING {pad_label}")

        # Re-align to locked heading to avoid drift during clearing
        if heading_before is not None:
            current = get_heading()
            if current is not None:
                drift = abs(current - heading_before)
                if drift > 180:
                    drift = 360 - drift
                if drift > 5:
                    print(f"  Correcting heading drift: {current}° -> {heading_before}°")
                    turn_to_heading(heading_before)

        # Now look for the next yellow line
        # Give a short forward push to get fully past junction
        fly_forward_timed(CLEAR_SPEED, 1.0, "POST-JUNCTION ADVANCE")

        # Check if line is directly ahead
        img = get_frame()
        if img is not None:
            angle, area = detect_yellow_line_direction(img)
            if angle is not None and area > 500:
                print(f"  Next line visible at angle {angle:.1f}°")
                if abs(angle) > 20:
                    current = get_heading()
                    if current is not None:
                        new_heading = (current + angle) % 360
                        turn_to_heading(int(new_heading))
            else:
                # No line visible — try rotating to find one
                print("  No line ahead — searching...")
                found = search_for_line(timeout=30, label="FIND NEXT LINE")
                if not found:
                    fly_forward_timed(SEARCH_SPEED, 3.0, "ADVANCE & SEARCH")
                    search_for_line(timeout=30, label="FIND NEXT LINE (2)")

        time.sleep(0.5)

    # ---- FINAL: LAND ----
    if mission_done:
        stop_drone()
        time.sleep(0.5)
        land_and_disarm()

        print("\n" + "="*55)
        print("  MISSION COMPLETE")
        print("="*55)
        print(f"  Airports requested: {Airport}")
        print(f"  Tags detected:")
        for label, tid in detected_tags.items():
            print(f"    {label}: AprilTag ID = {tid}")
        print("="*55 + "\n")

        deadline = time.time() + 5
        while time.time() < deadline:
            img = get_frame()
            if img is not None:
                cv2.putText(img,
                            f"MISSION COMPLETE | Tags: {detected_tags}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Drone Mission", img)
            cv2.waitKey(30)
    else:
        print("\n  Mission incomplete — executing RTL")
        set_mode("RTL")

except RuntimeError as e:
    print(f"\n\nMISSION ABORTED: {e}")
    try:
        set_mode("RTL")
    except Exception:
        pass

except KeyboardInterrupt:
    print("\n\nInterrupted - RTL")
    try:
        set_mode("RTL")
    except Exception:
        pass

finally:
    running = False
    cv2.destroyAllWindows()
    cam_socket.close()
    print("Done.")
