#!/usr/bin/env python3
"""
PHASE 2: DRONE AIRPORT NAVIGATION — YAW-TRACKING APPROACH
============================================================
Instead of flying with a fixed heading and correcting laterally,
the drone YAWS to follow the curve so its front always faces along
the yellow line direction.

How it works:
  1. Detect yellow line in the bottom portion of the camera frame
  2. Fit a direction vector through the line pixels
  3. Calculate how much the line deviates from straight-ahead
  4. Command a yaw rate to steer the drone's nose onto the line
  5. Always fly forward in body frame — no sideways drift needed

This handles curves naturally — the drone turns with the road
like a car following a lane.
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
TARGET_ALT = 1.5

# Yellow line HSV (color stream — pure yellow is H≈30, S=255, V=255)
YELLOW_H_MIN = 15
YELLOW_H_MAX = 45
YELLOW_S_MIN = 80
YELLOW_S_MAX = 255
YELLOW_V_MIN = 100
YELLOW_V_MAX = 255

# Line detection ROI
LINE_ROI_START       = 0.35   # look further ahead for early curve anticipation
LINE_MIN_AREA        = 300


# ---- LINE-FOLLOWING PD (aggressive – must track curves tightly) ----
YAW_KP               = 0.50   # proportional gain (deg/s per degree of error)
YAW_KD               = 0.15   # derivative gain (damping to prevent oscillation)
MAX_YAW_RATE          = 35     # max degrees/sec yaw rate
CURVE_SLOWDOWN_ANGLE  = 15     # start slowing when line angle exceeds this (degrees)
CURVE_MIN_SPEED_RATIO = 0.4    # at max angle, speed drops to 40% of LINE_SPEED
# Small lateral PD to keep centered on the line (minor correction only)
LAT_KP                = 0.0008
MAX_LATERAL           = 0.06

# Speeds (m/s)
LINE_SPEED           = 0.20   # Forward speed while following the yellow line
LINE_SPEED_SLOW      = 0.10   # Reduced speed when approaching a pad
FWD_SPEED            = 0.30   # Speed for initial forward flight to find the first line
SEARCH_SPEED         = 0.15   # Forward creep speed while searching for a lost line

# Pad clearing
CLEAR_SPEED          = 0.25   # Forward speed when flying away from a pad to reach the next line
CLEAR_ADVANCE        = 5.0    # Seconds to fly forward after picking an exit line
CLEAR_DURATION       = 5.0    # Seconds to fly forward if no exit lines found (fallback clearing)

# Line lost
LINE_LOST_TIMEOUT    = 15.0   # Abort if the yellow line disappears for this many seconds
LINE_MIN_FOLLOW_TIME = 1.5    # Must follow line for at least 1.5s before a pad detection counts

# Black pad detection
PAD_V_MAX              = 60   # HSV V threshold — pixels darker than this are considered "pad"
PAD_ROI_TOP            = 0.20 # Top of the ROI rectangle for pad detection (20% of image height)
PAD_ROI_BOT            = 0.80 # Bottom of the ROI rectangle (80% of image height)
PAD_ROI_LEFT           = 0.15 # Left of the ROI rectangle (15% of image width)
PAD_ROI_RIGHT          = 0.85 # Right of the ROI rectangle (85% of image width)
PAD_FILL_RATIO         = 0.18 # 18%+ of ROI is dark → pad detected → stop the drone
PAD_SLOWDOWN_THRESHOLD = 0.12 # 12%+ dark but <18% → approaching pad → slow down

# AprilTag voting
VOTE_WINDOW      = 15         # Number of recent frames to vote over for tag confirmation
VOTE_THRESHOLD   = 0.55       # Tag ID must appear in 55%+ of the window to be confirmed
APRILTAG_TIMEOUT = 12         # Give up reading the AprilTag after 12 seconds

# ---- PAD TURNS (smooth – gentle yaw while hovering over pads) ----
PAD_TURN_SPEED   = 15         # Yaw rate in °/s for heading turns on the pad
PAD_TURN_TIMEOUT = 10         # Abort turn if target heading not reached in 10 seconds
PAD_TURN_MARGIN  = 8          # Accept turn as complete when within 8° of target heading

# Search
SEARCH_TIMEOUT  = 20          # Full rotation search gives up after 20 seconds

CAM_HOST    = "127.0.0.1"
CAM_PORT    = 5599
MAVLINK_URI = "udp:0.0.0.0:14550"

# ================================================================
# CAMERA (grayscale stream)
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
    global frame, running
    while running:
        try:
            header = b""
            while len(header) < header_size:
                chunk = cam_socket.recv(header_size - len(header))
                if not chunk:
                    running = False
                    return
                header += chunk
            width, height = struct.unpack("=HH", header)

            payload_size = width * height * 3  # 3-channel BGR color image
            img_bytes = b""
            while len(img_bytes) < payload_size:
                chunk = cam_socket.recv(min(payload_size - len(img_bytes), 65536))
                if not chunk:
                    running = False
                    return
                img_bytes += chunk

            img = np.frombuffer(img_bytes, np.uint8).reshape((height, width, 3))
            with frame_lock:
                frame = img  # Already BGR from Webots
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
    """Send body-frame velocity command."""
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0, 0, 0)

def send_velocity_and_yaw_rate(vx, vy, vz, yaw_rate_deg):
    """
    Send body-frame velocity + yaw rate command.
    This is the key function for yaw-tracking line following.
    yaw_rate_deg: positive = turn right (clockwise), negative = turn left.
    """
    # Bits: pos(0-2) vel(3-5) accel(6-8) force(9) yaw(10) yaw_rate(11)
    # 1 = ignore, 0 = use
    # We want: ignore pos, use vel, ignore accel, IGNORE yaw(10), USE yaw_rate(11)
    type_mask = 0b0000010111000111
    #                  ^^ bit11=0 (USE yaw_rate), bit10=1 (IGNORE yaw angle)
    yaw_rate_rad = math.radians(yaw_rate_deg)
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0, 0, 0,       # position (ignored)
        vx, vy, vz,    # velocity (body frame)
        0, 0, 0,        # acceleration (ignored)
        0,               # yaw (ignored)
        yaw_rate_rad)    # yaw rate

def stop_drone():
    for _ in range(10):
        send_velocity(0, 0, 0)
        time.sleep(0.05)

def get_heading():
    msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=3)
    if msg:
        return msg.heading
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
    time.sleep(0.3)

def turn_to_heading(target_heading, speed=None, margin=None, timeout_s=None):
    if speed is None:
        speed = PAD_TURN_SPEED
    if margin is None:
        margin = PAD_TURN_MARGIN
    if timeout_s is None:
        timeout_s = PAD_TURN_TIMEOUT

    current = get_heading()
    if current is None:
        time.sleep(3)
        return

    diff = (target_heading - current) % 360
    if diff > 180:
        diff -= 360
    direction = 1 if diff >= 0 else -1
    degrees = abs(diff)

    print(f"  Turning {degrees:.0f}° {'CW' if direction == 1 else 'CCW'} "
          f"from {current}° to {target_heading}°")
    # MAV_CMD_CONDITION_YAW params:
    #   p1=target heading, p2=speed, p3=direction (ignored in abs mode), p4=0=absolute
    abs_heading = target_heading % 360
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0, abs_heading, speed, direction, 0, 0, 0, 0)

    deadline = time.time() + timeout_s
    target_int = int(round(abs_heading)) % 360
    while time.time() < deadline:
        m = master.recv_match(type='VFR_HUD', blocking=False)
        if m:
            d = abs(m.heading - target_int)
            if d > 180:
                d = 360 - d
            show_frame(f"TURNING | {m.heading}° -> {target_int}° diff={d:.1f}°")
            if d < margin:
                print(f"\n  Turn complete at {m.heading}°")
                break
        time.sleep(0.05)
    time.sleep(0.3)

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

def land_on_pad():
    """Land on the current pad without disarming."""
    print("\nLanding on pad...")
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
                print("\nLanded on pad!")
                break
        time.sleep(0.05)

# ================================================================
# CV — LINE DETECTION WITH DIRECTION ANGLE
# ================================================================
def detect_line_with_angle(img):
    """
    Detect the yellow line and compute:
      - offset_error: horizontal pixel offset from center (for minor lateral correction)
      - line_angle: angle of the line relative to straight-ahead in degrees
                    positive = line curves right, negative = curves left
      - end_of_line: True if black pad detected
      - pad_ratio: how much black pad fills the ROI
      - display: annotated image

    The line angle is computed by fitting a line through the contour points
    using PCA or two-point method (top vs bottom of contour).
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

    # ---- Yellow line detection ----
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
        return 0, 0.0, False, end_of_line, pad_ratio, display

    # Pick the largest contour that is near center
    center_x = w // 2
    best_contour = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < LINE_MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        # Accept if centroid is in the middle 80% of the image
        margin = int(w * 0.4)
        if cx < center_x - margin or cx > center_x + margin:
            display_roi = display[roi_y:h, :]
            cv2.drawContours(display_roi, [cnt], -1, (0, 0, 180), 1)
            continue
        if area > best_area:
            best_area = area
            best_contour = cnt

    if best_contour is None:
        return 0, 0.0, False, end_of_line, pad_ratio, display

    # Compute centroid for lateral offset
    M = cv2.moments(best_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    offset_error = cx - center_x

    # ---- COMPUTE LINE DIRECTION ANGLE ----
    # Split contour points into top half and bottom half of ROI
    # The angle between their centroids gives the line direction
    roi_h = h - roi_y
    points = best_contour.reshape(-1, 2)  # shape (N, 2) with (x, y)

    mid_y = roi_h // 2
    top_points = points[points[:, 1] < mid_y]
    bot_points = points[points[:, 1] >= mid_y]

    if len(top_points) > 5 and len(bot_points) > 5:
        # Centroid of top and bottom sections
        top_cx = np.mean(top_points[:, 0])
        top_cy = np.mean(top_points[:, 1])
        bot_cx = np.mean(bot_points[:, 0])
        bot_cy = np.mean(bot_points[:, 1])

        # Direction vector from bottom to top (forward direction of the line)
        dx = top_cx - bot_cx
        dy = top_cy - bot_cy  # negative = upward in image = forward

        # Angle: 0° = straight ahead, positive = line goes right
        # atan2(dx, -dy) gives angle from vertical
        line_angle = math.degrees(math.atan2(dx, -dy))

        # Draw direction arrow on display
        arrow_start = (int(bot_cx), roi_y + int(bot_cy))
        arrow_end   = (int(top_cx), roi_y + int(top_cy))
        cv2.arrowedLine(display, arrow_start, arrow_end, (255, 0, 255), 3, tipLength=0.3)
    else:
        # Fallback: use fitLine on all contour points
        if len(points) > 10:
            vx_f, vy_f, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            line_angle = math.degrees(math.atan2(vx_f, -vy_f))
            # Ensure angle is in a sensible range
            if line_angle > 90:
                line_angle -= 180
            elif line_angle < -90:
                line_angle += 180
        else:
            line_angle = 0.0

    # Draw contour and centroid
    display_roi = display[roi_y:h, :]
    cv2.drawContours(display_roi, [best_contour], -1, (0, 255, 0), 2)
    cv2.circle(display_roi, (cx, cy), 10, (0, 0, 255), -1)
    cv2.line(display, (center_x, roi_y), (center_x, h), (255, 0, 0), 2)

    # Show angle on display
    cv2.putText(display, f"angle={line_angle:+.1f} deg",
                (10, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return offset_error, line_angle, True, end_of_line, pad_ratio, display

# ================================================================
# APRILTAG — VOTING CONFIRMATION
# ================================================================
def read_apriltag(pad_label="PAD"):
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
                cv2.waitKey(500)
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
# APRILTAG DECODE
# ================================================================
def decode_apriltag_id(tag_id):
    """Decode a 3-digit AprilTag ID.
    Format: [Country Code][Status][Reachable Airports]
    Example: tag_id=112 -> country=1, status=1, reachable=2
    """
    country_code = tag_id // 100
    status       = (tag_id // 10) % 10
    reachable    = tag_id % 10
    return country_code, status, reachable

# ================================================================
# CV — PAD DETECTION & CENTERING
# ================================================================
def detect_pad_center(img):
    """Detect the black landing pad and return its center offset from image center.
    Returns (offset_x, offset_y, found, display).
    offset_x positive = pad is to the right of camera center.
    offset_y positive = pad is below camera center (need to fly forward).
    """
    h, w = img.shape[:2]
    display = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Detect dark pixels (the grey/black pad)
    pad_mask = cv2.inRange(hsv,
                           np.array([0, 0, 0]),
                           np.array([179, 50, PAD_V_MAX]))

    k = np.ones((9, 9), np.uint8)
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_CLOSE, k)
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(pad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, False, display

    # Find the largest dark contour (the pad)
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 800:
        return 0, 0, False, display

    M = cv2.moments(best)
    if M["m00"] == 0:
        return 0, 0, False, display

    pad_cx = int(M["m10"] / M["m00"])
    pad_cy = int(M["m01"] / M["m00"])
    img_cx = w // 2
    img_cy = h // 2

    offset_x = pad_cx - img_cx
    offset_y = pad_cy - img_cy

    # Draw annotations
    cv2.drawContours(display, [best], -1, (0, 255, 255), 2)
    cv2.circle(display, (pad_cx, pad_cy), 8, (0, 0, 255), -1)   # pad center
    cv2.circle(display, (img_cx, img_cy), 8, (255, 0, 0), -1)   # image center
    cv2.line(display, (img_cx, img_cy), (pad_cx, pad_cy), (0, 255, 0), 2)
    cv2.putText(display, f"pad=({offset_x:+d},{offset_y:+d})",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return offset_x, offset_y, True, display


def detect_pad_angle(img):
    """Detect the pad's rotation angle from its corners using minAreaRect.
    Returns (angle_correction, corners, found, display).
    angle_correction is in [-45, +45] degrees — the yaw needed to align
    the drone parallel/perpendicular to the pad edges.
    """
    h, w = img.shape[:2]
    display = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pad_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 50, PAD_V_MAX]))

    k = np.ones((9, 9), np.uint8)
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_CLOSE, k)
    pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(pad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None, False, display

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 800:
        return 0, None, False, display

    # Minimum area rotated rectangle
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    box_int = np.intp(box)

    # Draw corners and edges
    cv2.drawContours(display, [box_int], 0, (0, 255, 0), 2)
    for i, pt in enumerate(box_int):
        cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(display, f"C{i}", (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Compute midpoints of opposite sides to find the vertical/horizontal axes
    # box points are ordered: 0-1-2-3 around the rectangle
    edge0 = box[1] - box[0]
    edge1 = box[2] - box[1]
    len0 = np.linalg.norm(edge0)
    len1 = np.linalg.norm(edge1)

    # Midpoints of opposite sides
    mid01 = (box[0] + box[1]) / 2   # midpoint of side 0-1
    mid23 = (box[2] + box[3]) / 2   # midpoint of side 2-3
    mid12 = (box[1] + box[2]) / 2   # midpoint of side 1-2
    mid30 = (box[3] + box[0]) / 2   # midpoint of side 3-0

    # Draw both midpoint-to-midpoint axes
    cv2.line(display, tuple(mid01.astype(int)), tuple(mid23.astype(int)), (255, 0, 255), 2)
    cv2.line(display, tuple(mid12.astype(int)), tuple(mid30.astype(int)), (255, 255, 0), 2)

    # Pick the axis that is more vertical (closer to image Y axis)
    # This is the line we want the drone's forward direction to align with
    axis_a = mid23 - mid01  # connects midpoints of sides 0-1 and 2-3
    axis_b = mid30 - mid12  # connects midpoints of sides 1-2 and 3-0

    # Angle from vertical (image up = -Y): atan2(dx, -dy)
    angle_a = abs(math.degrees(math.atan2(axis_a[0], -axis_a[1])))
    angle_b = abs(math.degrees(math.atan2(axis_b[0], -axis_b[1])))

    # Choose the more vertical axis
    if angle_a <= angle_b:
        chosen_axis = axis_a
        label_axis = "A"
    else:
        chosen_axis = axis_b
        label_axis = "B"

    # Angle of the chosen axis from straight-up (drone's forward)
    # Positive = axis tilts right = need to yaw right
    angle_correction = math.degrees(math.atan2(chosen_axis[0], -chosen_axis[1]))

    cv2.putText(display, f"Pad axis {label_axis}: {angle_correction:+.1f} deg",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return angle_correction, box_int, True, display


def align_with_pad(timeout=8.0, label="ALIGN PAD"):
    """Detect pad corners, compute rotation angle, and yaw the drone
    to be perpendicular to two pad edges and parallel to the other two.
    Samples the angle multiple times for stability, then applies one yaw correction.
    """
    print(f"  {label}: Detecting pad corners for alignment...")

    SAMPLES_NEEDED = 5
    ANGLE_THRESHOLD = 3.0  # degrees — skip if already aligned

    angle_samples = []
    start = time.time()

    while running and (time.time() - start) < timeout:
        img = get_frame()
        if img is None:
            time.sleep(0.05)
            continue

        angle_corr, corners, found, display = detect_pad_angle(img)

        if not found:
            cv2.putText(display, f"{label} | No pad visible",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Drone Mission", display)
            cv2.waitKey(1)
            time.sleep(0.05)
            continue

        angle_samples.append(angle_corr)

        cv2.putText(display,
                    f"{label} | Sample {len(angle_samples)}/{SAMPLES_NEEDED} "
                    f"angle={angle_corr:+.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("Drone Mission", display)
        cv2.waitKey(1)

        if len(angle_samples) >= SAMPLES_NEEDED:
            break

        time.sleep(0.1)

    if not angle_samples:
        print(f"  {label}: Could not detect pad corners")
        return False

    avg_angle = sum(angle_samples) / len(angle_samples)
    print(f"  {label}: Avg pad rotation = {avg_angle:+.1f}°")

    if abs(avg_angle) < ANGLE_THRESHOLD:
        print(f"  {label}: Already aligned (< {ANGLE_THRESHOLD}°)")
        return True

    # Yaw by angle_correction to align pad edges with image axes
    current_heading = get_heading()
    if current_heading is None:
        print(f"  {label}: Cannot get heading")
        return False

    target_heading = (current_heading + avg_angle) % 360
    print(f"  {label}: Heading {current_heading}° -> {target_heading:.0f}° "
          f"(yaw {avg_angle:+.1f}°)")

    turn_to_heading(target_heading)
    time.sleep(0.5)

    print(f"  {label}: Alignment complete")
    return True


def center_over_pad(timeout=10.0, label="CENTERING"):
    """Detect the pad center ONCE (averaged over a few samples),
    then fly a fixed correction to place the drone over that locked point.
    No continuous re-detection — avoids the shifting-target problem.
    """
    print(f"  {label}: Locking pad center position...")

    SAMPLES_NEEDED = 5
    MOVE_SPEED = 0.06          # m/s — gentle correction speed
    PIXELS_TO_METERS = 0.003   # approx m per pixel at ~2 m altitude
    ALREADY_CENTERED = 15      # pixels — skip move if already close

    # ── Phase 1: collect a few samples and average them ──
    samples_x, samples_y = [], []
    no_pad_streak = 0
    start = time.time()

    while running and (time.time() - start) < timeout / 2:
        img = get_frame()
        if img is None:
            time.sleep(0.05)
            continue

        offset_x, offset_y, found, display = detect_pad_center(img)

        if not found:
            no_pad_streak += 1
            if no_pad_streak < 40:
                send_velocity_and_yaw_rate(0.04, 0, 0, 0)
            else:
                send_velocity_and_yaw_rate(0, 0, 0, 0)
            cv2.putText(display, f"{label} | Searching for pad...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Drone Mission", display)
            cv2.waitKey(1)
            time.sleep(0.05)
            continue

        no_pad_streak = 0
        samples_x.append(offset_x)
        samples_y.append(offset_y)

        cv2.putText(display,
                    f"{label} | Sampling {len(samples_x)}/{SAMPLES_NEEDED}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("Drone Mission", display)
        cv2.waitKey(1)

        if len(samples_x) >= SAMPLES_NEEDED:
            stop_drone()
            break

        time.sleep(0.05)

    if not samples_x:
        stop_drone()
        print(f"  {label}: Could not detect pad at all")
        return False

    # ── Locked target (averaged) ──
    locked_x = int(sum(samples_x) / len(samples_x))
    locked_y = int(sum(samples_y) / len(samples_y))
    print(f"  {label}: Locked offset = ({locked_x:+d}, {locked_y:+d}) px")

    if abs(locked_x) < ALREADY_CENTERED and abs(locked_y) < ALREADY_CENTERED:
        print(f"  {label}: Already centered — no move needed")
        return True

    # ── Phase 2: dead-reckon toward locked point ──
    dist_x_m = locked_x * PIXELS_TO_METERS   # rightward in image → +vy
    dist_y_m = locked_y * PIXELS_TO_METERS   # downward in image

    move_time = max(abs(dist_x_m), abs(dist_y_m)) / MOVE_SPEED
    if move_time < 0.1:
        move_time = 0.1

    # Compute constant velocity components
    # offset_y positive = pad below center = pad is closer = fly BACKWARD (negative vx)
    vx = -dist_y_m / move_time  # forward/back (inverted)
    vy = dist_x_m / move_time   # left/right

    print(f"  {label}: Moving vx={vx:+.3f} vy={vy:+.3f} for {move_time:.2f}s")
    move_start = time.time()
    while running and (time.time() - move_start) < move_time:
        send_velocity_and_yaw_rate(vx, vy, 0, 0)
        # Show progress on display
        img = get_frame()
        if img is not None:
            display = img.copy()
            h, w = display.shape[:2]
            elapsed = time.time() - move_start
            cv2.putText(display,
                        f"{label} | Moving {elapsed:.1f}/{move_time:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            # Draw locked target crosshair
            tx = w // 2 + locked_x
            ty = h // 2 + locked_y
            cv2.drawMarker(display, (tx, ty), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(display, (w // 2, h // 2), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.imshow("Drone Mission", display)
            cv2.waitKey(1)
        time.sleep(0.05)

    stop_drone()
    print(f"  {label}: Centering move complete")
    return True


def detect_exit_lines(img):
    """Detect all yellow line segments visible in the full image.
    Returns a list of exit lines, each as a dict:
      {'angle': degrees from center (negative=left, positive=right),
       'area': contour area (bigger = closer),
       'cx': centroid x, 'cy': centroid y}
    Sorted by area descending (closest first).
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv,
        np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN]),
        np.array([YELLOW_H_MAX, YELLOW_S_MAX, YELLOW_V_MAX]))

    k = np.ones((7, 7), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, k)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    img_cx = w // 2
    img_cy = h // 2
    lines = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < LINE_MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Angle from image center: atan2(dx, -dy)
        # 0° = straight ahead (top of image), positive = right, negative = left
        dx = cx - img_cx
        dy = cy - img_cy
        angle = math.degrees(math.atan2(dx, -dy))

        lines.append({
            'angle': angle,
            'area': area,
            'cx': cx,
            'cy': cy,
            'contour': cnt
        })

    # Sort by area descending (closest/biggest first)
    lines.sort(key=lambda l: l['area'], reverse=True)
    return lines


# ================================================================
# YAW-TRACKING LINE FOLLOWING
# ================================================================

# Reference heading — set after takeoff, used to turn perpendicular to pads
reference_heading = None

def follow_line_to_pad(label="LINE"):
    """
    Follow yellow line by yawing the drone to match the line direction.
    The drone keeps following the line even after seeing the pad — it only
    stops once the line disappears (meaning the drone has reached the pad
    edge). Then it turns back to the reference heading (perpendicular to
    the pad), creeps forward to center, and returns for AprilTag reading.
    """
    global reference_heading

    print(f"\n{'='*55}")
    print(f"  YAW-TRACKING: {label}")
    print(f"  Reference heading: {reference_heading}")
    print(f"{'='*55}\n")

    line_lost_since  = None
    line_seen_once   = False
    line_seen_since  = None
    pad_seen_once    = False
    prev_angle_err   = 0.0
    last_time        = time.time()

    while running:
        img = get_frame()
        if img is None:
            time.sleep(0.05)
            continue

        offset_error, line_angle, line_found, end_of_line, pad_ratio, display = \
            detect_line_with_angle(img)

        if line_found and not line_seen_once:
            line_seen_once  = True
            line_seen_since = time.time()

        if end_of_line:
            pad_seen_once = True

        line_followed_enough = (
            line_seen_once and
            (time.time() - line_seen_since) >= LINE_MIN_FOLLOW_TIME
        )

        # PAD DETECTED — stop immediately, don't keep following the line
        if end_of_line and line_followed_enough:
            print(f"\n\n  PAD DETECTED on {label} — stopping immediately")
            stop_drone()

            # Hold hover for 0.8s to let drone settle
            hover_start = time.time()
            while time.time() - hover_start < 0.8:
                send_velocity_and_yaw_rate(0, 0, 0, 0)
                time.sleep(0.05)

            # CV-BASED CENTERING over pad
            center_over_pad(timeout=4.0, label=f"CENTER {label}")

            # Hold stable
            stable_start = time.time()
            while time.time() - stable_start < 0.5:
                send_velocity_and_yaw_rate(0, 0, 0, 0)
                show_frame(f"STABLE over {label}")
                time.sleep(0.05)

            stop_drone()
            time.sleep(0.2)
            print(f"  Perpendicular & centered — ready for AprilTag!")
            return True

        # LINE STILL VISIBLE and no pad — keep following
        if line_found:
            line_lost_since = None
            now = time.time()
            dt  = max(now - last_time, 0.02)

            # Yaw PD controller
            angle_err = line_angle
            d_angle   = YAW_KD * (angle_err - prev_angle_err) / dt
            yaw_rate  = YAW_KP * angle_err + d_angle
            yaw_rate  = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, yaw_rate))

            prev_angle_err = angle_err
            last_time      = now

            # Small lateral correction
            lateral = LAT_KP * offset_error
            lateral = max(-MAX_LATERAL, min(MAX_LATERAL, lateral))

            # Forward speed — slow down on curves and when approaching pad
            if pad_ratio > PAD_FILL_RATIO:
                fwd        = LINE_SPEED_SLOW * 0.5
                spd_label  = f"PAD NEAR pad={pad_ratio:.2f}"
                status_col = (0, 0, 255)
            elif pad_ratio > PAD_SLOWDOWN_THRESHOLD:
                fwd        = LINE_SPEED_SLOW
                spd_label  = f"SLOWING pad={pad_ratio:.2f}"
                status_col = (0, 165, 255)
            else:
                # Slow down proportionally when the line curves
                abs_angle = abs(line_angle)
                if abs_angle > CURVE_SLOWDOWN_ANGLE:
                    # Linear interpolation: 15°->100%, 45°->40%
                    t = min((abs_angle - CURVE_SLOWDOWN_ANGLE) / 30.0, 1.0)
                    speed_ratio = 1.0 - t * (1.0 - CURVE_MIN_SPEED_RATIO)
                    fwd = LINE_SPEED * speed_ratio
                    spd_label  = f"CURVE {abs_angle:.0f}° spd={fwd:.2f}"
                    status_col = (0, 200, 255)
                else:
                    fwd        = LINE_SPEED
                    spd_label  = f"fwd={fwd:.2f}"
                    status_col = (0, 255, 0)

            send_velocity_and_yaw_rate(fwd, lateral, 0, yaw_rate)

            msg     = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
            alt_str = f"{msg.relative_alt/1000:.1f}m" if msg else "---"

            cv2.putText(display,
                        f"{label} | yaw={yaw_rate:+.1f} | "
                        f"angle={line_angle:+.1f} | {spd_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 2)
            cv2.imshow("Drone Mission", display)
            cv2.waitKey(1)

            print(f"\r{label} | alt={alt_str:5s} | angle={line_angle:+5.1f}° | "
                  f"yaw={yaw_rate:+5.1f}°/s | lat={lateral:+.2f} | {spd_label}",
                  end="", flush=True)
            time.sleep(0.02)
            continue

        # LINE LOST
        if line_lost_since is None:
            line_lost_since = time.time()
        lost = time.time() - line_lost_since

        if lost > LINE_LOST_TIMEOUT:
            stop_drone()
            print(f"\nLine lost {lost:.1f}s - aborting")
            return False

        # Line lost but no pad yet — creep forward slowly
        send_velocity(0.05, 0, 0)
        cv2.putText(display,
                    f"LINE LOST pad={pad_ratio:.2f} | {lost:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Drone Mission", display)
        cv2.waitKey(1)
        time.sleep(0.05)

    return False

def search_for_line(timeout=SEARCH_TIMEOUT, label="SEARCHING"):
    """
    Rotate to find the yellow line, then yaw to face along it.
    Tries clockwise first, then counter-clockwise if not found.
    Once found, keeps rotating until the line is roughly centered
    and the drone's nose points along the line direction.
    """
    print(f"\n{label}: Looking for yellow line...")

    for direction, dir_label in [(1, "CW"), (-1, "CCW")]:
        yaw_search_rate = 20 * direction  # degrees/sec
        start = time.time()
        half_timeout = timeout / 2

        while running and (time.time() - start) < half_timeout:
            img = get_frame()
            if img is None:
                time.sleep(0.05)
                continue

            offset_error, line_angle, found, _, _, display = detect_line_with_angle(img)
            if found:
                print(f"\n  Line found ({dir_label})! angle={line_angle:+.1f}° offset={offset_error:+d}px")
                # Now yaw to align the drone's nose with the line
                align_start = time.time()
                while running and (time.time() - align_start) < 8.0:
                    img2 = get_frame()
                    if img2 is None:
                        time.sleep(0.05)
                        continue
                    off2, ang2, still_found, _, _, disp2 = detect_line_with_angle(img2)
                    if not still_found:
                        break
                    # Check if aligned: line angle near 0 AND offset near center
                    if abs(ang2) < 10 and abs(off2) < 60:
                        stop_drone()
                        print(f"  Aligned! angle={ang2:+.1f}° offset={off2:+d}px")
                        cv2.putText(disp2, f"ALIGNED angle={ang2:+.1f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Drone Mission", disp2)
                        cv2.waitKey(500)
                        return True
                    # Keep yawing toward the line
                    align_yaw = 0.5 * ang2  # proportional correction
                    align_yaw = max(-25, min(25, align_yaw))
                    send_velocity_and_yaw_rate(0, 0, 0, align_yaw)
                    cv2.putText(disp2, f"ALIGNING ang={ang2:+.1f} off={off2:+d}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("Drone Mission", disp2)
                    cv2.waitKey(1)
                    time.sleep(0.05)

                # Even if not perfectly aligned, we found the line
                stop_drone()
                print(f"  Line found, alignment partial")
                return True

            # Keep rotating to search
            send_velocity_and_yaw_rate(0, 0, 0, yaw_search_rate)
            cv2.putText(display, f"{label} | {dir_label} {time.time()-start:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Drone Mission", display)
            cv2.waitKey(1)
            time.sleep(0.1)

        stop_drone()
        time.sleep(0.3)

    print(f"\n  Line not found after {timeout}s")
    return False

# ================================================================
# MAIN MISSION
# ================================================================
# Filter out 0 (means "no second destination")
countries_needed = [c for c in Airport if c != 0]
countries_landed = set()
detected_tags    = {}

try:
    print("\n" + "="*55)
    print("  PHASE 2: AIRPORT NAVIGATION")
    print(f"  Target countries: {countries_needed}")
    print("="*55 + "\n")

    # STEP 1: ARM + TAKEOFF
    set_mode("GUIDED")
    arm_drone()
    mission_start_time = time.time()
    takeoff(TARGET_ALT)
    time.sleep(0.5)

    # Record initial heading as reference — drone starts perpendicular to pads
    reference_heading = get_heading()
    if reference_heading is None:
        reference_heading = 0
    print(f"  REFERENCE HEADING SET: {reference_heading}")
    show_frame(f"HOVERING {TARGET_ALT}m - REF HDG={reference_heading}")
    time.sleep(0.5)

    # STEP 2: FLY FORWARD to get over first yellow line
    fly_forward_timed(FWD_SPEED, 3.0, "FINDING FIRST LINE")

    # STEP 3: FOLLOW LINE NETWORK
    pad_count    = 0
    mission_done = False

    while running and not mission_done:
        pad_count += 1
        pad_label = f"PAD {pad_count}"

        # Follow line to next pad
        reached = follow_line_to_pad(f"LINE -> {pad_label}")
        if not reached or not running:
            print(f"\nFailed to reach {pad_label} — searching...")
            found = search_for_line(label=f"RE-SEARCH {pad_label}")
            if found:
                reached = follow_line_to_pad(f"LINE -> {pad_label} (retry)")
            if not reached:
                print(f"Cannot reach {pad_label} — aborting mission")
                break

        # Read AprilTag
        tag_id = read_apriltag(pad_label)
        print(f">>> {pad_label}  AprilTag ID: {tag_id}")

        # Decode tag and make landing decision
        landed_here = False
        if tag_id is not None:
            country_code, status, reachable = decode_apriltag_id(tag_id)
            detected_tags[pad_label] = {
                'tag_id': tag_id,
                'country': country_code,
                'status': status,
                'reachable': reachable
            }
            print(f"  Decoded: Country={country_code}, "
                  f"Status={'SAFE' if status == 1 else 'UNSAFE'}, "
                  f"Reachable={reachable}")

            # Check if this is a target airport with safe landing status
            if (country_code in countries_needed and
                    country_code not in countries_landed and
                    status == 1):
                print(f"\n  *** TARGET AIRPORT FOUND! Country {country_code} ***")

                # Re-center over pad before landing
                center_over_pad(timeout=4.0, label=f"PRE-LAND CENTER {pad_label}")
                # Brief hover to stabilize
                for _ in range(10):
                    send_velocity_and_yaw_rate(0, 0, 0, 0)
                    time.sleep(0.05)

                # Land on the pad
                land_on_pad()

                # Wait at least 4 seconds on the ground (use 5 for safety)
                print(f"  Waiting on landing pad (4s required)...")
                wait_start = time.time()
                while time.time() - wait_start < 4.2:
                    show_frame(f"GROUNDED Country={country_code} | "
                               f"{time.time()-wait_start:.1f}s / 4.2s")
                    time.sleep(0.1)
                print(f"  Wait complete!")

                countries_landed.add(country_code)
                landed_here = True
                print(f"  Countries landed: {countries_landed} "
                      f"/ needed: {set(countries_needed)}")

                # Check if all target countries visited
                if set(countries_needed) <= countries_landed:
                    print(f"\n{'='*55}")
                    print(f"  ALL TARGET COUNTRIES VISITED!")
                    print(f"  Disarming on final pad.")
                    print(f"{'='*55}")
                    master.arducopter_disarm()
                    time.sleep(0.5)
                    mission_done = True
                    break
                else:
                    # More countries to visit — take off again
                    print(f"\n  Taking off to continue mission...")
                    set_mode("GUIDED")
                    arm_drone()
                    takeoff(TARGET_ALT)
                    time.sleep(0.5)
            else:
                if country_code in countries_landed:
                    print(f"  Country {country_code} already visited — skipping")
                elif status != 1:
                    print(f"  Landing UNSAFE (status={status}) — skipping")
                else:
                    print(f"  Country {country_code} not a target — skipping")
        else:
            detected_tags[pad_label] = None
            print(f"  Could not read tag — continuing")

        # ============================================
        # POST-PAD: Find next line to continue
        # ============================================
        if mission_done:
            break

        # Use the reachable count from the tag to know how many exit lines to expect
        expected_exits = 0
        if tag_id is not None:
            _, _, reachable = decode_apriltag_id(tag_id)
            # reachable = total connected lines; minus 1 for the arrival line
            expected_exits = max(reachable - 1, 0)
            print(f"\n  POST-PAD: Reachable={reachable}, "
                  f"expecting {expected_exits} exit line(s)")
        else:
            print(f"\n  POST-PAD: No tag — scanning for any exit lines")

        # CV-BASED EXIT LINE DETECTION — no rotation needed
        # The drone faces forward (arrival direction), so the arrival line is
        # behind the camera and invisible. Any yellow lines visible are exits.
        exit_lines = []
        for _ in range(8):  # sample multiple frames
            img = get_frame()
            if img is None:
                time.sleep(0.05)
                continue
            exit_lines = detect_exit_lines(img)
            if exit_lines:
                break
            time.sleep(0.1)

        if exit_lines:
            # Classify lines before drawing
            forward_lines = [l for l in exit_lines if abs(l['angle']) <= 90]
            AHEAD_THRESHOLD = 45  # degrees from center to count as "ahead"
            ahead_lines = [l for l in forward_lines if abs(l['angle']) <= AHEAD_THRESHOLD]

            # Determine which line will be selected
            if ahead_lines:
                chosen = max(ahead_lines, key=lambda l: l['area'])
                choice_label = "AHEAD"
            elif forward_lines:
                chosen = max(forward_lines, key=lambda l: l['area'])
                choice_label = "FWD SIDE"
            else:
                chosen = exit_lines[0]
                choice_label = "BEHIND"

            # Draw all detected exit lines on display
            display = img.copy()
            h_img, w_img = display.shape[:2]
            for i, el in enumerate(exit_lines):
                is_chosen = (el is chosen)
                behind = abs(el['angle']) > 90

                if is_chosen:
                    color = (0, 255, 0)      # green = selected
                elif behind:
                    color = (0, 0, 255)      # red = behind drone
                else:
                    color = (0, 165, 255)    # orange = forward but not chosen

                thickness = 3 if is_chosen else 1
                cv2.drawContours(display, [el['contour']], -1, color, thickness)
                cv2.circle(display, (el['cx'], el['cy']), 8, color, -1)

                tag = "SELECTED" if is_chosen else ("BEHIND" if behind else "")
                cv2.putText(display,
                    f"L{i}: {el['angle']:+.0f}° a={el['area']:.0f} {tag}",
                    (el['cx'] + 10, el['cy']),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.putText(display,
                f"EXIT: {len(exit_lines)} lines | {len(forward_lines)} fwd | "
                f"choice={choice_label} {chosen['angle']:+.0f}°",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("Drone Mission", display)
            cv2.waitKey(500)

            if ahead_lines:
                # Line directly in front — fly forward with minor correction
                best_exit = chosen
                yaw_needed = best_exit['angle']
                print(f"  Line ahead: angle={yaw_needed:+.1f}°, "
                      f"area={best_exit['area']:.0f} — flying forward")
                if abs(yaw_needed) > 5:
                    current_hdg = get_heading()
                    if current_hdg is not None:
                        target_hdg = (current_hdg + yaw_needed) % 360
                        print(f"  Minor correction {yaw_needed:+.1f}° "
                              f"(heading {current_hdg}° -> {target_hdg:.0f}°)")
                        turn_to_heading(target_hdg)
                        time.sleep(0.5)
                else:
                    print(f"  Line is straight ahead — no turn needed")
            elif forward_lines:
                # Forward line exists but off to the side (45-90°)
                best_exit = max(forward_lines, key=lambda l: l['area'])
                yaw_needed = best_exit['angle']
                print(f"  Forward side line: angle={yaw_needed:+.1f}°, "
                      f"area={best_exit['area']:.0f}")
                current_hdg = get_heading()
                if current_hdg is not None:
                    target_hdg = (current_hdg + yaw_needed) % 360
                    print(f"  Turning {yaw_needed:+.1f}° toward forward line "
                          f"(heading {current_hdg}° -> {target_hdg:.0f}°)")
                    turn_to_heading(target_hdg)
                    time.sleep(0.5)
            else:
                # Only behind lines visible — use the largest one
                best_exit = exit_lines[0]
                yaw_needed = best_exit['angle']
                print(f"  No forward line — using behind line: "
                      f"angle={yaw_needed:+.1f}°, area={best_exit['area']:.0f}")
                current_hdg = get_heading()
                if current_hdg is not None:
                    target_hdg = (current_hdg + yaw_needed) % 360
                    print(f"  Turning {yaw_needed:+.1f}° toward line "
                          f"(heading {current_hdg}° -> {target_hdg:.0f}°)")
                    turn_to_heading(target_hdg)
                    time.sleep(0.5)

            # Fly forward to clear the pad and reach the line
            fly_forward_timed(CLEAR_SPEED, CLEAR_ADVANCE, f"ADVANCING TO LINE from {pad_label}")
        else:
            # No lines visible in current view — do a minimal ±90° scan
            print("  No exit lines visible in current view — scanning ±90°...")
            start_heading = get_heading()
            if start_heading is None:
                start_heading = 0

            best_heading = None
            best_area = 0
            SCAN_STEP = 30
            SCAN_SETTLE = 0.4

            for offset in [30, -30, 60, -60, 90, -90]:
                check_hdg = (start_heading + offset) % 360
                turn_to_heading(check_hdg)
                time.sleep(SCAN_SETTLE)

                img = get_frame()
                if img is None:
                    continue
                found_lines = detect_exit_lines(img)
                if found_lines:
                    print(f"    Heading {check_hdg:.0f}°: {len(found_lines)} line(s), "
                          f"best area={found_lines[0]['area']:.0f}")
                    if found_lines[0]['area'] > best_area:
                        best_area = found_lines[0]['area']
                        # Account for line's angle within the frame
                        best_heading = (check_hdg + found_lines[0]['angle']) % 360
                else:
                    print(f"    Heading {check_hdg:.0f}°: no lines")

            if best_heading is not None:
                print(f"\n  Best exit at heading {best_heading:.0f}°")
                turn_to_heading(best_heading)
                time.sleep(0.5)
                fly_forward_timed(CLEAR_SPEED, CLEAR_ADVANCE, f"ADVANCING TO LINE from {pad_label}")
            else:
                print("  No exit lines found — advancing and searching...")
                turn_to_heading(start_heading)
                time.sleep(0.5)
                fly_forward_timed(CLEAR_SPEED, CLEAR_DURATION, f"CLEARING {pad_label}")
                search_for_line(timeout=30, label="FULL SEARCH")

        time.sleep(0.5)

    # FINAL SUMMARY
    mission_elapsed = time.time() - mission_start_time
    mins = int(mission_elapsed // 60)
    secs = mission_elapsed % 60

    if mission_done:
        print("\n" + "="*55)
        print("  MISSION COMPLETE")
        print("="*55)
        print(f"  Total flight time: {mins}m {secs:.1f}s")
        print(f"  Target countries: {countries_needed}")
        print(f"  Countries landed: {countries_landed}")
        print(f"  Tags detected:")
        for label, info in detected_tags.items():
            if info and isinstance(info, dict):
                print(f"    {label}: ID={info['tag_id']} "
                      f"Country={info['country']} "
                      f"Status={info['status']} "
                      f"Reachable={info['reachable']}")
            else:
                print(f"    {label}: {info}")
        print("="*55 + "\n")

        deadline = time.time() + 5
        while time.time() < deadline:
            img = get_frame()
            if img is not None:
                cv2.putText(img,
                            f"MISSION COMPLETE | {mins}m {secs:.1f}s | Landed: {countries_landed}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Drone Mission", img)
            cv2.waitKey(30)
    else:
        print(f"\n  Mission incomplete — RTL")
        print(f"  Flight time so far: {mins}m {secs:.1f}s")
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
