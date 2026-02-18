import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import math
import serial

# -----------------------------
# Utilities
# -----------------------------
def l2(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def ema(x, prev, a):
    return x if prev is None else a * x + (1 - a) * prev

# -----------------------------
# UI Drawing
# -----------------------------
def draw_metrics_panel(img, state, perclos, distraction_ratio, gaze_status,
                       yaw, pitch, risk_score, risk_level, is_sustained):
    h, w = img.shape[:2]

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (310, 340), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    y_offset = 35
    line_height = 32

    cv2.putText(img, "DRIVER MONITORING", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    y_offset += line_height
    state_color = (0, 255, 0) if state == "AWAKE" else (0, 0, 255)
    if state == "CALIBRATING":
        state_color = (0, 165, 255)
    cv2.putText(img, f"State: {state}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    y_offset += line_height + 5
    risk_text, risk_color = risk_level
    cv2.putText(img, f"RISK: {risk_text} ({risk_score:.0f}%)", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, risk_color, 2)

    bar_width = int((risk_score / 100) * 270)
    cv2.rectangle(img, (20, y_offset + 8), (290, y_offset + 23), (60, 60, 60), -1)
    cv2.rectangle(img, (20, y_offset + 8), (20 + bar_width, y_offset + 23), risk_color, -1)

    y_offset += line_height + 20
    perclos_color = (0, 255, 0) if perclos < 15 else (0, 165, 255) if perclos < 30 else (0, 0, 255)
    cv2.putText(img, f"Sleep Ratio: {perclos:.1f}%", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, perclos_color, 1)

    bar_width = int((perclos / 100) * 270)
    cv2.rectangle(img, (20, y_offset + 6), (290, y_offset + 18), (60, 60, 60), -1)
    cv2.rectangle(img, (20, y_offset + 6), (20 + bar_width, y_offset + 18), perclos_color, -1)

    y_offset += line_height
    dist_color = (0, 255, 0) if distraction_ratio < 20 else (0, 165, 255) if distraction_ratio < 40 else (0, 0, 255)
    sustained_text = " [SUSTAINED]" if is_sustained else ""
    cv2.putText(img, f"Distraction: {distraction_ratio:.1f}%{sustained_text}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if is_sustained else dist_color, 1)

    bar_width = int((distraction_ratio / 100) * 270)
    cv2.rectangle(img, (20, y_offset + 6), (290, y_offset + 18), (60, 60, 60), -1)
    cv2.rectangle(img, (20, y_offset + 6), (20 + bar_width, y_offset + 18), dist_color, -1)

    y_offset += line_height + 5
    gaze_color = (0, 255, 0) if gaze_status == "CENTER" else (0, 165, 255)
    cv2.putText(img, f"Gaze: {gaze_status}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gaze_color, 1)

    y_offset += line_height - 5
    cv2.putText(img, f"Head: Yaw {yaw:.1f}  Pitch {pitch:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    y_offset = h - 70
    if is_sustained or risk_score > 60:
        alert_text = "!! CRITICAL ALERT"
        cv2.rectangle(img, (10, y_offset - 5), (310, y_offset + 35), (0, 0, 255), -1)
        cv2.putText(img, alert_text, (25, y_offset + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

# -----------------------------
# Small helpers for timing
# -----------------------------
class DurationCounter:
    def __init__(self, fps=20):
        self.fps = fps
        self.count = 0

    def update(self, flag):
        self.count = self.count + 1 if flag else 0

    def seconds(self):
        return self.count / max(1.0, self.fps)

class RatioTracker:
    def __init__(self, window_sec=10, fps=20):
        self.maxlen = int(window_sec * fps)
        self.buf = deque(maxlen=self.maxlen)

    def update(self, flag):
        self.buf.append(1 if flag else 0)

    def ratio(self):
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

# -----------------------------
# FaceMesh Detector (MediaPipe)
# -----------------------------
class FaceMeshDetector:
    def __init__(self, max_faces=1, min_det=0.5, min_track=0.5):
        self.mpfm = mp.solutions.face_mesh
        self.fm = self.mpfm.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    def find(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        faces = []
        if res.multi_face_landmarks:
            for f in res.multi_face_landmarks:
                faces.append([(lm.x, lm.y, lm.z) for lm in f.landmark])
        return faces

# -----------------------------
# Landmark indices (MediaPipe)
# -----------------------------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

POSE_LM = {
    "nose": 1,
    "chin": 152,
    "leye": 33,
    "reye": 263,
    "lmouth": 61,
    "rmouth": 291
}

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_eye_overlay(img, lm):
    h, w = img.shape[:2]
    px = lambda i: (int(lm[i][0] * w), int(lm[i][1] * h))

    for eye in (LEFT_EYE, RIGHT_EYE):
        for i in range(len(eye)):
            cv2.line(img, px(eye[i]), px(eye[(i + 1) % len(eye)]), (0, 200, 0), 1)

    for iris in (LEFT_IRIS, RIGHT_IRIS):
        pts = [px(i) for i in iris]
        for p in pts:
            cv2.circle(img, p, 2, (255, 0, 255), -1)
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)

# -----------------------------
# Head pose estimation (solvePnP)
# -----------------------------
def head_pose(lm, img_shape):
    h, w = img_shape[:2]
    image_points = np.array([
        (lm[POSE_LM["nose"]][0] * w,   lm[POSE_LM["nose"]][1] * h),
        (lm[POSE_LM["chin"]][0] * w,   lm[POSE_LM["chin"]][1] * h),
        (lm[POSE_LM["leye"]][0] * w,   lm[POSE_LM["leye"]][1] * h),
        (lm[POSE_LM["reye"]][0] * w,   lm[POSE_LM["reye"]][1] * h),
        (lm[POSE_LM["lmouth"]][0] * w, lm[POSE_LM["lmouth"]][1] * h),
        (lm[POSE_LM["rmouth"]][0] * w, lm[POSE_LM["rmouth"]][1] * h),
    ], dtype="double")

    model_points = np.array([
        (0.0,   0.0,   0.0),
        (0.0,  -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3,  32.7, -26.0),
        (-28.9,-28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    yaw   = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    pitch = math.degrees(math.atan2(-rmat[2, 0],
                                     math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2)))
    return yaw, pitch

# -----------------------------
# Gaze estimation
# -----------------------------
def gaze_offset(lm, eye, iris):
    ex = [lm[i][0] for i in eye]
    ix = [lm[i][0] for i in iris]
    eye_left   = min(ex)
    eye_right  = max(ex)
    eye_center = (eye_left + eye_right) / 2.0
    iris_center = sum(ix) / len(ix)
    return (iris_center - eye_center) / (eye_right - eye_left + 1e-6)

# -----------------------------
# EAR for blink / PERCLOS
# -----------------------------
def eye_aspect_ratio(lm, eye):
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in eye]
    a = l2(p2, p6)
    b = l2(p3, p5)
    c = l2(p1, p4)
    return (a + b) / (2.0 * c + 1e-6)

# -----------------------------
# Sleep Detector
# -----------------------------
class SleepDetector:
    def __init__(self, fps=20, alpha=0.12):
        self.fps = fps
        self.alpha = alpha
        self.sleep_frames = int(3.5 * fps)
        self.wake_frames  = int(2.0 * fps)
        self.ear_ema      = None
        self.calib        = []
        self.calibrated   = False
        self.ear_th       = 0.18
        self.state        = "CALIBRATING"
        self.consec_closed = 0
        self.consec_open   = 0
        self.sleep_ratio   = RatioTracker(window_sec=10, fps=fps)

    def process(self, lm):
        ear = (eye_aspect_ratio(lm, LEFT_EYE) + eye_aspect_ratio(lm, RIGHT_EYE)) / 2.0
        self.ear_ema = ema(ear, self.ear_ema, self.alpha)

        if not self.calibrated:
            if self.ear_ema > 0.16:
                self.calib.append(self.ear_ema)
            if len(self.calib) > self.fps * 4:
                mu = np.mean(self.calib)
                sd = np.std(self.calib)
                self.ear_th     = max(0.12, mu - 1.2 * sd)
                self.calibrated = True
                self.state      = "AWAKE"
            return self.state

        closed = self.ear_ema < self.ear_th
        self.sleep_ratio.update(closed)

        if closed:
            self.consec_closed += 1
            self.consec_open    = 0
        else:
            self.consec_open   += 1
            self.consec_closed  = 0

        if self.state == "AWAKE"  and self.consec_closed >= self.sleep_frames:
            self.state = "ASLEEP"
        elif self.state == "ASLEEP" and self.consec_open >= self.wake_frames:
            self.state = "AWAKE"

        return self.state

# -----------------------------
# Distraction Detector
# -----------------------------
class DistractionDetector:
    def __init__(self, fps=20):
        self.fps             = fps
        self.YAW_TH          = 20.0
        self.PITCH_DOWN_TH   = 15.0
        self.GAZE_TH         = 0.25
        self.LEFT_RIGHT_DUR  = 2.0
        self.DOWN_DUR        = 1.5
        self.left_right_counter = DurationCounter(fps)
        self.down_counter       = DurationCounter(fps)
        self.distract_ratio     = RatioTracker(window_sec=10, fps=fps)

    def process(self, lm, img):
        yaw, pitch = head_pose(lm, img.shape)
        gL = gaze_offset(lm, LEFT_EYE,  LEFT_IRIS)
        gR = gaze_offset(lm, RIGHT_EYE, RIGHT_IRIS)
        gaze_avg = (abs(gL) + abs(gR)) / 2.0

        head_off_lr = abs(yaw) > self.YAW_TH
        head_down   = pitch < -self.PITCH_DOWN_TH
        gaze_off    = gaze_avg > self.GAZE_TH

        self.left_right_counter.update(head_off_lr and not head_down)
        self.down_counter.update(head_down)

        distracted_frame = head_off_lr or head_down or gaze_off
        self.distract_ratio.update(distracted_frame)

        left_right_sec = self.left_right_counter.seconds()
        down_sec       = self.down_counter.seconds()

        flagged       = False
        high_priority = False

        if down_sec >= self.DOWN_DUR:
            flagged = True
            high_priority = True
        elif left_right_sec >= self.LEFT_RIGHT_DUR:
            flagged = True

        ratio = self.distract_ratio.ratio()
        if ratio > 0.45:
            flagged = True

        gaze_status = "CENTER"
        avg_offset  = (gL + gR) / 2.0
        if avg_offset < -0.18:
            gaze_status = "LEFT"
        elif avg_offset > 0.18:
            gaze_status = "RIGHT"
        if head_down or pitch < -12:
            gaze_status = "DOWN"

        return {
            "yaw": yaw, "pitch": pitch,
            "gaze": gaze_avg, "gL": gL, "gR": gR,
            "left_right_sec": left_right_sec,
            "down_sec": down_sec,
            "distract_ratio": ratio,
            "flagged": flagged,
            "high_priority": high_priority,
            "gaze_status": gaze_status
        }

# -----------------------------
# Main loop
# -----------------------------
def main():

    # ── Serial (Arduino Nano on COM6) ─────────────────────────────
    try:
        ser = serial.Serial('COM6', 9600, timeout=1)
        time.sleep(2)          # wait for Nano to reset after connection
        print("Serial connected on COM6")
    except Exception as e:
        print(f"Serial NOT connected: {e}")
        ser = None

    def send_signal(sig):
        if ser and ser.is_open:
            try:
                ser.write((sig + '\n').encode())
            except Exception as e:
                print(f"Serial write error: {e}")

    # ── Camera ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera — trying index 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("No camera found. Exiting.")
            return

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(cam_fps) if cam_fps and cam_fps > 1 else 20
    print(f"Camera OK | FPS: {fps}")

    # ── Detectors ─────────────────────────────────────────────────
    detector = FaceMeshDetector()
    sleep    = SleepDetector(fps=fps)
    distract = DistractionDetector(fps=fps)

    last_signal = ""

    print("Starting main loop — press Q in window to quit")

    while True:
        ok, img = cap.read()
        if not ok:
            print("Camera read failed")
            break

        faces = detector.find(img)

        if faces:
            lm = faces[0]
            draw_eye_overlay(img, lm)

            state = sleep.process(lm)
            d     = distract.process(lm, img)

            perclos           = sleep.sleep_ratio.ratio() * 100.0
            distraction_ratio = d["distract_ratio"] * 100.0

            risk_score = 0.6 * perclos + 0.4 * distraction_ratio
            risk_score = max(0.0, min(100.0, risk_score))

            if risk_score > 70:
                risk_level = ("HIGH",   (0,   0, 255))
            elif risk_score > 40:
                risk_level = ("MEDIUM", (0, 165, 255))
            else:
                risk_level = ("LOW",    (0, 255,   0))

            is_sustained = (d["high_priority"] or d["flagged"] or
                            (perclos > 30 and distraction_ratio > 25))

            # ── Signal decision ───────────────────────────────────
            # SLEEP   → Red LED + Blue LED + Buzzer  (driver asleep → slow down)
            # CRITICAL→ Red LED + Buzzer              (high risk / distraction)
            # NORMAL  → All OFF
            if state == "ASLEEP" or perclos > 30:
                signal = "SLEEP"
            elif is_sustained or risk_score > 60:
                signal = "CRITICAL"
            else:
                signal = "NORMAL"

            if signal != last_signal:
                send_signal(signal)
                last_signal = signal
            # ─────────────────────────────────────────────────────

            draw_metrics_panel(img, state, perclos, distraction_ratio,
                               d["gaze_status"], d["yaw"], d["pitch"],
                               risk_score, risk_level, is_sustained)

            # HUD — top right
            cv2.putText(img, f"FPS: {fps}",
                        (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            signal_color = (0, 0, 255) if signal == "CRITICAL" else \
                           (255, 165, 0) if signal == "SLEEP" else (0, 255, 0)
            cv2.putText(img, f"Signal: {signal}",
                        (330, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, signal_color, 2)

            nano_status = "Nano: OK" if ser else "Nano: --"
            cv2.putText(img, nano_status,
                        (330, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if ser else (0, 0, 255), 1)

        else:
            # No face detected → safe default
            if last_signal != "NORMAL":
                send_signal("NORMAL")
                last_signal = "NORMAL"
            cv2.putText(img, "STATE: NO FACE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Driver Monitoring System", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────────────
    send_signal("NORMAL")   # turn everything off on exit
    if ser:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")