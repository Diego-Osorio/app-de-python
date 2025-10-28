import streamlit as st
import cv2
import numpy as np
import sounddevice as sd
import librosa
import math
import time

st.set_page_config(page_title="Afinador de Percusión Pro", layout="wide")

# ---------------------------
# 1️⃣ Utilidades de visión
# ---------------------------

def detect_drum_head_and_tuners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150,
                               param1=80, param2=40, minRadius=50, maxRadius=700)
    head = None
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        head = max(circles, key=lambda c: c[2])
    if head is None:
        return None, []
    
    x, y, r = head
    # Detección simple de tensores (puntos brillantes cerca del borde)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), int(r * 1.1), 255, thickness=20)
    masked = cv2.bitwise_and(blurred, mask)
    _, thresh = cv2.threshold(masked, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tuners = []
    for c in contours:
        (tx, ty), rad = cv2.minEnclosingCircle(c)
        if 4 < rad < 40:
            tuners.append((int(tx), int(ty)))
    return (x, y, r), tuners

def draw_overlay(frame, head, tuners):
    vis = frame.copy()
    if head:
        cv2.circle(vis, (head[0], head[1]), head[2], (0,255,0), 2)
    for i, (tx, ty) in enumerate(tuners):
        cv2.circle(vis, (tx, ty), 6, (0,0,255), -1)
        cv2.putText(vis, f"T{i+1}", (tx+8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return vis

# ---------------------------
# 2️⃣ Audio
# ---------------------------

def record_audio(duration=0.6, fs=44100):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def estimate_freq(y, fs=44100, fmin=50, fmax=2000):
    try:
        y = y - np.mean(y)
        f_lib = librosa.yin(y, fmin=fmin, fmax=fmax, sr=fs)
        f_med = np.median(f_lib[np.isfinite(f_lib)])
        return float(f_med)
    except Exception:
        return 0.0

# ---------------------------
# 3️⃣ Clasificación del instrumento
# ---------------------------

def classify_instrument(radius):
    """Clasifica según tamaño detectado (en píxeles relativos a cámara)."""
    if radius < 100:
        return "Caja"
    elif radius < 180:
        return "Repique"
    elif radius < 260:
        return "Zurdo"
    else:
        return "Bombo"

# ---------------------------
# 4️⃣ Aplicación principal
# ---------------------------

def main():
    st.title("🥁 Afinador Pro de Percusión en Vivo")
    st.markdown("Detecta automáticamente el tipo de instrumento, los tensores y su afinación.")

    # Variables de estado
    if "stage" not in st.session_state:
        st.session_state.stage = "detect_instrument"
        st.session_state.tuners_detected = []
        st.session_state.instrument = None

    run = st.checkbox("Activar cámara", value=False)
    frame_placeholder = st.empty()
    info = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        time.sleep(1.0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("No se puede acceder a la cámara.")
                break

            frame = cv2.flip(frame, 1)
            head, tuners = detect_drum_head_and_tuners(frame)

            # 1️⃣ Paso: Detectar instrumento
            if st.session_state.stage == "detect_instrument":
                if head:
                    instr = classify_instrument(head[2])
                    st.session_state.instrument = instr
                    st.session_state.stage = "detect_tuners"
                    st.toast(f"✅ Instrumento detectado: {instr}")
                    time.sleep(1.5)

            # 2️⃣ Paso: Detectar tensores
            elif st.session_state.stage == "detect_tuners":
                if head and len(tuners) >= 3:
                    st.session_state.tuners_detected = tuners
                    st.session_state.stage = "tuning"
                    st.toast(f"🎯 Detectados {len(tuners)} tensores.")
                    time.sleep(1.5)

            vis = draw_overlay(frame, head, tuners)
            frame_placeholder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB")

            if st.session_state.stage == "tuning":
                info.info(f"✅ Instrumento: {st.session_state.instrument} — Tensores: {len(st.session_state.tuners_detected)}")
                cap.release()
                break

        cap.release()

    # 3️⃣ Paso: Afinación por tensor
    if st.session_state.stage == "tuning":
        st.subheader("🎵 Afinación por tensor")
        st.write("Golpea cada tensor cuando se te indique:")

        num_tuners = len(st.session_state.tuners_detected)
        results = []
        for i in range(num_tuners):
            if st.button(f"🎤 Grabar Tensor {i+1}"):
                y = record_audio()
                f = estimate_freq(y)
                st.write(f"T{i+1}: {f:.1f} Hz")
                results.append(f)

        if len(results) == num_tuners and num_tuners > 0:
            f_avg = np.mean(results)
            diffs = [abs(f - f_avg) for f in results]
            if all(d < 5 for d in diffs):
                st.success("✅ El instrumento está bien afinado.")
            else:
                st.warning("⚠️ Algunos tensores necesitan ajuste.")

if __name__ == "__main__":
    main()
