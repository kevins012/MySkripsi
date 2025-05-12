import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Warna dan ukuran font untuk teks tampilan
MARGIN = 10  # pixels
FONT_SIZE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)  # Warna hijau untuk teks koordinat
LANDMARK_TEXT_COLOR = (255, 0, 0)  # Warna biru untuk teks landmark
LINE_COLOR = (0, 255, 255)  # Warna garis (kuning-hijau)

# Buka video
cap_video = cv2.VideoCapture(0)

# Pastikan video dapat dibuka
if not cap_video.isOpened():
    print("Gagal membuka video.")
    exit()

# STEP 2: Buat objek HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE  # Gunakan mode gambar
)
detector = vision.HandLandmarker.create_from_options(options)

print("Menampilkan video dengan deteksi tangan. Tekan 's' untuk pause, 'q' untuk keluar.")

# Variabel global untuk koordinat yang ditampilkan saat kursor diarahkan
hovered_text = ""
hovered_position = (0, 0)
paused = False  # Status video (berjalan atau pause)

# Buat jendela sebelum loop untuk menghindari error OpenCV
cv2.namedWindow("Rekaman Video dengan Hand Detection")

# Inisialisasi data untuk plotting
x_time = []  # Sumbu X (waktu berjalan)
y_landmark = []  # Sumbu Y (landmark X1)
start_time = time.time()

fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-', lw=2)
ax.set_xlim(0, 10)  # Sumbu X tetap 10 detik terakhir
ax.set_ylim(0, 1)  # Rentang nilai X landmark normalisasi
ax.set_xlabel("Waktu (s)")
ax.set_ylabel("Landmark X1")
ax.set_title("Real-time Hand Landmark X1 Plot")

def update_plot(frame):
    ax.set_xlim(max(0, time.time() - start_time - 10), time.time() - start_time)
    line.set_data(x_time, y_landmark)
    return line,

ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=True)

# Fungsi untuk menangkap event mouse
def mouse_callback(event, x, y, flags, param):
    global hovered_text, hovered_position
    if event == cv2.EVENT_MOUSEMOVE:  # Jika mouse bergerak
        hovered_text = ""
        for idx, (lx, ly, lz) in enumerate(param):
            px, py = int(lx * frame.shape[1]), int(ly * frame.shape[0])  # Skala ke ukuran gambar
            if abs(x - px) < 10 and abs(y - py) < 10:  # Jika kursor dekat titik
                hovered_text = f"ID: {idx} | X:{lx:.2f}, Y:{ly:.2f}, Z:{lz:.2f}"
                hovered_position = (px, py)
                break

cv2.setMouseCallback("Rekaman Video dengan Hand Detection", mouse_callback)

while True:
    if not paused:
        ret, frame = cap_video.read()
        
        if not ret:  
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Jika video habis, ulang dari awal
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        all_landmarks = []  # Simpan semua landmark untuk digunakan dalam mouse event
        if detection_result and detection_result.hand_landmarks:
            for hand in detection_result.hand_landmarks:
                x1 = hand[0].x if hand else None
                if x1 is not None:
                    x_time.append(time.time() - start_time)
                    y_landmark.append(x1)
                    x_time = x_time[-100:]
                    y_landmark = y_landmark[-100:]

                connections = mp.solutions.hands.HAND_CONNECTIONS  # Sambungan antar titik
                landmark_points = []  # Menyimpan titik untuk koneksi garis

                for idx, landmark in enumerate(hand):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    all_landmarks.append((landmark.x, landmark.y, landmark.z))  # Simpan koordinat
                    landmark_points.append((x, y))  # Simpan titik untuk menggambar garis

                # Menghubungkan titik dengan garis sesuai koneksi tangan
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                        cv2.line(frame, landmark_points[start_idx], landmark_points[end_idx], LINE_COLOR, 2)

    if hovered_text:
        cv2.putText(frame, hovered_text, hovered_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    cv2.setMouseCallback("Rekaman Video dengan Hand Detection", mouse_callback, all_landmarks)
    cv2.imshow("Rekaman Video dengan Hand Detection", frame)
    plt.pause(0.01)
    plt.draw()

    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        paused = not paused
        if paused:
            print("Video dijeda. Tekan 's' untuk melanjutkan.")
        else:
            print("Melanjutkan video...")
    elif key == ord('d'):
        paused = True
        print("Video dijeda. Tekan 's' untuk melanjutkan.")

cap_video.release()
cv2.destroyAllWindows()
plt.close()