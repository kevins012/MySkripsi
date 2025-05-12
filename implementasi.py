import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import threading
import time
import pickle
from collections import deque
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
output_mlp = ['cepat1','paham1','tidak1','lihat1','menang1','z','a','i','k']
output_mlp2 = ['cepat2','paham2','tidak1','lihat2','menang2','z','10_2','j2','k']
output_lstm = ['cepat','paham','tidak','lihat','menang','z','10','j','k']
# ==================== NORMALISASI DAN UTIL ================q====
def scale_points(points, new_x_max):
    x_max_original = np.max(points[:, 0])
    scale = new_x_max / x_max_original
    transformed_points = points * scale
    return transformed_points[:, 0], transformed_points[:, 1]

def normalisasi(data):
    dmin, dmax = np.min(data), np.max(data)
    return (data - dmin)

def trim_sequence(seq, target_len=25):
    if len(seq) <= target_len:
        return list(seq)
    keep_first = seq[0]
    keep_last = seq[-1]
    middle = list(seq)[1:-1]
    step = len(middle) / (target_len - 2)
    trimmed_middle = [middle[int(i * step)] for i in range(target_len - 2)]
    return [keep_first] + trimmed_middle + [keep_last]

# ==================== LOAD MODEL DAN KONFIG ====================
with open('csv/label map/model_2.pkl', 'rb') as f:
    label_map = pickle.load(f)

with open('csv/label map/static.pkl', 'rb') as f:
    label_map_static = pickle.load(f)

model_dynamic = tf.keras.models.load_model(
    "model/dinamic/model_12.h5",
    custom_objects={"LSTM": LSTM, "Bidirectional": Bidirectional, "Dense": Dense}
)
model_static = tf.keras.models.load_model("model/static/model_1.h5")

frame_count = model_dynamic.input_shape[1]
feature_per_frame = model_dynamic.input_shape[2]

# Dinamis
cols_X = sorted([1,9,10,12,13,16,17,20,4,6,8,11,16])
cols_Y = sorted([2,3,4,7,9,10,11,12,15,19,20,16,17])
cols_Z = [5,8,12,20]
cols_RX = [4,6,8,10,12,16,19,20]
cols_RY = [4,6,8,10,12,16,19,20]

# Statis
column_numbersX = sorted([5,1,9,10,12,13,15,16,17,20,4,6,7,8,11,16,17])
column_numbersY = sorted([2,3,4,7,9,10,11,12,15,19,20,16,17])
column_numbersZ = [5,8,12,20]

# ==================== SINKRONISASI ====================
pred_static = "Menunggu..."
pred_dynamic = "Menunggu..."
lock_static = threading.Lock()
lock_dynamic = threading.Lock()

# ==================== THREADING VIDEO ====================
class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()

# ==================== SETUP MEDIAPIPE ====================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# ==================== THREAD STATIC MODEL ====================
def static_prediction_thread(vc):
    global pred_static
    last_prediction = None
    final_text = ""
    min_confidence = 0.92
    stability_threshold = 0.02
    stable_frames_counter = 0
    stable_frames_required = 3
    nowX, nowY = 0, 0

    while vc.running:
        ret, frame = vc.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            nilai_X = np.array([lm.x for lm in hand])
            nilai_Y = np.array([lm.y for lm in hand])
            nilai_Z = np.array([lm.z for lm in hand])[column_numbersZ]

            stabilX, stabilY = abs(nilai_X[5]-nowX), abs(nilai_Y[5]-nowY)
            nowX, nowY = nilai_X[5], nilai_Y[5]

            newX = normalisasi(nilai_X)
            newY = normalisasi(nilai_Y)
            newXY = np.column_stack((newX, newY))
            newX, newY = scale_points(newXY, 10)

            fiturX = newX[column_numbersX]
            fiturY = newY[column_numbersY]
            features = np.concatenate((fiturX, fiturY, nilai_Z)).astype(np.float32)
            input_data = np.expand_dims(features, axis=0)

            if stabilX < stability_threshold and stabilY < stability_threshold:
                stable_frames_counter += 1
            else:
                stable_frames_counter = 0

            if stable_frames_counter >= stable_frames_required:
                prediction = model_static.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                with lock_static:
                    pred_static = f"{label_map_static[predicted_class]} ({confidence:.2f})" if confidence >= min_confidence else "Confidence rendah"
                    if label_map_static[predicted_class] in output_mlp:
                        p1 = output_mlp.index(label_map_static[predicted_class])

# ==================== THREAD DYNAMIC MODEL ====================
def dynamic_prediction_thread(vc):
    global pred_dynamic
    sequence = deque(maxlen=35)
    X_before = Y_before = None

    while vc.running:
        ret, frame = vc.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            nilai_X = np.array([lm.x for lm in hand])
            nilai_Y = np.array([lm.y for lm in hand])
            nilai_Z = np.array([lm.z for lm in hand])[cols_Z]

            try:
                vektorX = nilai_X[cols_RX] - X_before
                vektorY = nilai_Y[cols_RY] - Y_before
            except:
                vektorX = [0] * len(cols_RX)
                vektorY = [0] * len(cols_RY)

            X_before = nilai_X[cols_RX]
            Y_before = nilai_Y[cols_RY]

            newX = normalisasi(nilai_X)
            newY = normalisasi(nilai_Y)
            newXY = np.column_stack((newX, newY))
            newX, newY = scale_points(newXY, 10)

            features = np.concatenate([
                np.array(newX)[cols_X],
                np.array(newY)[cols_Y],
                np.array(nilai_Z),
                np.array(vektorX),
                np.array(vektorY)
            ])

            if len(features) == feature_per_frame:
                sequence.append(features)

            if len(sequence) == 35:
                trimmed = trim_sequence(sequence, target_len=25)
                input_data = np.array(trimmed).reshape(1, 25, feature_per_frame)
                prediction = model_dynamic.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                with lock_dynamic:
                    pred_dynamic = f"{label_map[predicted_class]} ({confidence:.2f})" if confidence >= 0.7 else "Confidence rendah"

# ==================== MAIN ====================
vc = VideoCaptureThread()
threading.Thread(target=static_prediction_thread, args=(vc,), daemon=True).start()
threading.Thread(target=dynamic_prediction_thread, args=(vc,), daemon=True).start()

time.sleep(1)

while True:
    ret, frame = vc.read()
    if not ret:
        break

    with lock_static:
        static_text = pred_static
    with lock_dynamic:
        dynamic_text = pred_dynamic

    cv2.putText(frame, f"Prediksi Dinamis: {dynamic_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Prediksi Statis : {static_text}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Prediksi Gesture Gabungan", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()


