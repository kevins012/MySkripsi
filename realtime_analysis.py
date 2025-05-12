
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import tensorflow as tf
def normalisasi2(data):
    vektor = [(d-data[0] )for d in data[1:] ]
    return vektor
def scale_points(points, new_x_max):
    """
    Melakukan transformasi skala pada kumpulan titik berdasarkan nilai maksimum baru untuk sumbu X.
    
    Parameters:
        points (numpy.ndarray): Array 2D berisi koordinat titik, dengan kolom pertama sebagai X dan kedua sebagai Y.
        new_x_max (float): Nilai maksimum baru untuk sumbu X setelah transformasi.
        
    Returns:
        numpy.ndarray: Array 2D dari titik yang telah ditransformasi.
    """
    # Nilai maksimum awal untuk X
    x_max_original = np.max(points[:, 0])
    
    # Hitung skala
    scale = new_x_max / x_max_original
    
    # Transformasi titik berdasarkan skala
    transformed_points = (points * scale)
    
    return transformed_points[:,0],transformed_points[:,1]
    
    
def trim_sequence(seq, target_len=20):
    if len(seq) <= target_len:
        return list(seq)

    keep_first = seq[0]
    keep_last = seq[-1]
    middle = list(seq)[1:-1]

    total_to_keep = target_len - 2
    step = len(middle) / total_to_keep

    trimmed_middle = [middle[int(i * step)] for i in range(total_to_keep)]

    return [keep_first] + trimmed_middle + [keep_last]
def normalisasi(data):
    dmin, dmax = np.min(data), np.max(data)
    return (data - dmin) 
cols_X = sorted([1,9,10,12,13,16,17,20,4,6,8,11,16])
cols_Y = sorted([2,3,4,7,9,10,11,12,15,19,20,16,17])
# cols_Z = 
cols_RX= [4,6,8,10,12,16,19,20]
cols_RY= [4,6,8,10,12,16,19,20]

# cols_Z= [4,7,11,15]
cols_Z = [5,8,12,20]

s=''
# ==================== SETUP ====================
import pickle

with open('csv/label map/dinamic.pkl', 'rb') as f:
    label_map = pickle.load(f)
# Load model
model = tf.keras.models.load_model("model/dinamic/1.h5")

# Cek input shape model
input_shape = model.input_shape  # (None, 35, 21)
frame_count = input_shape[1]     # 35
feature_per_frame = input_shape[2]  # 21

# Mapping label

# cols_X = [5,1,9,10,12,13,15,16,17,20,4,6,7,8,11,16,17]
# cols_Y = [2,3,4,7,9,10,11,12,15,19,20,16,17]

# cols_X = sorted([1,9,10,12,13,16,17,20,4,6,8,11,16])


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# Inisialisasi deque untuk menyimpan fitur dari frame sebelumnya
sequence = deque(maxlen=30)
a=0
# Buka video
cap = cv2.VideoCapture('video2/test/1.mp4')
if not cap.isOpened():
    print("Gagal membuka video.")
    exit()
k=False
# ==================== LOOP VIDEO FRAME ====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    label_text = "Menunggu 35 frame..."

    if result.hand_landmarks:
        a+=1
        for hand in result.hand_landmarks:
            
            features = []
            
                
            nilai_X = np.array([landmark.x for landmark in hand])
            nilai_Y = np.array([landmark.y for landmark in hand])
            nilai_Z = np.array([landmark.z for landmark in hand])[cols_Z]
        
            try:
                vektorX = nilai_X[cols_RX] - X_before
                vektorY = nilai_Y[cols_RY] - Y_before
            except:
                vektorX = [0]*8
                vektorY = [0]*8
            X_before = nilai_X[cols_RX]
            Y_before = nilai_Y[cols_RY]

            # Normalisasi dan scaling
            newX = normalisasi(nilai_X)
            newY = normalisasi(nilai_Y)
            newXY = np.column_stack((newX, newY) )  # Simpan dalam list koordinat int
            
            newX , newY = scale_points(newXY,1)
            features = np.concatenate([
                np.array(newX)[cols_X],
                np.array(newY)[cols_Y],
                np.array(nilai_Z),
                np.array(vektorX),
                np.array(vektorY)
            ])

            if len(features) == feature_per_frame:
                sequence.append(features)
        
            
    
    # else:
    #     features = [np.nan]*(21 + 21 + len(cols_Z2))
    #     sequence.append(features)



    if len(sequence) ==30:
     
        trimmed = trim_sequence(sequence, target_len=25)
        trimmed_array = np.array(trimmed)  # Ubah deque/list menjadi array 2D




    
        input_data = np.array(trimmed).reshape(1, 25, len(trimmed[0]))
   
        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if  (confidence > 0.9):
            label_text = f"{label_map[predicted_class]} ({confidence:.2f})"
            # s=label_map[predicted_class]
            print(label_text)
    
    # ==================== TAMPILKAN HASIL ====================
    cv2.putText(frame, label_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.imshow("Prediksi Gesture", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
print(a)
# ==================== SELESAI ====================
cap.release()
cv2.destroyAllWindows()
