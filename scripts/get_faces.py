import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# TODO: Cambiar nombre de directorio input he output
input_dir = '/home/branco/Documentos/Investigacion/SMDD/SMDD_release_train/os25k_bf_t'
output_dir = '/home/branco/Documentos/Investigacion/scripts/SMDD'
os.makedirs(output_dir, exist_ok=True)

FINAL_W = 260
FINAL_H = 360
TARGET_EYE_DIST = 90.0       
TARGET_EYE_X = FINAL_W // 2  
TARGET_EYE_Y = 160           

TOLERANCE_YAW_PITCH = 15.0
TOLERANCE_ROLL = 5.0

for filename in os.listdir(input_dir):
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
    
    faces = app.get(img)
    if len(faces) == 0:
        continue
    
    face = faces[0]
    pitch, yaw, roll = face.pose

    # Validar pose frontal 
    if abs(pitch) <= TOLERANCE_YAW_PITCH and abs(yaw) <= TOLERANCE_YAW_PITCH and abs(roll) <= TOLERANCE_ROLL:
        
        # Obtener landmarks clave de InsightFace (kps: 0=ojo izq, 1=ojo der, 3=comisura boca izq, 4=comisura boca der)
        ojo_izq = face.kps[0]
        ojo_der = face.kps[1]
        boca_izq = face.kps[3]
        boca_der = face.kps[4]
        
        # Análisis geométrico de la emoción (Neutral vs Sonrisa)
        dist_ojos = np.linalg.norm(ojo_der - ojo_izq)
        ancho_boca = np.linalg.norm(boca_der - boca_izq)
        
        # Ratio anatómico: Proporción entre el ancho de la boca y la distancia interocular.
        # En rostros neutrales, la boca es proporcionalmente más estrecha respecto a los ojos.
        # Cuando hay sonrisa, el ancho de la boca aumenta notablemente.
        ratio_sonrisa = ancho_boca / dist_ojos
        
        # Umbral empírico para detectar sonrisa (si el ratio supera 0.95, se considera sonrisa)
        UMBRAL_SONRISA = 0.95 
        
        if ratio_sonrisa > UMBRAL_SONRISA:
            print(f"❌ Descartada por sonrisa detectada visualmente ({filename}, ratio: {ratio_sonrisa:.2f})")
            continue
        
        # Escalado uniforme para alinear los ojos a 90 píxeles (Norma ICAO)
        centro_ojos_orig_x = (ojo_izq[0] + ojo_der[0]) / 2.0
        centro_ojos_orig_y = (ojo_izq[1] + ojo_der[1]) / 2.0
        
        scale = TARGET_EYE_DIST / dist_ojos
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        scaled_eye_x = int(centro_ojos_orig_x * scale)
        scaled_eye_y = int(centro_ojos_orig_y * scale)
        
        # Coordenadas del recorte estricto de 260x360
        start_x = scaled_eye_x - TARGET_EYE_X
        start_y = scaled_eye_y - TARGET_EYE_Y
        end_x = start_x + FINAL_W
        end_y = start_y + FINAL_H
        
        if start_x < 0 or start_y < 0 or end_x > img_scaled.shape[1] or end_y > img_scaled.shape[0]:
            print(f"⚠️ {filename} descartada (El recorte excede los márgenes).")
            continue

        rostro_final = img_scaled[start_y:end_y, start_x:end_x]
        
        out_path = os.path.join(output_dir, f"neutral_{filename}")
        cv2.imwrite(out_path, rostro_final)
        print(f"✅ Guardada (Neutral verificada por geometría): {filename}")
        
    else:
        print(f"❌ Descartada por mala pose: {filename}")

print("¡Proceso terminado!")