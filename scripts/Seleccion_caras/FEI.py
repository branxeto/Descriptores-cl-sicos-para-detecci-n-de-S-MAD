import cv2
import os

# Directorios de entrada y salida
input_dir = '/home/branco/Documentos/Investigacion/FEI/frontalimages_manuallyaligned_part2'
output_dir = '/home/branco/Documentos/Investigacion/scripts/FEI'
os.makedirs(output_dir, exist_ok=True)

# Define el formato o patrón que deben cumplir los nombres de archivo.
# Por ejemplo, aquí filtramos para que el archivo obligatoriamente contenga "neutral" 
# o comience con un ID específico (puedes cambiar esta condición a tu gusto).


contador_guardadas = 0

for filename in os.listdir(input_dir):
    # 1. Verificar extensiones de imagen válidas de entrada
    if not filename.lower().endswith(('a.jpg')):
        continue
    
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️ Error al leer la imagen: {filename}")
        continue
    
    # 3. Extraer el nombre base sin su extensión original (ej. "imagen.png" -> "imagen")
    nombre_base, _ = os.path.splitext(filename)
    
    # 4. Forzar la extensión de salida a .jpg
    nuevo_nombre_jpg = f"{nombre_base}.jpg"
    out_path = os.path.join(output_dir, nuevo_nombre_jpg)
    
    # 5. Guardar la imagen explícitamente en formato JPG (con calidad opcional de 95)
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    contador_guardadas += 1
    print(f"✅ Guardada en formato .jpg: {nuevo_nombre_jpg}")

print(f"\n¡Proceso finalizado! Se procesaron y guardaron {contador_guardadas} imágenes con el formato correcto.")