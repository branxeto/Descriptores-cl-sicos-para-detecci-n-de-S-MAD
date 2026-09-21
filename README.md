# Descriptores clásicos para detección de S-MAD. Extraer características con LBP, BSIF, HOG entre otros.


## Requisitos
Se necesita tener instalado:
- OpenCV
- Numpy
- Insightface

## Creación de dataset

Las bases de datos que se utilizaron para esta investigación fueron:

*FRLL*
https://zenodo.org/records/4415159
*ASML*
https://omen.cs.uni-magdeburg.de/disclaimer/index.php
*FEI*
https://fei.edu.br/~cet/facedatabase.html
*SMDD*
https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset
*Defacto-face*
https://www.kaggle.com/datasets/defactodataset/defactoface?resource=download-directory&select=reference

Cada una de estas tiene sus propias cualidades, por lo que se debieron de normalizar para cumplieran con los mismos estandares, los cuales son:

- La distancia entre los centros de las pupilas tiene que ser de 90px.
- La resolución es de 260x360px.

A su ves las imagenes fueron alineadas para facilitar su posterior transformación. Este paso se realizo de la siguiente manera:

- Obtener los landmarks con libreria Insightface. La cantidad de landmarks que se obtienen son 68, de los cuales del 36 a 41 corresponden al ojo izquierdo, los puntos 42 a 47 corresponden al ojo derecho.
- Se calcula el centro de la pupila de cada ojo.
- Se escalan las imagenes para que cada cara tenga una distancia de 90 pixeles entre los centros de las pupilas.
- Se identifica la emoción de la persona a través del ancho de la boca, para identificar si esta con emoción Neutra. Para identificar si la persona sonrie o esta neutra se utiliza la siguiente formula $Ancho Boca/DistanciaOjos>0.95$.
- Obteniendo la ubicación de los ojos en pixeles, se recorta la imagen para que tenga la resolución de 160x360px.

