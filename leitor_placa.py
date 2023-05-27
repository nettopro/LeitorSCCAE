import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
#Coloca na imagem uma escala de cinza e enfatiza seus contornos.
img = cv2.imread('placa1.jpg')
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(cv2.cvtColor(cinza, cv2.COLOR_BGR2RGB))
filtrobi = cv2.bilateralFilter(cinza, 11, 17, 17) #Noise reduction
cantos = cv2.Canny(filtrobi, 30, 200) #Edge detection
#plt.imshow(cv2.cvtColor(cantos, cv2.COLOR_BGR2RGB))

#Aproxima o que o contorno devia ser (Nesse caso um de 4 lados para enfatizar a placa do veículo)
chave = cv2.findContours(cantos.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos = imutils.grab_contours(chave)
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

local = None
for contorno in contornos:
    aprox = cv2.approxPolyDP(contorno, 10, True)
    if len(aprox) == 4:
        local = aprox
        break
#Cria a mascara e reduz a imagem apenas a placa do veículo
mascara = np.zeros(cinza.shape, np.uint8)
nova_img = cv2.drawContours(mascara, [local], 0,255, -1)
nova_img = cv2.bitwise_and(img, img, mask=mascara)
#plt.imshow(cv2.cvtColor(nova_img, cv2.COLORBGR2RGB))

#Da "zoom" na placa do veiculo
(x,y) = np.where(mascara==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
placa = cinza[x1:x2+1, y1:y2+1]

#Realize e finaliza a leitura da placa.
leitor = easyocr.Reader(['en'])
resultado = leitor.readtext(placa)
print(resultado)

texto = resultado[0][-2]
fonte = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=texto, org=(aprox[0][0][0], aprox[1][0][1]+60), fontFace=fonte, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(aprox[0][0]), tuple(aprox[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()

