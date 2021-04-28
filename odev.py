import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

#Local threshold fonksiyonu
def localThreshold(src , dst , maxValue , blockSize , C):
    rowsNum = src.shape[0] #resmin satır sayısı
    colsNum = src.shape[1] #resmin kolon sayısı
        
    thresholdValue = 5 #threshold değeri
    
    #Görüntüyü integral görüntü'ye çevirmek için kullanılan kısım
    integralImage = np.zeros_like(src, dtype=np.uint32)
    for col in range(colsNum):
        for row in range(rowsNum):
            integralImage[row,col] = src[0:row,0:col].sum() 
    #Çevirme işlemi burada sona eriyor

    #Kolon döngüsü
    for col in range(colsNum):
        #Satır döngüsü
        for row in range(rowsNum):
            x0 = max(col - blockSize, 0) #x0 için max değeri döner
            x1 = min(col + blockSize, colsNum-1) #x1 için min değeri döner
            y0 = max(row - blockSize, 0) #y0 için max değeri döner
            y1 = min(row + blockSize, rowsNum-1) #y1 için min değeri döner
            
            count = (x1 - x0) * (y1 - y0)
            #Ağırlıklı toplam değerin hesaplaması burada yapılıyor
            weightedSum = integralImage[y0, x0] + integralImage[y1, x1] - integralImage[y0, x1] - integralImage[y1, x0] - C
            
            if src[row, col] * count < weightedSum * (100 - thresholdValue) / 100:
                #Resmin hesaplanan (Satır ve Kolonu * Count) değeri (Ağırlıklı Toplam * (100-thresholdValue = 95) / 100 )'den küçükse bu o satır ve kolon 0 yapılacak
                dst[row,col] = 0
            else:
                #Değilse 255'e eşit olacak (Binary=1)
                dst[row,col] = maxValue
                
    return dst #döngü bittikten sonra resmi geri dönüyoruz

#Local threshold için resim
imgForLocal = cv2.imread("img3.jpg" , 0)

blockSize = 9 #local threshold için gerekli blokların boyutu (bu değeri uygun gördüm)
destination = np.zeros_like(imgForLocal) #src dizisiyle aynı şekle ve türe sahip bir sıfır dizisi döndürür
destination = localThreshold(imgForLocal, destination, 255,  blockSize , 2) #local threshold fonksiyonum

#CV2 için resim
imgForCV2 = cv2.imread("img3.jpg" , 0)
#CV2 Thresholdlar
ret,th1 = cv2.threshold(imgForCV2,127,255,cv2.THRESH_BINARY) #global threshold
ret2,th2 = cv2.threshold(imgForCV2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #otsu threshold

"""Matplot İşlemleri"""
#Başlıkların listesi
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Otsu\'s Thresholding', 'Serkan Thresholding']
#Resimlerin listesi
images = [imgForCV2, th1, th2, destination]
#Resimleri ekranda görüntülemek için gerekli for döngüsü
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
#Resimleri göster
plt.show()