from pickletools import uint8
from os import environ
import cv2
import numpy as np
import  pywt
import matplotlib.image as mpimage
import matplotlib.pyplot as plt 
import sys  
from math import log10, sqrt

# np.set_printoptions(threshold=sys.maxsize)
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def dec2bin(n):
    return np.binary_repr(n, width=12)

def float2bin(number, places =5):
    whole, dec = str(number).split(".")
    whole = int(whole)
    dec = float ("."+dec)    
    res = bin(whole).lstrip("0b") + "."
    whole, dec = str('{0:.10f}'.format(dec*2)).split(".")
    for x in range(places):
        if(dec!=0):
            dec = float ("."+dec)
            res += whole
            whole, dec = str(float(decimal_converter(dec)) * 2).split(".")
    return res

def decimal_converter(num):
	while num >= 1:
		num /= 10
	return num

def bin2float(num):
    whole,dec=str(num).split(".")
    whole=whole[::-1]
    val1=0
    for i in range(0,len(whole)):
        if whole[i]=='1':
            val1+=pow(2,i)
            
    val2=0.0
    for i in range(0,len(dec)):
        if dec[i]=='1':
            val2+=pow(2,-(i+1))
    val=val1+val2

    return val

if __name__ == "__main__":
    suppress_qt_warnings()
    
#------------EMBEDDING-----------
# 1. Load the RGB cover image.
cover_img = cv2.imread(r'lena.jpg')
# cover_img = cv2.imread(r'face20x20.jpg')

# 2. Obtain the three different planes of the coverimage as three different gray scale images, R(R plane), G (G plane) and B (B plane). 
B_plane=cover_img[:,:,0] 
G_plane=cover_img[:,:,1]
R_plane=cover_img[:,:,2] 


# 3. Apply the 2-level Haar DWT on the R and G plane of the cover image.
coeff_R_plane=pywt.wavedec2(R_plane,'haar',level=2)
coeff_G_plane=pywt.wavedec2(G_plane,'haar',level=2)
LL2R,(HL2R,LH2R,HH2R),(HL1R,LH1R,HH1R)=coeff_R_plane
LL2G,(HL2G,LH2G,HH2G),(HL1G,LH1G,HH1G)=coeff_G_plane
print(HL2R)
row_R=LL2R.shape[0]
col_R=LL2R.shape[1]
arrLL2R=np.zeros((row_R,col_R))
arrLL2G=np.zeros((row_R,col_R))
arrHL2R=np.zeros((row_R,col_R))
# print(LL2R)
for i in range(row_R):
    for j in range(col_R):
        arrLL2R[i,j] =float2bin(LL2R[i,j])
        arrLL2G[i,j] =float2bin(LL2G[i,j])
        arrHL2R[i,j] =dec2bin(int(HL2R[i,j]))
        
# Plotting all wavelet coefficients as one matrix
# R Plane
arr_R,coeff_slice_R_plane=pywt.coeffs_to_array(coeff_R_plane)
plt.figure(figsize=(20,20))
plt.imshow(arr_R)
plt.title('All Wavelet Coeff. upto level 2', fontsize=10)
plt.show()

# 4. Load the RGB secret image (size of secret image (m) <= 1/4th of size of cover image).
secret_img = cv2.imread(r'dog.jpg')

# 5. Obtain the three different planes of the secret image in r (R plane), g (G plane) and b (B plane).
b_plane=secret_img[:,:,0] 
g_plane=secret_img[:,:,1] 
r_plane=secret_img[:,:,2] 

# 6. Concealing the secret plane in cover plane is done as:
# i. Select one of the appropriate bands among level DWT bands of the cover plane as‘ni’
# embed secret R plane in LL2 band of R plane of cover image : n1=LL2R
# embed secret G plane in LL2 band of G plane of cover image : n2=LL2G
# embed secret B plane in HL2 band of G plane of cover image : n3=HL2R

# ii. For each of the m*m coefficient of the LL2 band, the 'x MSB' are inserted in place of the 'x LSB' (i.e. 5 bits) of the band coefficient.

m=r_plane.shape[0]
arr_r=np.zeros((m,m)).astype(int)
arr_g=np.zeros((m,m)).astype(int)
arr_b=np.zeros((m,m)).astype(int)

#conert pixel value to binary
for i in range(m):
    for j in range (m):
        arr_r[i,j]=dec2bin(r_plane[i,j].astype(int))
        arr_g[i,j]=dec2bin(g_plane[i,j].astype(int))
        arr_b[i,j]=dec2bin(b_plane[i,j].astype(int))
        
# x MSB' are inserted in place of the 'x LSB' (i.e. 5 bits) of the band coefficient
for i in range (m):
    for j in range(m):
        x=str(arr_r[i,j]).rjust(8,'0')[:5] #secret image r plane
        y=str(arrLL2R[i,j]).rjust(16,'0')
        z=y[:len(y)-5]+x[:3] +'.'+x[3:]
        LL2R[i,j]=bin2float(z)
        
        x=str(arr_g[i,j]).rjust(8,'0')[:5] #secret image g plane
        y=str(arrLL2G[i,j]).rjust(16,'0')
        z=y[:len(y)-5]+x[:3]+'.'+x[3:]
        LL2G[i,j]=bin2float(z)
        
        x=str(arr_b[i,j]).rjust(8,'0')[:5] #secret image b plane
        y=str(arrHL2R[i,j]).rjust(16,'0')
        z=y[:len(y)-5]+x
        HL2R[i,j]=int(z,2)
        
coeff_R_plane=[LL2R,(HL2R,LH2R,HH2R),(HL1R,LH1R,HH1R)]
coeff_G_plane=[LL2G,(HL2G,LH2G,HH2G),(HL1G,LH1G,HH1G)]


# iii. Apply 2-level inverse DWT (IDWT) operation.
R=np.uint8(pywt.waverec2(coeff_R_plane,'haar'))
G=np.uint8(pywt.waverec2(coeff_G_plane,'haar'))

# The stego plane is obtained. 
merge=cv2.merge([B_plane,G,R])
cv2.imshow('Cover Image', cover_img)
cv2.imshow('Stego Image',merge)

# 7. The above step is repeated, for hiding each of the three planes of the secret image in the band coefficients of, one of the cover image plane.(n1, n2, n3 stores the name of the selected band to be modified for embedding R plane, G plane and B plane of secret image respectively).
print("PSNR = ",PSNR(cover_img,merge),"dB")
cv2.waitKey(0)