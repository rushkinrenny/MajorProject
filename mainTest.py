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
    # if n<=256 and  n>=0:
    #     return '{:08b}'.format(n)
    return np.binary_repr(n, width=12)
    # bnr = bin(n).replace('0b','')
    # x = bnr[::-1] #this reverses an array
    # while len(x) < 8:
    #     x += '0'
    # bnr = x[::-1].lstrip('.')
    # # print(bnr)
    # return bnr

# def float2bin(number, places = 3):
    # print(str(number))
    whole, dec = str(number).split(".")
    whole = int(whole)
    dec = int (dec)
    # print(decimal_converter(dec))
    res = bin(whole).lstrip("0b") + "."
    for x in range(places):
        if(dec!=0):
            whole, dec = str(float(decimal_converter(dec)) * 2).split(".")
            dec = int(dec)
            res += whole
    return res

def float2bin(number, places =10):
    whole, dec = str(number).split(".")
    whole = int(whole)
    dec = float ("."+dec)
    # print(dec)
    # print(decimal_converter(dec))
    
    res = bin(whole).lstrip("0b") + "."
    # print(str(dec * 2))
    whole, dec = str('{0:.10f}'.format(dec*2)).split(".")
    # print(whole,dec)
    for x in range(places):
        if(dec!=0):
            dec = float ("."+dec)
            res += whole
            whole, dec = str(float(decimal_converter(dec)) * 2).split(".")
    return res

def decimal_converter(num):
    while num>=1:
        num/=10
    # print(num)
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
# print(cover_img)
cv2.imshow("cover Image", cover_img)

# 2. Obtain the three different planes of the coverimage as three different gray scale images, R(R plane), G (G plane) and B (B plane). 
B_plane=cover_img[:,:,0] 
G_plane=cover_img[:,:,1] 
R_plane=cover_img[:,:,2] 
# merge=cv2.merge([B_plane,G_plane,R_plane])
# cv2.imshow('merge',merge)

# cv2.imshow('blue plane image',B_plane)
# cv2.imshow('green plane image',G_plane)
# cv2.imshow('Red plane image',R_plane)
# cv2.waitKey(0)

# 3. Apply the 2-level Haar DWT on the R and G plane of the cover image.
coeff_R_plane=pywt.wavedec2(R_plane,'haar',level=2)
coeff_G_plane=pywt.wavedec2(G_plane,'haar',level=2)
LL2R,(HL2R,LH2R,HH2R),(HL1R,LH1R,HH1R)=coeff_R_plane
LL2G,(HL2G,LH2G,HH2G),(HL1G,LH1G,HH1G)=coeff_G_plane
# cv2.imshow("after idwt", np.uint8(pywt.waverec2(coeff_R_plane,'haar','zero')))
print("Before sum",np.sum(coeff_R_plane[0]))
# LL2R=LL2R.astype(int)
# LL2G=LL2G.astype(int)
# HL2R=HL2R.astype(int)
# print(coeff_R_plane)
# print(LL2R.astype(int))
# print(coeff_R_plane)
row_R=LL2R.shape[0]
col_R=LL2R.shape[1]
arrLL2R=np.zeros((row_R,col_R))
arrLL2G=np.zeros((row_R,col_R))
arrHL2R=np.zeros((row_R,col_R))
# print(LL2R)
for i in range(row_R):
    for j in range(col_R):
        # print(LL2R[i,j])
        x =float2bin(LL2R[i,j])
        # print(x)
        arrLL2R[i,j] =x
        y =float2bin(LL2G[i,j])
        arrLL2G[i,j] =y
        z =dec2bin(int(HL2R[i,j]))
        arrHL2R[i,j] =z
        
        # print(x)
print(arrLL2R)
print(arrLL2R[0][0])

# Plotting all wavelet coefficients as one matrix
#R Plane
# arr_R,coeff_slice_R_plane=pywt.coeffs_to_array(coeff_R_plane)
# plt.figure(figsize=(20,20))
# plt.imshow(arr_R)
# plt.title('All Wavelet Coeff. upto level 2', fontsize=10)

#G Plane
# arr_G,coeff_slice_G_plane=pywt.coeffs_to_array(coeff_G_plane)
# plt.figure(figsize=(20,20))
# plt.imshow(arr_G)
# plt.title('All Wavelet Coeff. upto level 2', fontsize=10)
# plt.show()


# cv2.imshow('LL2R',LL2R)
# cv2.imshow('HL2R',HL2R)
# cv2.imshow('LH2R',LH2R)
# cv2.imshow('HH2R',HH2R)

# 4. Load the RGB secret image (size of secret image (m) <= 1/4th of size of cover image).
secret_img = cv2.imread(r'dog.jpg')
# secret_img = cv2.imread(r'face20x20.jpg')

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

# print(r_plane)
for i in range(m):
    for j in range (m):
        # print(r_plane[i,j])
        x=dec2bin(r_plane[i,j].astype(int))
        arr_r[i,j]=x
        y=dec2bin(g_plane[i,j].astype(int))
        arr_g[i,j]=y
        z=dec2bin(b_plane[i,j].astype(int))
        arr_b[i,j]=z
# print(LH2R)
# print(arr)


# print(arrLL2R)
for i in range (m):
    for j in range(m):
        x=str(arr_r[i,j]).rjust(8,'0')[:5] #secret image
        # print("x-> ",x)
        y=str(arrLL2R[i,j]).rjust(16,'0')
        # print("y-> ",y)
        # print("y[:len(y)-5]-> ",y[:len(y)-5])
        z=y[:len(y)-5]+x[:3] +'.'+x[3:]
        # print("z-> ",z)
        # print(float(literal_eval(z)))
        # print(x,z,int(z,2))
        # LL2R[i,j]=float(literal_eval(z))
        # print(z)
        LL2R[i,j]=bin2float(z)
        
        x=str(arr_g[i,j]).rjust(8,'0')[:5] #secret image
        y=str(arrLL2G[i,j]).rjust(16,'0')
        z=y[:len(y)-5]+x[:3]+'.'+x[3:]
        # print(x,z,int(z,2))
        # LL2G[i,j]=float(literal_eval(z))
        LL2G[i,j]=bin2float(z)
        
        x=str(arr_b[i,j]).rjust(8,'0')[:5] #secret image
        y=str(arrHL2R[i,j]).rjust(16,'0')
        z=y[:len(y)-5]+x
        # print(x,z,int(z,2))
        HL2R[i,j]=int(z,2)
# print((og_coeff_r==coeff_R_plane))
coeff_R_plane=[LL2R,(HL2R,LH2R,HH2R),(HL1R,LH1R,HH1R)]
coeff_G_plane=[LL2G,(HL2G,LH2G,HH2G),(HL1G,LH1G,HH1G)]
print("Before sum",np.sum(coeff_R_plane[0]))
# iii. Apply 2-level inverse DWT (IDWT) operation
# cv2.imshow("After IDWT R plane", R_plane)
R=np.uint8(pywt.waverec2(coeff_R_plane,'haar'))
G=np.uint8(pywt.waverec2(coeff_G_plane,'haar'))
merge=cv2.merge([B_plane,G,R])

cv2.imshow('Stego Image',merge)
# # G=pywt.waverec2(coeff_G_plane,'haar')
# cv2.imshow("Rplane", R_plane)
# cv2.imshow("after idwt R_plance", np.uint8(pywt.waverec2(coeff_R_plane,'haar','zero')))



# The stego plane is obtained. 

# 7. The above step is repeated, for hiding each of the three planes of the secret image in the band coefficients of, one of the cover image plane.(n1, n2, n3 stores the name of the selected band to be modified for embedding R plane, G plane and B plane of secret image respectively).
print(PSNR(cover_img,merge))
cv2.waitKey(0)