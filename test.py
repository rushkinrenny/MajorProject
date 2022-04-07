# a=[[1,2,3,4],[5,6,7,8]]
# b=a[:,:]
# print(a)
# print(b)
# b[0]=7
# print(b)
# print(a)

# str="hell0"
# str[0]='9'
# print(str)
# from pickletools import uint8
# from os import environ
# import cv2
import numpy as np
# import  pywt
# import matplotlib.image as mpimage
# import matplotlib.pyplot as plt 
# import sys  
# from math import log10, sqrt

def dec2bin(n,w=8):
    if n>=0:
        return np.binary_repr(n, width=w)
    else:
        return np.binary_repr(n)
    
print(dec2bin(255,w=9))

def float2bin(num, places =5):
    # print(str(number))
    if (int(num)==num):
        return dec2bin(num)
    whole, dec = str(num).split(".")
    whole = int(whole)
    dec = float ("."+dec)
    res = dec2bin(whole,0) + "."
    whole, dec = str(float(dec) * 2).split(".")
    for x in range(places):
        if(dec!=0):
            dec = float ("."+dec)
            res += whole
            whole, dec = str(float(decimal_converter(dec)) * 2).split(".")
    return res.rstrip("0")


def decimal_converter(num):
    while num>=1:
        num/=10
    return num

print(float2bin(-655.0))

def bin2float(num):
    flag=False
    if(num[0]=='-'):
        flag=True
    whole,dec=str(num).split(".")
    if(flag):
        whole=whole.lstrip('-')
    whole=whole[::-1]
    print(whole)
    val1=0
    for i in range(0,len(whole)):
        if whole[i]=='1':
            val1+=pow(2,i)
            
    val2=0.0
    for i in range(0,len(dec)):
        if dec[i]=='1':
            val2+=pow(2,-(i+1))
    val=val1+val2
    print(val)
    if(flag):
        val=val*(-1.0)
        print("why")
    return val

print(bin2float('-1010001111.11'))




########################################################################



# np.set_printoptions(threshold=sys.maxsize)
# def suppress_qt_warnings():
#     environ["QT_DEVICE_PIXEL_RATIO"] = "0"
#     environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
#     environ["QT_SCREEN_SCALE_FACTORS"] = "1"
#     environ["QT_SCALE_FACTOR"] = "1"
# def PSNR(original, compressed):
#     mse = np.mean((original - compressed) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * log10(max_pixel / sqrt(mse))
#     return psnr

# def dec2bin(n):
#     return np.binary_repr(n, width=12)

# def float2bin(number, places =10):
#     whole, dec = str(number).split(".")
#     whole = int(whole)
#     dec = float ("."+dec)    
#     res = bin(whole).lstrip("0b") + "."
#     whole, dec = str('{0:.10f}'.format(dec*2)).split(".")
#     for x in range(places):
#         if(dec!=0):
#             dec = float ("."+dec)
#             res += whole
#             whole, dec = str(float(decimal_converter(dec)) * 2).split(".")
#     return res

# def decimal_converter(num):
# 	while num >= 1:
# 		num /= 10
# 	return num

# def bin2float(num):
#     whole,dec=str(num).split(".")
#     whole=whole[::-1]
#     val1=0
#     for i in range(0,len(whole)):
#         if whole[i]=='1':
#             val1+=pow(2,i)
            
#     val2=0.0
#     for i in range(0,len(dec)):
#         if dec[i]=='1':
#             val2+=pow(2,-(i+1))
#     val=val1+val2

#     return val

# if __name__ == "__main__":
#     suppress_qt_warnings()
    
# #------------EMBEDDING-----------
# # 1. Load the RGB cover image.
# # cover_img = cv2.imread(r'lena.jpg')
# cover_img = cv2.imread(r'face20x20.jpg')

# # 2. Obtain the three different planes of the coverimage as three different gray scale images, R(R plane), G (G plane) and B (B plane). 
# B_plane=cover_img[:,:,0] 
# G_plane=cover_img[:,:,1]
# R_plane=cover_img[:,:,2] 


# # 3. Apply the 2-level Haar DWT on the R and G plane of the cover image.
# coeff_R_plane=pywt.wavedec2(R_plane,'haar',level=2)
# coeff_G_plane=pywt.wavedec2(G_plane,'haar',level=2)
# LL2R,(HL2R,LH2R,HH2R),(HL1R,LH1R,HH1R)=coeff_R_plane
# LL2G,(HL2G,LH2G,HH2G),(HL1G,LH1G,HH1G)=coeff_G_plane

# print(LL2R)
# print(LL2G)
# print(HL2R)
# for i in range(0,len(LL2R)):
#     for j in range(0,len(LL2R)):
#         print(LL2R[i,j]," -> ", float2bin(LL2R[i,j]))
# for i in range(0,len(LL2G)):
#     for j in range(0,len(LL2G)):
#         print(LL2G[i,j]," -> ", float2bin(LL2G[i,j]))
# row_R=LL2R.shape[0]
# col_R=LL2R.shape[1]
# arrHL2R=np.zeros((row_R,col_R))
# for i in range(0,len(HL2R)):
#     for j in range(0,len(HL2R)):
#         print(HL2R[i,j]," -> ", dec2bin(int(HL2R[i,j])))

if 1==int(1.1):
    print("true")
else:
    print("false")