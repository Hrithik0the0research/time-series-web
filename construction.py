import cv2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as sk
import pickle
from skimage.transform import resize

def find_contours(mask):
    cont,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    c=max(cont,key=cv2.contourArea)
    c1=c.reshape(c.shape[0],1*2)
    return c1
def distance(c,center):
    dis_list=[]
    for i in c[::-1]:
        x=(abs(i[0]-center[0]))**2
        y=(abs(i[1]-center[1]))**2
        dis=(x+y)**0.5
        dis_list.append(dis)
    return dis_list


def make_standard_Scaler(c1,length):
    st=StandardScaler()
    st_scaler=st.fit(c1)
    st_scaler_all=st_scaler.transform(c1[:length])
    return st_scaler_all,st

def train(n_past,n_future,st_scaler_all,c1):
    trainx=[]
    trainy=[]
    
    for i in range(n_past,len(st_scaler_all)-n_future+1):
        trainx.append(st_scaler_all[i-n_past:i,0:c1.shape[1]])
        trainy.append(st_scaler_all[i+n_future-1:i+n_future])
    trainx,trainy=np.array(trainx),np.array(trainy)
    trainy1=trainy.reshape(trainy.shape[0],trainy.shape[1]*trainy.shape[2])
    return trainx , trainy , trainy1
def make_inverse(st,x,z,model):
    v=st.inverse_transform(model.predict(x[:-100]))
    #vs=model.predict(x[:-100])
    #vs_real=z[:-100]
    #print(vs.shape,vs_real.shape)
    return v
def make_inverse_mlp(v1,st):
    v1=v1.reshape(v1.shape[0]*v1.shape[1],v1.shape[2])
    v11=st.inverse_transform(v1)
    return v11
def compare_img_in_mpl(img,c,c1):
    for i in range(len(c)):
        cv2.circle(img,c[i],1,(0,0,255,255),2)
    for j in range(len(c1)):
        cv2.circle(img,c1[j],1,(0,255,0,255),2)
    plt.axis("off")
    plt.imshow(img)
    plt.savefig("compare1.jpg")
def compare_img(file_name,img,c,c1):
    img=sk.imread(file_name)
    for i in range(len(c)):
        cv2.circle(img,c[i],1,(0,0,255,255),2)
    for j in range(len(c1)):
        cv2.circle(img,c1[j],1,(255,0,0,255),2)
    plt.axis("off")
    
    plt.imshow(img)
    plt.savefig("compare.jpg")