import streamlit as st
from construction import *
import math
import matplotlib.pyplot as plt
import random 
import skimage.io as sk
from glob import glob
import cv2
import numpy as np
st.set_page_config(page_title='Image-Reconstruction.AI-Home', layout = 'wide', page_icon = 'logo.png', initial_sidebar_state = 'auto')
st.title("Welcome to :blue[Image Reconstruction.AI ]")
demo=glob("./dataset/*")

demo_file = st.selectbox("Download demo images",demo)
print(demo_file)
with open(demo_file, "rb") as file:
        btn = st.download_button(
                label="Download demo images",
                data=file,
                file_name="demo.png")
name=random.randint(1,10000)
file=st.file_uploader("Upload an image",[".png",".jpg",".webp","jpeg"])
if file is not None:
    file_name="fig"+str(name)+".png"
    with open(file_name, mode='bx') as f:
        f.write(file.getvalue())
    single_image=sk.imread(file_name)
    st.image(file_name)
    
    if single_image.shape[2]==4:
        lower=np.array([10,2,1,1]) #lower
        higher=np.array([255,255,255,255]) #higher
        mask=cv2.inRange(single_image,lower,higher)
    else:
        lower=np.array([10,2,1]) #lower
        higher=np.array([255,255,255]) #higher
        mask=cv2.inRange(single_image,lower,higher)

    x1,y1=mask.shape[0]//2,mask.shape[1]//2
    c1=find_contours(mask)
    c2=(x1,y1)
    c2=(200,300)
    d=distance(c1,c2)
    plt.plot(d,label="original-time-series",color="blue")
    plt.xlabel("Contour Point")
    plt.ylabel("Distance")
    plt.axis("off")
    dis="distance"+".png"
    plt.savefig(dis)
    st.write("Radical scanning")
    st.image(dis)
#print(c1.shape)
    first,st1=make_standard_Scaler(c1,c1.shape[0]-100)
    x,y,z=train(10,1,first,c1)
    models=[" ","LSTM","MLP"]
    model=st.selectbox("Choose a Model",models)
    st.write(model)
    if model!=" ":
        if model=="LSTM":
            model1=load_model("lstm.h5")
            v=make_inverse(st1,x,z,model1)


    #print(v.shape)
    #v1=st.inverse_transform(x[:-600])

    #print(v)
            for i in v:
                for j in range(len(i)):
                    i[j]=math.floor(i[j])
            #print(v)
            print(v.shape)
            v=np.array(v,dtype="int32")
            cb=c1.copy()
            cb=cb[:]
            print(cb.shape)
            compare_img(file_name,single_image,cb,v)
            st.write("Blue is for real and Red is predicted using propsed model")
            st.image("compare.jpg")




        else:
            model1=load_model("mlp.h5")
            v1=model1.predict(x[:-100])
            v11=make_inverse_mlp(v1,st1)
            import math
            #print(v)
            for i in v11:
                for j in range(len(i)):
                    i[j]=math.ceil(i[j])
            #print(v)

            v11=np.array(v11,dtype="int32")
            print(v11)
            cb=c1.copy()
            cb=cb[:]
            compare_img_in_mpl(single_image,cb,v11)
            st.write("Blue is for real and Green is predicted using propsed model")
            st.image("compare1.jpg")

    #plt.axis("off")
    #mask=image_data
    #plt.imshow(mask)
    
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .main {background-color: #f8f9d2;
            background-image: linear-gradient(315deg, #f8f9d2 0%, #e8dbfc 74%);
            color:black;
            
            
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
