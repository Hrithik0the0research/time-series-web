import streamlit as st
st.set_page_config(page_title='Image-Reconstruction.AI-ABOUT', layout = 'wide', page_icon = 'logo.png', initial_sidebar_state = 'auto')
st.title("ABOUT")
wr="""
For a long time, time series have been an extensively researched topic of discussion. Once the conversion of pictures into time series is authorised, all of these clustering approaches become available. Hence, the time series distance computation consists of two steps: determining the centroid location and then calculating the distance between the centre and border pixels. The study employs numerous predictive algorithms to dissect the output given by each one and to ensure the restoration of the original irregularly shaped image, which gives a variety of benefits such as improved image analysis and innovative imaging enterprise to the field of predictive algorithms.
"""
st.write(wr)
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
