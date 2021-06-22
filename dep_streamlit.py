import streamlit as st
import numpy as np
from functions import *
import warnings
import base64
from PIL import Image
warnings.filterwarnings("ignore")

st.title("Detecting Depression")
img = Image.open("depression.png")
st.image(img, width=350)
main_bg = "dep.jpg"
main_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()}) no-repeat center center fixed; 
        background-size: cover;
    }}
        </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""Depression is a common and serious medical illness 
             that negatively affects how a person feels, the way a person thinks and how a person acts.
             Early detection of Depression is better for complete cure.
             One way of identifying possible signs of Depression in an individual can be done through Twitter tweets,
             Facebook posts and other Social media.
""")


text = st.text_area("Enter Text","Type Here")
text = np.array([[text]])

def prediction(t):
    res = clean_tweets(t)
    res = token(res)
    result = model1(res)
    return result

if __name__ == "__main__":
      if st.button("Predict"):
        out = prediction(text)
        out = [j for i in out for j in i] 
        out = ['Depression' if n==1 else 'No Depression'for n in out]
        st.success(out)
