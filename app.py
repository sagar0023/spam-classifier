import streamlit as st
import pickle
import text_transform

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
with st.sidebar:
    st.markdown("### 📬 Contact")
    st.markdown("""
    - **Created by**: sagar sharma 
    - **Email**: [sagarsharma23112@gmail.com](mailto:sagarsharma23112@gmail.com)  
    """)


input_sms = st.text_area("Enter message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = text_transform.transform_text(input_sms)
   
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


