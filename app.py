import streamlit as st
import numpy as np
import pickle

with open(r"breast.pickel", "rb") as f:
    model = pickle.load(f)

class_names = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6"]

st.title("🧬 Breast Tissue Classification App")
st.write("Fill the values below to predict the **Breast Tissue Class**")

# Input fields
I0 = st.number_input("I0", min_value=0.0, step=0.01)
PA500 = st.number_input("PA500", min_value=0.0, step=0.01)
HFS = st.number_input("HFS", min_value=0.0, step=0.01)
DA = st.number_input("DA", min_value=0.0, step=0.01)
Area = st.number_input("Area", min_value=0.0, step=0.01)
ADA = st.number_input("A.DA", min_value=0.0, step=0.01)
MaxIP = st.number_input("Max.IP", min_value=0.0, step=0.01)
DR = st.number_input("DR", min_value=0.0, step=0.01)
P = st.number_input("P", min_value=0.0, step=0.01)

# Collect all inputs into a numpy array
features = np.array([[I0, PA500, HFS, DA, Area, ADA, MaxIP, DR, P]])

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(features)
    class_label = class_names[prediction[0]]

    if class_label in ['Class1', 'Class2']:
        result = "Negative"
    elif class_label in ["Class3", "Class4"]:
        result = "Positive"
    else:
        result = "ERROR"


st.success(f"Predicted Breast Cancer Class: {class_label} \n Diagnosis = {result}")