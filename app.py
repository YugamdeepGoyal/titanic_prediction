import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost.pkl")

model = load_model()

st.title("TITANIC SURVIVAL PREDICTOR")
with st.sidebar:
    st.title("INPUT")
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children aboard", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 512.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
    if st.button("PREDICT"):
        input_df = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked,
        "name": f"Passenger, {title}. X",
        "ticket": "",
        "cabin": None
        }])
        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        st.session_state["proba"] = proba
        st.session_state["prediction"] = prediction
if "prediction" in st.session_state:
    if st.session_state["prediction"] == 1:
        st.subheader("Survived")
    else:
        st.subheader("Not Survived")
    st.write(f"Probability Predicted is: {st.session_state['proba']:.1%}")
else:
    st.write("Enter passenger details in the sidebar and click Predict.")