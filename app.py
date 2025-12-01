import streamlit as st
import joblib
import pandas as pd

# 1. Load the trained model
model = joblib.load("model/tic_tac_toe_rf.pkl")

# 2. Define the board columns (must match training)
COLUMNS = [
    "top_left", "top_middle", "top_right",
    "middle_left", "middle_middle", "middle_right",
    "bottom_left", "bottom_middle", "bottom_right",
]

st.title("Tic-Tac-Toe Win Predictor (for X)")
st.write("Select the current board state and let the model predict if this is a winning position for X.")

# 3. Build a 3x3 input grid
options = ["x", "o", "b"]

col1, col2, col3 = st.columns(3)

with col1:
    tl = st.selectbox("Top left", options, index=2)
    ml = st.selectbox("Middle left", options, index=2)
    bl = st.selectbox("Bottom left", options, index=2)

with col2:
    tm = st.selectbox("Top middle", options, index=2)
    mm = st.selectbox("Middle middle", options, index=2)
    bm = st.selectbox("Bottom middle", options, index=2)

with col3:
    tr = st.selectbox("Top right", options, index=2)
    mr = st.selectbox("Middle right", options, index=2)
    br = st.selectbox("Bottom right", options, index=2)

if st.button("Predict"):
    # 4. Create a one-row dataframe
    row = [tl, tm, tr, ml, mm, mr, bl, bm, br]
    x_df = pd.DataFrame([row], columns=COLUMNS)

    # 5. Predict label + probability
    pred = model.predict(x_df)[0]
    proba = model.predict_proba(x_df)[0]

    # Assume classes are ['negative', 'positive']
    class_names = model.classes_
    proba_dict = dict(zip(class_names, proba))

    st.subheader("Prediction")
    st.write(f"**Predicted class:** {pred}")

    st.subheader("Winning probability for X (class = 'positive')")
    win_prob = proba_dict.get("positive", None)
    if win_prob is not None:
        st.write(f"Estimated probability: **{win_prob:.3f}**")
        st.progress(min(max(win_prob, 0), 1))
    else:
        st.write("Could not find 'positive' class in model classes:", class_names)

    st.caption("Note: Model was trained on *endgame* positions only. "
               "Probabilities for intermediate boards are extrapolations, not true game-theoretic win chances.")
