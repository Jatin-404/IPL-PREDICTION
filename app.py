# app.py - updated for modern Streamlit
import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(page_title="IPL Win Predictor", layout="wide")

teams = [
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Kings XI Punjab",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Capitals",
]

cities = sorted(
    [
        "Hyderabad",
        "Bangalore",
        "Mumbai",
        "Indore",
        "Kolkata",
        "Delhi",
        "Chandigarh",
        "Jaipur",
        "Chennai",
        "Cape Town",
        "Port Elizabeth",
        "Durban",
        "Centurion",
        "East London",
        "Johannesburg",
        "Kimberley",
        "Bloemfontein",
        "Ahmedabad",
        "Cuttack",
        "Nagpur",
        "Dharamsala",
        "Visakhapatnam",
        "Pune",
        "Raipur",
        "Ranchi",
        "Abu Dhabi",
        "Sharjah",
        "Mohali",
        "Bengaluru",
    ]
)

st.title("IPL Win Predictor")

# --- load model (pipe.pkl) safely ---
MODEL_PATH = "pipe.pkl"
pipe = None
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: '{MODEL_PATH}'.\n\n"
        "Please place pipe.pkl in the same folder as app.py and restart the app."
    )
else:
    try:
        with open(MODEL_PATH, "rb") as f:
            pipe = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model ('{MODEL_PATH}').\n\nError: {e}")

# --- layout ---
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the batting team", sorted(teams))

with col2:
    bowling_team = st.selectbox("Select the bowling team", sorted(teams))

selected_city = st.selectbox("Select host city", cities)

target = st.number_input("Target", min_value=0, step=1, value=160)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Score", min_value=0, step=1, value=0)
with col4:
    overs = st.number_input("Overs completed", min_value=0.0, step=0.1, value=1.0)
with col5:
    wickets_out = st.number_input("Wickets out", min_value=0, max_value=10, step=1, value=0)

# --- button & prediction ---
if st.button("Predict Probability"):
    if pipe is None:
        st.warning("Model not loaded — cannot predict. See error above.")
    else:
        # basic sanity & ordering checks
        try:
            # avoid division by zero for overs
            overs_float = float(overs)
            if overs_float <= 0:
                st.warning("Overs must be > 0 to compute current run rate. Setting to 0.1 for calculation.")
                overs_float = 0.1

            runs_left = int(target) - int(score)
            # 20 overs match -> 120 balls
            balls_left = 120 - int(round(overs_float * 6))
            balls_left = max(balls_left, 1)  # avoid division by zero
            wickets_remaining = 10 - int(wickets_out)
            wickets_remaining = max(wickets_remaining, 0)

            crr = float(score) / overs_float
            rrr = (runs_left * 6) / balls_left

            input_df = pd.DataFrame(
                {
                    "batting_team": [batting_team],
                    "bowling_team": [bowling_team],
                    "city": [selected_city],
                    "runs_left": [runs_left],
                    "balls_left": [balls_left],
                    "wickets": [wickets_remaining],
                    "total_runs_x": [int(target)],
                    "crr": [crr],
                    "rrr": [rrr],
                }
            )

            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            st.metric(label=f"{batting_team} Win Probability", value=f"{round(win*100,2)}%")
            st.metric(label=f"{bowling_team} Win Probability", value=f"{round(loss*100,2)}%")

            st.write("### Model inputs used")
            st.table(input_df.T.rename(columns={0: "value"}))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
