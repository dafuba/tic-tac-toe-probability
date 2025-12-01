import streamlit as st
import joblib
import pandas as pd
from functools import lru_cache

# =========================
# 1. Load the trained model
# =========================
model = joblib.load("model/tic_tac_toe_rf.pkl")

# 2. Define the board columns (must match training)
COLUMNS = [
    "top_left", "top_middle", "top_right",
    "middle_left", "middle_middle", "middle_right",
    "bottom_left", "bottom_middle", "bottom_right",
]

# =========================
# 2. Minimax helper functions
# =========================

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

def winner(board: str):
    """Return 'x', 'o' or None for the given 9-char board string."""
    for a, b, c in LINES:
        if board[a] != "b" and board[a] == board[b] == board[c]:
            return board[a]
    return None

def is_full(board: str) -> bool:
    return "b" not in board

def next_player(board: str) -> str:
    """Determine whose turn it is, assuming X starts and players alternate."""
    xs = board.count("x")
    os = board.count("o")
    return "x" if xs == os else "o"

@lru_cache(None)
def minimax_value(board: str, to_move: str) -> int:
    """
    Returns value from X's perspective:
    1  -> X can force a win
    0  -> draw under perfect play
    -1 -> O can force a win
    """
    w = winner(board)
    if w == "x":
        return 1
    if w == "o":
        return -1
    if is_full(board):
        return 0

    moves = [i for i, c in enumerate(board) if c == "b"]

    if to_move == "x":
        best = -2
        for m in moves:
            nb = board[:m] + "x" + board[m+1:]
            val = minimax_value(nb, "o")
            if val > best:
                best = val
        return best
    else:
        best = 2
        for m in moves:
            nb = board[:m] + "o" + board[m+1:]
            val = minimax_value(nb, "x")
            if val < best:
                best = val
        return best

# =========================
# 3. Streamlit UI
# =========================

st.title("Tic-Tac-Toe Outcome Explorer")
st.write(
    "Select a board state and compare:\n"
    "- the **exact optimal outcome** from minimax (perfect play), and\n"
    "- the **Random Forest model's approximation** and its probabilities."
)

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
    # Build board row & string
    row = [tl, tm, tr, ml, mm, mr, bl, bm, br]
    x_df = pd.DataFrame([row], columns=COLUMNS)
    board_str = "".join(row)  # 'oxxbxoxbb' etc.

    # =========================
    # A) Exact minimax outcome
    # =========================
    to_move = next_player(board_str)
    exact_value = minimax_value(board_str, to_move)  # -1, 0, 1

    exact_outcome_map = {
        -1: "X will **lose** with perfect play (O can force a win).",
        0:  "Game will end in a **draw** with perfect play.",
        1:  "X will **win** with perfect play (X can force a win).",
    }

    st.subheader("Exact Minimax Outcome (Perfect Play)")
    st.write(exact_outcome_map.get(exact_value, f"Unknown outcome (value = {exact_value})"))
    st.caption(f"Minimax value from X's perspective: **{exact_value}**  (1 = win, 0 = draw, -1 = loss)")

    # =========================
    # B) Random Forest approximation
    # =========================
    pred = model.predict(x_df)[0]          # -1, 0, or 1
    proba = model.predict_proba(x_df)[0]   # probabilities
    classes = model.classes_               # e.g., [-1, 0, 1]

    rf_outcome_map = {
        -1: "Model predicts: X will **lose**.",
        0:  "Model predicts: **draw**.",
        1:  "Model predicts: X will **win**.",
    }

    st.subheader("Random Forest Approximation")
    st.write(rf_outcome_map.get(pred, f"Predicted class value: {pred}"))

    # helper to get probability for a class
    def get_prob_for_class(class_value):
        if class_value in classes:
            idx = list(classes).index(class_value)
            return float(proba[idx])
        return None

    win_prob = get_prob_for_class(1)
    draw_prob = get_prob_for_class(0)
    lose_prob = get_prob_for_class(-1)

    st.markdown("**Model confidence (probabilities):**")
    if win_prob is not None:
        st.write(f"🔹 **P(X win)** (value = 1): `{win_prob:.3f}`")
        st.progress(min(max(win_prob, 0.0), 1.0))
    if draw_prob is not None:
        st.write(f"🔹 **P(draw)** (value = 0): `{draw_prob:.3f}`")
    if lose_prob is not None:
        st.write(f"🔹 **P(X loss)** (value = -1): `{lose_prob:.3f}`")

    st.caption(
        "Minimax gives the **exact** optimal outcome under perfect play. "
        "The Random Forest model is trained on those minimax labels and "
        "learns to approximate them; its probabilities reflect model confidence, "
        "not randomness in the game outcome (perfect play is deterministic)."
    )
