🧠 Tic-Tac-Toe Optimal Outcome & Best-Move Predictor
Streamlit App · Minimax AI · Machine Learning Model

This project combines classic AI (minimax) and machine learning (Random Forest) to analyze any Tic-Tac-Toe board state.
The system can:

✅ Predict the true game-theoretic outcome (win/draw/loss)
✅ Recommend the best next move under perfect play
✅ Use an ML model to approximate minimax and show its confidence
✅ Provide an interactive UI for exploring game states

This is both an AI project and a machine learning project wrapped in a clean Streamlit app.

📌 Project Overview

Tic-Tac-Toe is fully solvable. For any board state, the outcome under perfect play is deterministic:

X can force a win

O can force a win

Both can force a draw

This project:

Computes exact minimax values for every reachable game state

Generates a dataset of all legal states labeled with optimal outcome:

1 → X win

0 → Draw

-1 → X loss

Trains a Random Forest model to approximate this mapping (state → value)

Builds a Streamlit app to show both:

the exact logical minimax result, and

the model’s prediction + probabilities

It also provides the best next move, so you can see exactly how perfect-play AI behaves.

📊 Dataset Description

A custom dataset is generated using a game simulator and minimax:

data/tic_tac_toe_minimax_states.csv


Each row is:

Cell 1	Cell 2	...	Cell 9	value
x/o/b	x/o/b	...	x/o/b	-1/0/1

Where:

value = 1 → X can force a win

value = 0 → Game ends in a draw under perfect play

value = -1 → O can force a win

This dataset contains all reachable legal states of a Tic-Tac-Toe game (≈ 5,478 states).

🎯 Why No Train/Test Split?

Normally, ML requires a test set.
But here:

✔ The dataset is complete and contains all possible states
✔ There are no “future” or “unseen” states
✔ Every input in the app corresponds to a row in the dataset
✔ This is a value-function approximation task, not prediction of unknown data

So the model is intentionally trained on 100% of the data, because:

The goal is for the model to approximate the minimax value function, not to generalize beyond unseen data.

The app still shows the difference between:

the true minimax outcome, and

the Random Forest approximation.

🤖 AI Component: Minimax (Perfect Play)

The app includes a full minimax solver that computes:

Exact optimal outcome (X win / draw / X loss)

Best next move(s)

Which player is to move

Terminal board detection

This is the same logic used in classic game AI and ensures fully correct ground-truth labels.
