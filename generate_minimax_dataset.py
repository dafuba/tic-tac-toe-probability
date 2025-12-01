# generate_minimax_dataset.py
from functools import lru_cache
import csv
import os

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

def winner(board: str):
    for a, b, c in LINES:
        if board[a] != "b" and board[a] == board[b] == board[c]:
            return board[a]
    return None

def is_full(board: str) -> bool:
    return "b" not in board

def next_player(board: str) -> str:
    xs = board.count("x")
    os = board.count("o")
    # X always starts; players alternate
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

states = {}

def generate(board: str = "b" * 9):
    to_move = next_player(board)
    v = minimax_value(board, to_move)
    states[board] = v

    # Stop if terminal
    if winner(board) or is_full(board):
        return

    for i, c in enumerate(board):
        if c == "b":
            nb = board[:i] + to_move + board[i+1:]
            if nb not in states:
                generate(nb)

def main():
    generate()

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "tic_tac_toe_minimax_states.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "top_left", "top_middle", "top_right",
            "middle_left", "middle_middle", "middle_right",
            "bottom_left", "bottom_middle", "bottom_right",
            "value"  # -1, 0, 1
        ])

        for board, v in states.items():
            row = list(board)  # 9 chars: x, o, b
            writer.writerow(row + [v])

    print(f"Saved {len(states)} states to {out_path}")

if __name__ == "__main__":
    main()
