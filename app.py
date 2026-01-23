import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import zipfile
import os

# --- 1. MODEL ARCHITECTURE (MUST MATCH TRAINING) ---
class OmniBrain(nn.Module):
    def __init__(self, input_dim=26, action_dim=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 2. GAME LOGIC ---
def check_win(board, win_len, size, player):
    for r in range(size):
        for c in range(size):
            if board[r,c] != player: continue
            for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                count = 0
                for i in range(win_len):
                    nr, nc = r + dr*i, c + dc*i
                    if 0 <= nr < size and 0 <= nc < size and board[nr,nc] == player:
                        count += 1
                    else: break
                if count == win_len: return True
    return False

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Dark Lucid: Omni-Gamer", page_icon="🧠")
st.title("🧠 Dark Lucid: The Omni-Gamer")
st.markdown("### 0% Cheat. 100% General Intelligence.")
st.markdown("This agent learned 10 different games simultaneously.")

# SIDEBAR: UPLOAD & SELECT
with st.sidebar:
    st.header("1. Activate Brain")
    uploaded_file = st.file_uploader("Upload 'omnibrain.zip'", type="zip")
    
    model = None
    configs = None
    
    if uploaded_file is not None:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall("temp_brain")
        
        # Load Configs
        with open("temp_brain/game_configs.pkl", "rb") as f:
            configs = pickle.load(f)
            
        # Load Model
        model = OmniBrain()
        model.load_state_dict(torch.load("temp_brain/omnibrain_model.pth", map_location=torch.device('cpu')))
        model.eval()
        st.success("Omni-Brain Active!")

# MAIN GAME AREA
if model and configs:
    game_names = [
        "Tic-Tac-Toe", "Mini Connect-4", "Mini Gomoku", "Anti-Tic-Tac-Toe",
        "4x4 Tic-Tac-Toe", "Gravity 5x5", "Connect-2 (Baby Mode)",
        "Hard Connect-4", "Blind Tic-Tac-Toe", "Loose Gomoku"
    ]
    
    selected_game_name = st.selectbox("Select Game Mode", game_names)
    game_id = game_names.index(selected_game_name)
    size, gravity, win_len, misere = configs[game_id]
    
    st.write(f"**Rules:** Grid {size}x{size} | Win Length: {win_len} | Gravity: {'ON' if gravity else 'OFF'} | {'Avoid Winning!' if misere else 'Connect to Win!'}")
    
    # Initialize Board
    if 'board' not in st.session_state or st.session_state.game_id != game_id:
        st.session_state.game_id = game_id
        st.session_state.board = np.zeros((size, size), dtype=int)
        st.session_state.game_over = False
        st.session_state.winner = None

    # DRAW GRID
    board_container = st.container()
    cols = board_container.columns(size)
    
    # HUMAN MOVE
    def human_move(r, c):
        if st.session_state.board[r, c] == 0 and not st.session_state.game_over:
            # Gravity Logic
            if gravity:
                actual_r = -1
                for row in range(size-1, -1, -1):
                    if st.session_state.board[row, c] == 0:
                        actual_r = row
                        break
                if actual_r != -1: r = actual_r
            
            # Place
            st.session_state.board[r, c] = -1 # Human is -1
            
            # Check Win
            if check_win(st.session_state.board, win_len, size, -1):
                st.session_state.game_over = True
                st.session_state.winner = "Human" if not misere else "AI"
            
            # AI MOVE (If game not over)
            elif not np.any(st.session_state.board == 0):
                st.session_state.game_over = True
                st.session_state.winner = "Draw"
            else:
                ai_move()

    def ai_move():
        # Prepare Input
        flat = st.session_state.board.flatten()
        padded = np.zeros(25, dtype=float)
        padded[:len(flat)] = flat
        input_vec = np.append(padded, game_id)
        input_tensor = torch.FloatTensor(input_vec).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            q_vals = model(input_tensor).squeeze()
        
        # Mask Illegal Moves
        while True:
            action = torch.argmax(q_vals).item()
            r, c = divmod(action, size)
            if action < size**2 and st.session_state.board[r, c] == 0:
                break
            else:
                q_vals[action] = -float('inf') # Mask and try next best
        
        # Apply Move
        st.session_state.board[r, c] = 1
        
        # Check Win
        if check_win(st.session_state.board, win_len, size, 1):
            st.session_state.game_over = True
            st.session_state.winner = "AI" if not misere else "Human"

    # RENDER BUTTONS
    for r in range(size):
        for c in range(size):
            val = st.session_state.board[r, c]
            label = " "
            if val == 1: label = "❌" # AI
            if val == -1: label = "⭕" # Human
            
            cols[c].button(label, key=f"{r}-{c}", on_click=human_move, args=(r,c), disabled=st.session_state.game_over)

    # RESULT
    if st.session_state.game_over:
        if st.session_state.winner == "AI":
            st.error("🤖 DARK LUCID WINS! (Resistance is futile)")
        elif st.session_state.winner == "Human":
            st.balloons()
            st.success("🎉 YOU WON! (Impossible...)")
        else:
            st.warning("🤝 DRAW!")
        
        if st.button("Replay"):
            st.session_state.board = np.zeros((size, size), dtype=int)
            st.session_state.game_over = False
            st.experimental_rerun()

else:
    st.info("👈 Upload 'omnibrain.zip' to start the simulation.")
