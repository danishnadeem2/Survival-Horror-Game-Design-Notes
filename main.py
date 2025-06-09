import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import pandas as pd
import joblib
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv('monster_moves.csv')
print("📄 CSV Columns:", df.columns.tolist())
df.head()



# Game Configuration
GRID_SIZE = 10
rooms = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
room_connections = {}
water_rooms = []
page_rooms = set()
player_pos = (0, 0)
monster_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
collected_pages = set()
game_over = False
turn = 0

# Generate room connections
def generate_sparse_connections():
    for room in rooms:
        x, y = room
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < GRID_SIZE and 0 <= ny_ < GRID_SIZE and random.random() < 0.6:
                neighbors.append((nx_, ny_))
        room_connections[room] = neighbors

generate_sparse_connections()
water_rooms = random.sample(rooms, 10)
page_rooms = set(random.sample(water_rooms, 3))

# Create graph
G = nx.Graph()
for room, neighbors in room_connections.items():
    for neighbor in neighbors:
        G.add_edge(room, neighbor)

# Plot Game
def plot_game(player_pos, monster_pos, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True)

    for room, neighbors in room_connections.items():
        for n in neighbors:
            ax.plot([room[0], n[0]], [room[1], n[1]], 'gray', alpha=0.3)

    for r in water_rooms:
        ax.add_patch(patches.Circle((r[0], r[1]), 0.3, color='blue', alpha=0.2))

    for r in page_rooms - collected_pages:
        ax.add_patch(patches.Circle((r[0], r[1]), 0.2, color='green'))

    if path:
        path_xs, path_ys = zip(*path)
        ax.plot(path_xs, path_ys, 'r--', linewidth=2)

    ax.add_patch(patches.Circle((player_pos[0], player_pos[1]), 0.4, color='yellow'))
    ax.add_patch(patches.Circle((monster_pos[0], monster_pos[1]), 0.4, color='red'))

    ax.set_title(f"Pages Collected: {len(collected_pages)} / 3")
    plt.show()

# Valid moves
def get_valid_moves(pos):
    return room_connections.get(pos, [])

# Move player
def move_player(pos, direction):
    dx, dy = {'w': (0, 1), 's': (0, -1), 'a': (-1, 0), 'd': (1, 0)}.get(direction, (0, 0))
    new_pos = (pos[0] + dx, pos[1] + dy)
    if new_pos in get_valid_moves(pos):
        return new_pos
    return pos

# ------------------------------
# ML MODEL SECTION
# ------------------------------

MODEL_PATH = 'monster_ai_model.joblib'
model_trained = os.path.exists(MODEL_PATH)

if not model_trained:
    print("⚠️ Model not found. Training from dataset...")

    df = pd.read_csv('monster_moves.csv')
    df.dropna(inplace=True)

    if not all(col in df.columns for col in ['PlayerX', 'PlayerY', 'MonsterX', 'MonsterY', 'NextMonsterX', 'NextMonsterY']):
        raise ValueError("CSV missing required columns.")

    X = df[['PlayerX', 'PlayerY', 'MonsterX', 'MonsterY']]
    y = df[['NextMonsterX', 'NextMonsterY']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    print("✅ Model trained and saved.")

# Load model
monster_model = joblib.load(MODEL_PATH)

# Predict monster's next move
def predict_monster_move(player_pos, monster_pos):
    px, py = player_pos
    mx, my = monster_pos
    pred = monster_model.predict([[px, py, mx, my]])
    pred_coords = tuple(int(round(x)) for x in pred.flatten())

    # Validate prediction
    if pred_coords in room_connections.get(monster_pos, []):
        return pred_coords
    else:
        # Fall back to a random neighbor
        neighbors = room_connections.get(monster_pos, [])
        return random.choice(neighbors) if neighbors else monster_pos

# Game loop
def game_turn():
    global player_pos, monster_pos, game_over, turn

    while not game_over:
        print(f"\nTurn {turn}")
        move = input("Move (WASD), C to check room, Q to quit: ").lower()

        if move == 'q':
            print("Game quit.")
            break

        if move in ['w', 'a', 's', 'd']:
            new_player_pos = move_player(player_pos, move)
            if new_player_pos != player_pos:
                player_pos = new_player_pos
                print(f"Moved to {player_pos}")
            else:
                print("Invalid move.")
        elif move == 'c':
            if player_pos in page_rooms and player_pos not in collected_pages:
                collected_pages.add(player_pos)
                print(f"Page found! {len(collected_pages)}/3 collected.")
            else:
                print("Nothing found.")
        else:
            print("Invalid input.")

        # Monster uses ML to predict movement
        new_monster_pos = predict_monster_move(player_pos, monster_pos)
        monster_pos = new_monster_pos

        # Check win/lose
        if monster_pos == player_pos:
            game_over = True
            print("💀 The monster caught you. Game over.")
        elif len(collected_pages) >= 3:
            game_over = True
            print("🏆 You collected all pages and escaped. You win!")

        # Optional path display
        path = [monster_pos, player_pos]
        plot_game(player_pos, monster_pos, path)

        turn += 1

# Start game
if __name__ == "__main__":
    print("🕹️ Welcome to the AI Horror Game with ML Monster!")
    game_turn()
