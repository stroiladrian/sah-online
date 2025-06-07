import chess
import chess.pgn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from filter import get_board_features, get_move_features

def load_training_data(pgn_file):
    """Load training data from PGN file."""
    X = []  # Features
    y = []  # Labels (1 for good moves, 0 for others)
    
    with open(pgn_file) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
                
            # Use all games with normal termination
            if game.headers.get("Termination") == "Normal":
                board = game.board()
                for move in game.mainline_moves():
                    # Get features for this move
                    board_features = get_board_features(board)
                    from_square, to_square = get_move_features(move)
                    features = np.concatenate((board_features, from_square, to_square))
                    
                    X.append(features)
                    y.append(1)  # This is a good move (led to checkmate)
                    
                    # Add some negative examples (random legal moves)
                    for _ in range(3):  # Add 3 random moves as negative examples
                        if len(list(board.legal_moves)) > 0:
                            random_move = np.random.choice(list(board.legal_moves))
                            board_features = get_board_features(board)
                            from_square, to_square = get_move_features(random_move)
                            features = np.concatenate((board_features, from_square, to_square))
                            
                            X.append(features)
                            y.append(0)  # This is likely not a good move
                    
                    board.push(move)
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train a Random Forest classifier."""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    clf.fit(X, y)
    return clf

def main():
    # Create trained_model directory if it doesn't exist
    os.makedirs("trained_model", exist_ok=True)
    
    # Load and process training data
    print("Loading training data...")
    X, y = load_training_data("ml/training_games.pgn")
    
    print(f"Training data shape: {X.shape}")
    print(f"Number of good moves: {sum(y)}")
    print(f"Number of other moves: {len(y) - sum(y)}")
    
    # Train the model
    print("\nTraining model...")
    clf = train_model(X, y)
    
    # Save the model
    print("\nSaving model...")
    with open("ml/trained_model/dumped_clf.pkl", "wb") as f:
        pickle.dump(clf, f)
    
    print("Done!")

if __name__ == "__main__":
    main() 