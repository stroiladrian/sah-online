import requests
import os
import time

def download_games():
    """Download high-quality chess games from Lichess."""
    # Create directory if it doesn't exist
    os.makedirs("back-end/ml", exist_ok=True)
    
    print("Downloading games from Lichess...")
    
    # Download games from Lichess API
    url = "https://lichess.org/api/games/user/DrNykterstein"
    params = {
        "max": 100,  # Get 100 games
        "pgnInJson": "false",
        "clocks": "false",
        "evals": "false",
        "opening": "false"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Save games to file
        with open("ml/training_games.pgn", "w") as f:
            f.write(response.text)
        print("Done!")
    else:
        print(f"Failed to download games. Status code: {response.status_code}")

if __name__ == "__main__":
    download_games() 