"""
Game registry for AlphaZero implementation.
This file allows registration and retrieval of different games and their neural networks.
"""

# Game registry dict that maps game names to (game_class, nnet_class) tuples
GAME_REGISTRY = {}

def register_game(name, game_class, nnet_class):
    """
    Register a game and its neural network implementation.
    
    Args:
        name: String identifier for the game
        game_class: Game class implementation 
        nnet_class: Neural network class implementation
    """
    GAME_REGISTRY[name] = (game_class, nnet_class)
    
def get_game(name):
    """
    Get a game and its neural network by name.
    
    Args:
        name: String identifier for the game
        
    Returns:
        Tuple of (game_class, nnet_class)
        
    Raises:
        ValueError if game is not registered
    """
    if name not in GAME_REGISTRY:
        raise ValueError(f"Game '{name}' not found in registry. Available games: {list(GAME_REGISTRY.keys())}")
    return GAME_REGISTRY[name]

def list_games():
    """List all registered games"""
    return list(GAME_REGISTRY.keys())

# Register TicTacToe game
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToeNet import TicTacToeNNetWrapper
register_game('tictactoe', TicTacToeGame, TicTacToeNNetWrapper)

# Register FrozenLake game
from frozenlake.FrozenLakeGame import FrozenLakeGame
from frozenlake.FrozenLakeNet import FrozenLakeNet
register_game('frozenlake', FrozenLakeGame, FrozenLakeNet)

# Register Connect4 game
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Net import Connect4NNetWrapper
register_game('connect4', Connect4Game, Connect4NNetWrapper)

# New games can be registered here