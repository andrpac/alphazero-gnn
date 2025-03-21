"""
Game registry for AlphaZero implementation.
This file allows registration and retrieval of different games and their neural networks.
"""

# Game registry dict that maps game names to game_class, standard_nnet_class, and gnn_nnet_class
GAME_REGISTRY = {}

def register_game(name, game_class, standard_nnet_class, gnn_nnet_class=None):
    """
    Register a game and its neural network implementations.
    
    Args:
        name: String identifier for the game
        game_class: Game class implementation 
        standard_nnet_class: Standard neural network class implementation
        gnn_nnet_class: GNN-enhanced neural network class implementation (optional)
    """
    GAME_REGISTRY[name] = (game_class, standard_nnet_class, gnn_nnet_class)
    
def get_game(name, use_gnn=False):
    """
    Get a game and its neural network by name.
    
    Args:
        name: String identifier for the game
        use_gnn: Whether to use the GNN-enhanced neural network
        
    Returns:
        Tuple of (game_class, nnet_class)
        
    Raises:
        ValueError if game is not registered or GNN version is not available
    """
    if name not in GAME_REGISTRY:
        raise ValueError(f"Game '{name}' not found in registry. Available games: {list(GAME_REGISTRY.keys())}")
    
    game_class, standard_nnet_class, gnn_nnet_class = GAME_REGISTRY[name]
    
    if use_gnn:
        if gnn_nnet_class is None:
            raise ValueError(f"GNN version of '{name}' is not implemented")
        return (game_class, gnn_nnet_class)
    else:
        return (game_class, standard_nnet_class)

def list_games():
    """List all registered games"""
    return list(GAME_REGISTRY.keys())

def has_gnn_version(name):
    """Check if a game has a GNN version implemented"""
    if name not in GAME_REGISTRY:
        return False
    return GAME_REGISTRY[name][2] is not None

# Import TicTacToe game and networks
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToeNet import TicTacToeNNetWrapper
from tictactoe.TicTacToeGNN import TicTacToeGNNWrapper

# Register TicTacToe game with both standard and GNN networks
register_game('tictactoe', TicTacToeGame, TicTacToeNNetWrapper, TicTacToeGNNWrapper)

# Import FrozenLake game and networks
from frozenlake.FrozenLakeGame import FrozenLakeGame
from frozenlake.FrozenLakeNet import FrozenLakeNet

# Register FrozenLake game (GNN version to be added later)
register_game('frozenlake', FrozenLakeGame, FrozenLakeNet)

# Import Connect4 game and networks
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Net import Connect4NNetWrapper
from connect4.Connect4GNN import Connect4GNNWrapper

# Register Connect4 game with both standard and GNN networks
register_game('connect4', Connect4Game, Connect4NNetWrapper, Connect4GNNWrapper)

# New games can be registered here by following the same pattern