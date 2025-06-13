GAME_REGISTRY = {}

def register_game(name, game_class, standard_nnet_class, gnn_nnet_class=None):
    GAME_REGISTRY[name] = (game_class, standard_nnet_class, gnn_nnet_class)

def get_game(name, use_gnn=False):
    if name not in GAME_REGISTRY:
        raise ValueError(f"Game '{name}' not found. Available: {list(GAME_REGISTRY.keys())}")
    game_class, std_class, gnn_class = GAME_REGISTRY[name]
    if use_gnn:
        if gnn_class is None:
            raise ValueError(f"GNN version of '{name}' is not implemented")
        return (game_class, gnn_class)
    else:
        return (game_class, std_class)

def list_games():
    return list(GAME_REGISTRY.keys())

def has_gnn_version(name):
    return (name in GAME_REGISTRY) and (GAME_REGISTRY[name][2] is not None)

# Register games and networks
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToeNet import TicTacToeNNetWrapper
from tictactoe.TicTacToeGNN import TicTacToeGNNWrapper
register_game('tictactoe', TicTacToeGame, TicTacToeNNetWrapper, TicTacToeGNNWrapper)

from frozenlake.FrozenLakeGame import FrozenLakeGame
from frozenlake.FrozenLakeNet import FrozenLakeNet
register_game('frozenlake', FrozenLakeGame, FrozenLakeNet)

from connect4.Connect4Game import Connect4Game
from connect4.Connect4Net import Connect4NNetWrapper
from connect4.Connect4GNN import Connect4GNNWrapper
register_game('connect4', Connect4Game, Connect4NNetWrapper, Connect4GNNWrapper)
