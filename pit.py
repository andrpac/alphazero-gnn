import argparse
import os
import yaml
import numpy as np

from register import get_game, list_games, has_gnn_version
from MCTS import MCTS
from Arena import Arena

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def copy(self):
        return dotdict(super().copy())

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def config_to_args(cfg):
    args = dotdict()
    for section, sec in cfg.items():
        if isinstance(sec, dict):
            for k, v in sec.items():
                args[k] = v
    if 'checkpoint_path' in args and 'checkpoint' not in args:
        args.checkpoint = args.checkpoint_path
    elif 'checkpoint' in args and 'checkpoint_path' not in args:
        args.checkpoint_path = args.checkpoint
    return args

def evaluate_models(game_name, model1_path, use_gnn1, model2_path, use_gnn2, config_args, num_games=50):
    GameCls, NetCls1 = get_game(game_name, use_gnn1)
    _,       NetCls2 = get_game(game_name, use_gnn2)

    board_size = getattr(config_args, 'board_size', None)
    if board_size is not None:
        game = GameCls(board_size=board_size)
    else:
        game = GameCls()

    cfg1 = config_args.copy()
    cfg1.gnn_predict = use_gnn1
    cfg2 = config_args.copy()
    cfg2.gnn_predict = use_gnn2

    net1 = NetCls1(game, cfg1)
    net2 = NetCls2(game, cfg2)

    folder1, fname1 = os.path.split(model1_path)
    folder2, fname2 = os.path.split(model2_path)
    net1.load_checkpoint(folder1, fname1)
    net2.load_checkpoint(folder2, fname2)

    def player1(board):
        mcts = MCTS(game, net1, cfg1)
        pi = mcts.getActionProb(board, temp=0)
        return np.argmax(pi)

    def player2(board):
        mcts = MCTS(game, net2, cfg2)
        pi = mcts.getActionProb(board, temp=0)
        return np.argmax(pi)

    # Use the provided Arena implementation, which logs via tqdm and logging
    arena = Arena(player1, player2, game)

    # This will display progress bars and logging as defined in Arena
    wins1, wins2, draws = arena.playGames(num_games)

    print(f"Final results after {num_games} games:")
    print(f"Model1 ({model1_path}) wins: {wins1}")
    print(f"Model2 ({model2_path}) wins: {wins2}")
    print(f"Draws: {draws}")
    return wins1, wins2, draws

def main():
    parser = argparse.ArgumentParser(description="Pit two AlphaZero models")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--game', type=str, required=True, help='Game name')
    parser.add_argument('--model1', type=str, required=True, help='First model checkpoint')
    parser.add_argument('--use_gnn1', action='store_true', help='First model is GNN')
    parser.add_argument('--model2', type=str, required=True, help='Second model checkpoint')
    parser.add_argument('--use_gnn2', action='store_true', help='Second model is GNN')
    parser.add_argument('--num_games', type=int, default=50, help='Number of games')

    args = parser.parse_args()

    cfg = load_config(args.config)
    config_args = config_to_args(cfg)

    if args.game not in list_games():
        print(f"Unknown game '{args.game}'. Known: {list_games()}")
        exit(1)
    if args.use_gnn1 and not has_gnn_version(args.game):
        print(f"No GNN version for model1")
        exit(1)
    if args.use_gnn2 and not has_gnn_version(args.game):
        print(f"No GNN version for model2")
        exit(1)

    evaluate_models(
        game_name   = args.game,
        model1_path = args.model1,
        use_gnn1    = args.use_gnn1,
        model2_path = args.model2,
        use_gnn2    = args.use_gnn2,
        config_args = config_args,
        num_games   = args.num_games
    )

if __name__ == '__main__':
    main()
