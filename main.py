import argparse
import logging
import os
import sys
import yaml
import coloredlogs

from Coach import Coach
from register import get_game, list_games, has_gnn_version

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, val):
        self[name] = val
    def copy(self):
        return dotdict(dict.copy(self))

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def config_to_args(cfg):
    args = dotdict({})
    for section in cfg.values():
        for k, v in section.items():
            args[k] = v
    if 'checkpoint_path' in args and 'checkpoint' not in args:
        args.checkpoint = args.checkpoint_path
    elif 'checkpoint' in args and 'checkpoint_path' not in args:
        args.checkpoint_path = args.checkpoint
    return args

def get_checkpoint_path(game_name, filename, use_gnn=False, base_path="./checkpoints"):
    folder = os.path.join(base_path, game_name)
    if not filename.endswith('.pth.tar'):
        filename += '.pth.tar'
    if use_gnn and not filename.endswith('_gnn.pth.tar'):
        filename = filename[:-len('.pth.tar')] + '_gnn.pth.tar'
    return folder, filename

def create_game_instance(GameClass, config_args):
    name = getattr(config_args, 'game', None)
    if name == 'tictactoe':
        return GameClass(n=config_args.board_size)
    elif name == 'frozenlake':
        return GameClass(
            map_size=config_args.board_size,
            custom_map=getattr(config_args, 'custom_map', None),
            is_slippery=getattr(config_args, 'is_slippery', False),
            render_mode=getattr(config_args, 'render_mode', None)
        )
    elif name == 'connect4':
        return GameClass(board_size=config_args.board_size)
    else:
        return GameClass()

def main():
    parser = argparse.ArgumentParser(description="AlphaZero with optional GNN")
    parser.add_argument('--game', type=str, required=True,
                        help=f"Game to train. Available: {list_games()}")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config.yaml (default: <game>/config.yaml)")
    parser.add_argument('--load_model', action='store_true',
                        help="Load latest checkpoint before training")

    # GNN options
    parser.add_argument('--gnn_accumulate', action='store_true',
                        help="Collect graph examples during self-play")
    parser.add_argument('--gnn_collect_interval', type=int,
                        help="Moves between graph-expansion steps")
    parser.add_argument('--gnn_train', action='store_true',
                        help="Train the GNN periodically")
    parser.add_argument('--gnn_train_interval', type=int,
                        help="Iterations between GNN training phases")
    parser.add_argument('--gnn_predict', action='store_true',
                        help="Route inference through GNN pipeline")

    # Core overrides
    parser.add_argument('--board_size', type=int, help="Override board size from config")
    parser.add_argument('--numIters', type=int, help="Override number of training iterations")
    parser.add_argument('--numMCTSSims', type=int, help="Override number of MCTS simulations")
    args = parser.parse_args()

    if args.game not in list_games():
        log.error(f"Game '{args.game}' not found. Available: {list_games()}")
        sys.exit(1)

    if args.config is None:
        args.config = os.path.join(args.game, 'config.yaml')
    log.info(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    config_args = config_to_args(cfg)

    # Apply overrides
    if args.board_size is not None: config_args.board_size = args.board_size
    if args.numIters   is not None: config_args.numIters   = args.numIters
    if args.numMCTSSims is not None: config_args.numMCTSSims= args.numMCTSSims
    if args.gnn_accumulate:   config_args.gnn_accumulate   = True
    if args.gnn_collect_interval: config_args.gnn_collect_interval = args.gnn_collect_interval
    if args.gnn_train:        config_args.gnn_train        = True
    if args.gnn_train_interval: config_args.gnn_train_interval  = args.gnn_train_interval
    if args.gnn_predict:      config_args.gnn_predict      = True

    config_args.game = args.game
    use_gnn = any([getattr(config_args, flag, False) 
                   for flag in ['gnn_accumulate','gnn_train','gnn_predict']])
    if use_gnn and not has_gnn_version(args.game):
        log.error(f"GNN version for '{args.game}' is not implemented.")
        sys.exit(1)

    ckpt_folder, best_file = get_checkpoint_path(
        args.game, 'best', use_gnn, base_path=config_args.checkpoint)
    os.makedirs(ckpt_folder, exist_ok=True)
    config_args.checkpoint = ckpt_folder
    config_args.load_folder_file = (ckpt_folder, best_file)

    GameClass, BaseNetClass = get_game(args.game, use_gnn=False)
    _, GnnNetClass = get_game(args.game, use_gnn=True)
    log.info(f"Creating game '{args.game}' (board_size={config_args.board_size})")
    game = create_game_instance(GameClass, config_args)

    NetClass = GnnNetClass if use_gnn else BaseNetClass
    log.info(f"Initializing {'GNN-enhanced' if use_gnn else 'regular'} network")
    nnet = NetClass(game, config_args)

    if args.load_model:
        log.info(f"Loading checkpoint {config_args.load_folder_file[0]}/{config_args.load_folder_file[1]}")
        nnet.load_checkpoint(*config_args.load_folder_file)
    else:
        log.info("Starting from scratch")

    coach = Coach(game, nnet, config_args)
    coach.learn()

if __name__ == '__main__':
    main()
