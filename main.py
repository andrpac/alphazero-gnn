import argparse
import logging
import coloredlogs
import os
import yaml
import sys
import numpy as np

from Coach import Coach
from Arena import Arena
from MCTS import MCTS
from register import get_game, list_games, has_gnn_version

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# A simple dotdict implementation for args
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
    def __setattr__(self, name, value):
        self[name] = value

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config_to_args(config):
    args = dotdict({})
    
    for section in config:
        for key, value in config[section].items():
            args[key] = value
    
    # Add backward compatibility for checkpoint paths
    if 'checkpoint_path' in args and 'checkpoint' not in args:
        args.checkpoint = args.checkpoint_path
    elif 'checkpoint' in args and 'checkpoint_path' not in args:
        args.checkpoint_path = args.checkpoint
    
    return args

def get_checkpoint_path(game_name, filename, use_gnn=False, base_path="./checkpoints"):
    # Create game-specific checkpoint directory 
    folder_path = os.path.join(base_path, game_name)
    
    # Determine filename based on GNN flag
    if use_gnn and not filename.endswith('_gnn.pth.tar'):
        if filename.endswith('.pth.tar'):
            filename = filename.replace('.pth.tar', '_gnn.pth.tar')
        else:
            filename = f"{filename}_gnn.pth.tar"
    elif not filename.endswith('.pth.tar'):
        filename = f"{filename}.pth.tar"
        
    return folder_path, filename

def pit_gnn_vs_regular(game_name, config_args):
    log.info(f'Pitting GNN-enhanced model against regular model for {game_name}')
    
    if not has_gnn_version(game_name):
        log.error(f"Game '{game_name}' does not have a GNN version implemented")
        return
    
    # Set up paths
    checkpoint_folder = os.path.join(config_args.checkpoint_path, game_name)
    
    reg_filename = 'best.pth.tar'
    gnn_filename = 'best_gnn.pth.tar'
    
    reg_path = os.path.join(checkpoint_folder, reg_filename)
    gnn_path = os.path.join(checkpoint_folder, gnn_filename)
    
    # Check if models exist
    if not os.path.exists(reg_path):
        log.error(f"Regular model not found at {reg_path}")
        log.info("You need to train a regular model first without the --use_gnn flag")
        log.info("Run: python main.py --game " + game_name)
        sys.exit(1)
        
    if not os.path.exists(gnn_path):
        log.error(f"GNN model not found at {gnn_path}")
        log.info("You need to train a GNN model first using the --use_gnn flag")
        log.info("Run: python main.py --game " + game_name + " --use_gnn")
        sys.exit(1)
    
    # Get the game classes
    GameClass, RegNNetClass = get_game(game_name, use_gnn=False)
    GameClass, GNNNNetClass = get_game(game_name, use_gnn=True)
    
    # Initialize game
    game = create_game_instance(GameClass, config_args)
    
    # Initialize networks
    reg_config = dotdict(config_args.copy())
    reg_config.use_gnn = False
    reg_nnet = RegNNetClass(game, reg_config)
    
    gnn_config = dotdict(config_args.copy())
    gnn_config.use_gnn = True
    gnn_nnet = GNNNNetClass(game, gnn_config)
    
    # Load models
    log.info(f"Loading regular model from {reg_path}")
    reg_nnet.load_checkpoint(checkpoint_folder, reg_filename)
    
    log.info(f"Loading GNN model from {gnn_path}")
    gnn_nnet.load_checkpoint(checkpoint_folder, gnn_filename)
    
    # Initialize MCTS
    reg_mcts = MCTS(game, reg_nnet, reg_config)
    gnn_mcts = MCTS(game, gnn_nnet, gnn_config)
    
    # Set up arena
    arena = Arena(
        lambda x: np.argmax(gnn_mcts.getActionProb(x, temp=0)),
        lambda x: np.argmax(reg_mcts.getActionProb(x, temp=0)), 
        game,
        display=getattr(GameClass, 'display', None)
    )
    
    # Play games
    log.info(f"Playing {config_args.arenaCompare} games...")
    gnn_wins, reg_wins, draws = arena.playGames(config_args.arenaCompare)
    
    log.info('GNN/REGULAR WINS : %d / %d ; DRAWS : %d' % (gnn_wins, reg_wins, draws))
    
    # Calculate win percentage
    total_games = gnn_wins + reg_wins + draws
    if total_games > 0:
        gnn_win_pct = 100 * gnn_wins / total_games
        reg_win_pct = 100 * reg_wins / total_games
        draw_pct = 100 * draws / total_games
        log.info(f"GNN Win %: {gnn_win_pct:.1f}%, Regular Win %: {reg_win_pct:.1f}%, Draw %: {draw_pct:.1f}%")
    
    return gnn_wins, reg_wins, draws

def create_game_instance(GameClass, config_args):
    game_name = config_args.game
    
    if game_name == 'tictactoe':
        return GameClass(n=config_args.board_size)
    elif game_name == 'frozenlake':
        return GameClass(
            map_size=config_args.board_size,
            custom_map=config_args.get('custom_map', None),
            is_slippery=config_args.get('is_slippery', False),
            render_mode=config_args.get('render_mode', None)
        )
    elif game_name == 'connect4':
        return GameClass(board_size=config_args.board_size)
    else:
        return GameClass(**{k: v for k, v in config_args.items() 
                         if k in GameClass.__init__.__code__.co_varnames})

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AlphaZero for Multiple Games')
    parser.add_argument('--game', type=str, required=True, 
                       help=f'Game to train. Available games: {", ".join(list_games())}')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: <game>/config.yaml)')
    parser.add_argument('--load_model', action='store_true', 
                       help='Load the latest model before training')
    
    # GNN-specific arguments
    parser.add_argument('--use_gnn', action='store_true',
                       help='Use Graph Neural Network to enhance MCTS predictions')
    parser.add_argument('--gnn_layers', type=int, default=2,
                       help='Number of message passing layers in the GNN')
    parser.add_argument('--pit_gnn', action='store_true',
                       help='Pit a GNN-enhanced model against a regular model')
    
    # Allow overriding specific config values from command line
    parser.add_argument('--board_size', type=int, help='Override board size from config')
    parser.add_argument('--numIters', type=int, help='Override number of iterations from config')
    parser.add_argument('--numMCTSSims', type=int, help='Override number of MCTS simulations from config')
    
    args = parser.parse_args()
    
    # Check if game exists
    if args.game not in list_games():
        log.error(f"Game '{args.game}' not found in registry. Available games: {list_games()}")
        sys.exit(1)
    
    # Check if GNN is requested but not available
    if args.use_gnn and not has_gnn_version(args.game):
        log.error(f"GNN version of '{args.game}' is not implemented")
        sys.exit(1)
    
    # Set default config path if not provided
    if args.config is None:
        args.config = f"{args.game}/config.yaml"
    
    # Load configuration
    log.info(f'Loading configuration from {args.config}')
    try:
        config = load_config(args.config)
    except Exception as e:
        log.error(f'Error loading configuration: {e}')
        sys.exit(1)
    
    # Convert config to args format
    config_args = config_to_args(config)
    
    # Override config with command line arguments if provided
    if args.board_size is not None:
        config_args.board_size = args.board_size
    if args.numIters is not None:
        config_args.numIters = args.numIters
    if args.numMCTSSims is not None:
        config_args.numMCTSSims = args.numMCTSSims
    
    # Add GNN-specific configuration and game name
    config_args.use_gnn = args.use_gnn
    config_args.gnn_layers = args.gnn_layers
    config_args.game = args.game
    
    # Add load_model flag
    config_args.load_model = args.load_model
    
    # Setup checkpoint directory structure
    checkpoint_folder, best_filename = get_checkpoint_path(
        args.game, 'best', use_gnn=args.use_gnn,
        base_path=config_args.checkpoint_path
    )
    
    # Create the game-specific checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    # Set up checkpoint paths
    config_args.checkpoint = checkpoint_folder
    config_args.load_folder_file = (checkpoint_folder, best_filename)
    
    # Check if we should just pit GNN vs regular
    if args.pit_gnn:
        pit_gnn_vs_regular(args.game, config_args)
        return
    
    # Get the game and neural network classes from registry
    try:
        GameClass, NNetClass = get_game(args.game, use_gnn=args.use_gnn)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)
    
    # Initialize the game
    log.info(f'Creating {args.game} game with board size {config_args.board_size}')
    game = create_game_instance(GameClass, config_args)
    
    # Initialize the neural network
    log.info(f"Initializing Neural Network {'with GNN' if args.use_gnn else ''}...")
    nnet = NNetClass(game, config_args)
    
    # Load model if requested
    if config_args.load_model:
        log.info(f'Loading checkpoint "{config_args.load_folder_file[0]}/{config_args.load_folder_file[1]}"...')
        try:
            nnet.load_checkpoint(config_args.load_folder_file[0], config_args.load_folder_file[1])
        except Exception as e:
            log.warning(f'Could not load model checkpoint: {e}')
            log.warning('Starting with a new model')
    else:
        log.info('Starting with a new model')
    
    # Initialize the coach
    log.info('Initializing the Coach...')
    coach = Coach(game, nnet, config_args)
    
    # Load training examples if model is loaded
    if config_args.load_model:
        log.info("Loading 'trainExamples' from file...")
        try:
            coach.loadTrainExamples()
        except Exception as e:
            log.warning(f'Could not load training examples: {e}')
            log.warning('Starting with empty training examples')
    
    # Start training
    log.info(f'Starting the learning process for {args.game} {"with GNN" if args.use_gnn else ""}')
    try:
        coach.learn()
    except KeyboardInterrupt:
        log.warning('Training interrupted by user')
        # Save model on interrupt
        _, interrupted_filename = get_checkpoint_path(
            args.game, 'interrupted', use_gnn=args.use_gnn
        )
        nnet.save_checkpoint(checkpoint_folder, interrupted_filename)
        log.info(f"Model saved as '{interrupted_filename}'")

if __name__ == "__main__":
    main()