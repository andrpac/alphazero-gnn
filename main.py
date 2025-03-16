import argparse
import logging
import coloredlogs
import os
import yaml
import sys

from Coach import Coach
from Arena import Arena
from register import get_game, list_games

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# A simple dotdict implementation for args
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
    def __setattr__(self, name, value):
        self[name] = value

def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config_to_args(config):
    """Convert config dictionary to dotdict for compatibility"""
    args = dotdict({})
    
    # Process all sections of the config
    for section in config:
        for key, value in config[section].items():
            args[key] = value
    
    # Add backward compatibility for checkpoint paths
    if 'checkpoint_path' in args and 'checkpoint' not in args:
        args.checkpoint = args.checkpoint_path
    elif 'checkpoint' in args and 'checkpoint_path' not in args:
        args.checkpoint_path = args.checkpoint
    
    return args

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AlphaZero for Multiple Games')
    parser.add_argument('--game', type=str, required=True, 
                        help=f'Game to train. Available games: {", ".join(list_games())}')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (default: configs/<game>_config.yaml)')
    parser.add_argument('--load_model', action='store_true', 
                        help='Load the latest model before training')
    
    # Allow overriding specific config values from command line
    parser.add_argument('--board_size', type=int, help='Override board size from config')
    parser.add_argument('--numIters', type=int, help='Override number of iterations from config')
    parser.add_argument('--numMCTSSims', type=int, help='Override number of MCTS simulations from config')
    
    args = parser.parse_args()
    
    # Set default config path if not provided
    if args.config is None:
        args.config = f"configs/{args.game}_config.yaml"
    
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
    
    # Add load_model flag
    config_args.load_model = args.load_model
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(config_args.checkpoint_path, args.game)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    config_args.load_folder_file = (checkpoint_dir, 'best.pth.tar')
    
    # Get the game and neural network classes from registry
    try:
        GameClass, NNetClass = get_game(args.game)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)
    
    # Initialize the game
    log.info(f'Creating {args.game} game with board size {config_args.board_size}')
    
    # Initialize game with appropriate parameters
    if args.game == 'tictactoe':
        game = GameClass(n=config_args.board_size)
    elif args.game == 'frozenlake':
        game = GameClass(
            map_size=config_args.board_size,
            custom_map=config_args.get('custom_map', None),
            is_slippery=config_args.get('is_slippery', False),
            render_mode=config_args.get('render_mode', None)
        )
    else:
        # Generic initialization for future games
        # This assumes the Game constructor can take all config args as kwargs
        game = GameClass(**{k: v for k, v in config_args.items() 
                          if k in GameClass.__init__.__code__.co_varnames})
    
    # Initialize the neural network
    log.info('Initializing Neural Network...')
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
    log.info(f'Starting the learning process for {args.game}')
    try:
        coach.learn()
    except KeyboardInterrupt:
        log.warning('Training interrupted by user')
        # Save model on interrupt
        nnet.save_checkpoint(checkpoint_dir, 'interrupted.pth.tar')
        log.info("Model saved as 'interrupted.pth.tar'")

if __name__ == "__main__":
    main()