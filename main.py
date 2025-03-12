import argparse
import logging
import coloredlogs
import os

from Coach import Coach
from MCTS import MCTS
from frozenlake.FrozenLakeGame import FrozenLakeGame
from frozenlake.FrozenLakeNet import FrozenLakeNet

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# A simple dotdict implementation for args
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
    def __setattr__(self, name, value):
        self[name] = value

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AlphaZero for FrozenLake with GNN enhancements')
    parser.add_argument('--map_size', type=int, default=4, choices=[4, 8], 
                        help='Size of the FrozenLake map (4 or 8)')
    parser.add_argument('--numIters', type=int, default=30, 
                        help='Number of training iterations')
    parser.add_argument('--numEps', type=int, default=50, 
                        help='Number of episodes per iteration')
    parser.add_argument('--numMCTSSims', type=int, default=50, 
                        help='Number of MCTS simulations per move')
    parser.add_argument('--arenaCompare', type=int, default=30, 
                        help='Number of games to play for model comparison')
    parser.add_argument('--cpuct', type=float, default=2.0, 
                        help='Exploration constant for MCTS')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/', 
                        help='Directory to save checkpoints')
    parser.add_argument('--load_model', action='store_true', 
                        help='Load the latest model before training')
    # GNN-specific parameters
    parser.add_argument('--gnn_layers', type=int, default=2, 
                        help='Number of GNN layers for message passing')
    parser.add_argument('--embedding_dim', type=int, default=64, 
                        help='Dimension of state embeddings for GNN')
    
    args = parser.parse_args()
    
    # Convert args to dotdict for easier access
    args = dotdict({
        'numIters': args.numIters,
        'numEps': args.numEps,
        'tempThreshold': 10,  # Lower temp threshold for faster convergence
        'updateThreshold': 0.6,  # Higher threshold for more confident updates
        'maxlenOfQueue': 100000,  # Reduced memory usage
        'numMCTSSims': args.numMCTSSims,
        'arenaCompare': args.arenaCompare,
        'cpuct': args.cpuct,
        'checkpoint': args.checkpoint,
        'load_model': args.load_model,
        'numItersForTrainExamplesHistory': 3,  # Reduced memory usage
        'lr': 0.001,  # Smaller learning rate for stability
        'dropout': 0.2,  # Less dropout for FrozenLake
        'epochs': 15,  # Fewer epochs to prevent overfitting
        'batch_size': 32,
        'num_channels': 32,  # Fewer channels for simpler game
        'map_size': args.map_size,
        # GNN-specific parameters
        'gnn_layers': args.gnn_layers,
        'embedding_dim': args.embedding_dim,
    })
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    # Set up load_folder_file for loading models/examples
    args.load_folder_file = (args.checkpoint, 'best.pth.tar')
    
    # Initialize the game
    log.info(f'Creating FrozenLake game with map size {args.map_size}x{args.map_size}')
    game = FrozenLakeGame(map_size=args.map_size)
    
    # Initialize the neural network
    log.info('Initializing Neural Network with GNN enhancements...')
    nnet = FrozenLakeNet(game, args)
    
    # Load model if requested
    if args.load_model:
        log.info(f'Loading checkpoint "{args.load_folder_file[0]}/{args.load_folder_file[1]}"...')
        try:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        except Exception as e:
            log.warning(f'Could not load model checkpoint: {e}')
            log.warning('Starting with a new model')
    else:
        log.info('Starting with a new model')
    
    # Initialize the coach
    log.info('Initializing the Coach...')
    coach = Coach(game, nnet, args)
    
    # Load training examples if model is loaded
    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        try:
            coach.loadTrainExamples()
        except Exception as e:
            log.warning(f'Could not load training examples: {e}')
            log.warning('Starting with empty training examples')
    
    # Start training
    log.info('Starting the learning process with GNN enhancement')
    try:
        coach.learn()
    except KeyboardInterrupt:
        log.warning('Training interrupted by user')
        # Save model on interrupt
        nnet.save_checkpoint(args.checkpoint, 'interrupted.pth.tar')
        log.info("Model saved as 'interrupted.pth.tar'")

if __name__ == "__main__":
    main()