import argparse
import logging
import os
import sys
import numpy as np
import re
import matplotlib.pyplot as plt  # Added for plotting

from Arena import Arena
from MCTS import MCTS
from register import get_game, list_games, has_gnn_version
from main import dotdict, load_config, config_to_args, get_checkpoint_path, create_game_instance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def get_checkpoint_number(filename):
    """Extract the checkpoint number from a filename."""
    # Try to match "checkpoint_X" pattern
    match = re.search(r'checkpoint_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # Try to match "best" or "best_gnn" pattern
    if "best_gnn" in filename:
        return "best_gnn"
    elif "best" in filename:
        return "best"
    
    return None

def extract_checkpoint_info(checkpoint_path):
    """Extract checkpoint number and whether it's a GNN checkpoint from path."""
    filename = os.path.basename(checkpoint_path)
    is_gnn = "_gnn.pth.tar" in filename
    checkpoint_num = get_checkpoint_number(filename)
    return checkpoint_num, is_gnn

def list_all_checkpoints(folder, use_gnn=False):
    """List all available checkpoints in a folder, sorted by checkpoint number."""
    if not os.path.exists(folder):
        return []
    
    checkpoints = []
    suffix = "_gnn.pth.tar" if use_gnn else ".pth.tar"
    
    for filename in os.listdir(folder):
        if filename.endswith(suffix) and ("checkpoint_" in filename or filename.startswith("best")):
            checkpoint_id = get_checkpoint_number(filename)
            if checkpoint_id is not None:
                checkpoints.append((checkpoint_id, filename))
    
    # Sort by checkpoint number (only numeric checkpoints)
    numeric_checkpoints = [(n, f) for n, f in checkpoints if isinstance(n, int)]
    special_checkpoints = [(n, f) for n, f in checkpoints if not isinstance(n, int)]
    
    numeric_checkpoints.sort(key=lambda x: x[0])
    return numeric_checkpoints + special_checkpoints

def compare_reference_vs_all(game_name, config_args, ref_checkpoint_id, ref_is_gnn=False, num_games=20):
    """Compare reference checkpoint against all other checkpoints."""
    # Get checkpoint path
    checkpoint_folder = os.path.join(config_args.checkpoint_path, game_name)
    
    # Define reference checkpoint path
    if isinstance(ref_checkpoint_id, int):
        if ref_is_gnn:
            ref_filename = f"checkpoint_{ref_checkpoint_id}_gnn.pth.tar"
        else:
            ref_filename = f"checkpoint_{ref_checkpoint_id}.pth.tar"
    else:
        # Handle special checkpoint IDs like "best" or "best_gnn"
        if ref_checkpoint_id == "best_gnn":
            ref_filename = "best_gnn.pth.tar"
            ref_is_gnn = True
        else:  # "best"
            ref_filename = "best.pth.tar"
    
    ref_path = os.path.join(checkpoint_folder, ref_filename)
    
    # Check if reference checkpoint exists
    if not os.path.exists(ref_path):
        log.error(f"Reference checkpoint {ref_path} not found!")
        sys.exit(1)
    
    # List all checkpoints
    standard_checkpoints = list_all_checkpoints(checkpoint_folder, use_gnn=False)
    gnn_checkpoints = list_all_checkpoints(checkpoint_folder, use_gnn=True)
    
    # Initialize the game
    log.info(f'Creating {game_name} game with board size {config_args.board_size}')
    GameClass, _ = get_game(game_name, use_gnn=False)
    game = create_game_instance(GameClass, config_args)
    
    # Get NNet classes
    _, StandardNNetClass = get_game(game_name, use_gnn=False)
    _, GNNNNetClass = get_game(game_name, use_gnn=True)
    
    # Initialize reference model
    ref_config = dotdict(config_args.copy())
    ref_config.use_gnn = ref_is_gnn
    
    if ref_is_gnn:
        ref_nnet = GNNNNetClass(game, ref_config)
    else:
        ref_nnet = StandardNNetClass(game, ref_config)
    
    log.info(f"Loading reference model from {ref_path}")
    ref_nnet.load_checkpoint(checkpoint_folder, ref_filename)
    ref_mcts = MCTS(game, ref_nnet, ref_config)
    
    # Create dict to store comparison results
    results = {
        'standard': [],  # Will store (checkpoint_id, win_rate, wins, losses, draws)
        'gnn': []        # Will store (checkpoint_id, win_rate, wins, losses, draws)
    }
    
    # Compare reference vs all standard checkpoints
    log.info(f"Comparing reference {'GNN' if ref_is_gnn else 'standard'} checkpoint {ref_checkpoint_id} vs all standard checkpoints")
    
    for checkpoint_id, filename in standard_checkpoints:
        # Skip self-comparison for same type
        if checkpoint_id == ref_checkpoint_id and not ref_is_gnn:
            log.info(f"Skipping self-comparison: Ref {ref_checkpoint_id} vs Std {checkpoint_id}")
            continue
            
        # Print clear message about which checkpoints are fighting
        log.info(f"MATCH: Reference {ref_checkpoint_id} {'(GNN)' if ref_is_gnn else ''} vs Standard {checkpoint_id}")
        
        # Load challenger model
        challenger_config = dotdict(config_args.copy())
        challenger_config.use_gnn = False
        challenger_nnet = StandardNNetClass(game, challenger_config)
        
        challenger_nnet.load_checkpoint(checkpoint_folder, filename)
        challenger_mcts = MCTS(game, challenger_nnet, challenger_config)
        
        # Set up arena and play games
        arena = Arena(
            lambda x: np.argmax(ref_mcts.getActionProb(x, temp=0)),
            lambda x: np.argmax(challenger_mcts.getActionProb(x, temp=0)),
            game
        )
        
        ref_wins, challenger_wins, draws = arena.playGames(num_games)
        win_rate = (ref_wins + 0.5 * draws) / num_games
        
        # Store the result
        if isinstance(checkpoint_id, int):
            results['standard'].append((checkpoint_id, win_rate, ref_wins, challenger_wins, draws))
        
        # Log the result
        log.info(f"RESULT: Ref {ref_checkpoint_id} vs Std {checkpoint_id}: Win rate: {win_rate:.3f} ({ref_wins}/{challenger_wins}/{draws})")

    # Compare reference vs all GNN checkpoints
    if gnn_checkpoints:
        log.info(f"Comparing reference {'GNN' if ref_is_gnn else 'standard'} checkpoint {ref_checkpoint_id} vs all GNN checkpoints")
        
        for checkpoint_id, filename in gnn_checkpoints:
            # Skip self-comparison for same type
            if checkpoint_id == ref_checkpoint_id and ref_is_gnn:
                log.info(f"Skipping self-comparison: Ref {ref_checkpoint_id} vs GNN {checkpoint_id}")
                continue
                
            # Print clear message about which checkpoints are fighting
            log.info(f"MATCH: Reference {ref_checkpoint_id} {'(GNN)' if ref_is_gnn else ''} vs GNN {checkpoint_id}")
            
            # Load challenger model
            challenger_config = dotdict(config_args.copy())
            challenger_config.use_gnn = True
            challenger_nnet = GNNNNetClass(game, challenger_config)
            
            challenger_nnet.load_checkpoint(checkpoint_folder, filename)
            challenger_mcts = MCTS(game, challenger_nnet, challenger_config)
            
            # Set up arena and play games
            arena = Arena(
                lambda x: np.argmax(ref_mcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(challenger_mcts.getActionProb(x, temp=0)),
                game
            )
            
            ref_wins, challenger_wins, draws = arena.playGames(num_games)
            win_rate = (ref_wins + 0.5 * draws) / num_games
            
            # Store the result
            if isinstance(checkpoint_id, int):
                results['gnn'].append((checkpoint_id, win_rate, ref_wins, challenger_wins, draws))
            
            # Log the result
            log.info(f"RESULT: Ref {ref_checkpoint_id} vs GNN {checkpoint_id}: Win rate: {win_rate:.3f} ({ref_wins}/{challenger_wins}/{draws})")
    
    log.info("All comparisons completed")
    
    # Create and save plot
    create_comparison_plot(results, game_name, ref_checkpoint_id, ref_is_gnn)
    
    return results

def create_comparison_plot(results, game_name, ref_checkpoint_id, ref_is_gnn):
    """Create a plot visualizing the comparison results and save it to a file."""
    plt.figure(figsize=(12, 6))
    
    # Plot standard model results
    if results['standard']:
        checkpoints, win_rates, _, _, _ = zip(*sorted(results['standard']))
        plt.plot(checkpoints, win_rates, 'o-', color='blue', label='vs Standard', markersize=8)
    
    # Plot GNN model results
    if results['gnn']:
        checkpoints, win_rates, _, _, _ = zip(*sorted(results['gnn']))
        plt.plot(checkpoints, win_rates, 's-', color='green', label='vs GNN', markersize=8)
    
    # Add reference line at 0.5 (equal performance)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Equal strength')
    
    # Add labels and title
    plt.xlabel('Checkpoint Iteration')
    plt.ylabel('Win Rate of Reference Model')
    ref_type = "GNN" if ref_is_gnn else "Standard"
    plt.title(f'{game_name}: Performance of Reference {ref_type} Model (Checkpoint {ref_checkpoint_id})')
    
    # Adjust y-axis
    plt.ylim(0, 1)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    output_dir = os.path.join("plots", game_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{game_name}_ref_{ref_checkpoint_id}_{ref_type.lower()}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    log.info(f"Plot saved to {output_path}")
    
    # Show plot labels for numeric checkpoints
    plt.xticks(range(1, max([c[0] for c in results['standard'] + results['gnn'] if isinstance(c[0], int)], default=1) + 1))
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Compare checkpoint performance against others')
    parser.add_argument('--game', type=str, required=True, help='Game name (e.g., connect4)')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Reference checkpoint number or path to checkpoint file')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--use_gnn', '--use-gnn', action='store_true', help='Use GNN checkpoint as reference (default: standard)')
    parser.add_argument('--num_games', '--num-games', type=int, default=20, 
                       help='Total number of games to play for each comparison (default: 20, split equally between reference-first and opponent-first)')
    
    args = parser.parse_args()
    
    # Set default config path if not provided
    if args.config is None:
        args.config = f"{args.game}/config.yaml"
    
    # Load configuration
    try:
        config = load_config(args.config)
        config_args = config_to_args(config)
    except Exception as e:
        log.error(f'Error loading configuration: {e}')
        sys.exit(1)
    
    # Add game name to config
    config_args.game = args.game
    
    # Parse checkpoint argument
    ref_checkpoint_id = args.checkpoint
    ref_is_gnn = args.use_gnn
    
    # If checkpoint is a path, extract the information
    if os.path.exists(args.checkpoint):
        checkpoint_id, is_gnn = extract_checkpoint_info(args.checkpoint)
        ref_checkpoint_id = checkpoint_id
        ref_is_gnn = is_gnn
    elif args.checkpoint.isdigit():
        ref_checkpoint_id = int(args.checkpoint)
    elif args.checkpoint in ["best", "best_gnn"]:
        ref_checkpoint_id = args.checkpoint
        ref_is_gnn = "gnn" in args.checkpoint
    
    # Run comparisons
    compare_reference_vs_all(
        args.game, 
        config_args, 
        ref_checkpoint_id,
        ref_is_gnn=ref_is_gnn,
        num_games=args.num_games
    )

if __name__ == "__main__":
    main()