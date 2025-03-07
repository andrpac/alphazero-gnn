import numpy as np
import random
import time
import argparse
import logging
import sys
import os
from collections import deque

from MCTS import MCTS
from frozenlake.FrozenLakeGame import FrozenLakeGame
from frozenlake.FrozenLakeNet import FrozenLakeNet

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Set recursion limit
sys.setrecursionlimit(3000)

class Args:
    """Arguments for the AlphaZero training process with improved settings"""
    def __init__(self):
        # Training parameters
        self.numIters = 100  # Number of iterations
        self.numEps = 50     # Episodes per iteration - reduced for faster iterations
        self.tempThreshold = 15  # When to switch to lower temperature
        self.updateThreshold = 0.55  # Win rate threshold for model update
        self.maxlenOfQueue = 100000  # Max size of the training buffer
        
        # MCTS parameters
        self.numMCTSSims = 50  # Simulations per move - reduced for speed
        self.cpuct = 1.5  # Exploration constant - balanced
        
        # Neural Network parameters
        self.lr = 0.001  # Learning rate - slightly higher to speed up learning
        self.dropout = 0.2  # Lower dropout for more stable learning
        self.epochs = 15  # More training epochs per iteration
        self.batch_size = 32  # Batch size - must be at least 8 for batch norm
        self.num_channels = 64  # Network size - smaller but adequate


class TrainPipeline:
    """Main training pipeline for AlphaZero-style learning"""
    def __init__(self, args, map_size=4):
        self.args = args
        self.game = FrozenLakeGame(map_size=map_size)
        self.nnet = FrozenLakeNet(self.game, args)
        
        # Training history
        self.trainExamplesHistory = []
        self.win_rates = []
        self.episode_steps = []
        
        # Add evaluations track
        self.evaluations = []
    
    def executeEpisode(self):
        """Execute a single self-play episode"""
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        
        # Reasonable max steps based on map size
        max_steps = self.game.getBoardSize()[0] * self.game.getBoardSize()[1] * 3
        
        # Initialize MCTS
        mcts = MCTS(self.game, self.nnet, self.args)
        visited_states = {}
        
        # Iteration count for exploration schedule
        iteration = len(self.trainExamplesHistory)
        
        while episodeStep < max_steps:
            episodeStep += 1
            
            # Get state string for cycle detection
            board_str = self.game.stringRepresentation(board)
            
            # Handle cycles with a simple strategy - apply small perturbation
            visited_states[board_str] = visited_states.get(board_str, 0) + 1
            if visited_states[board_str] > 2:
                # Simple random action to break cycle
                valid_moves = self.game.getValidMoves(board, self.curPlayer)
                valid_indices = np.where(valid_moves > 0)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                    board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
                    continue
        
            # Calculate temperature - start high, gradually reduce
            temp = max(0.1, 1.0 - (iteration * 0.02)) 
            
            # Get MCTS action probabilities
            try:
                pi = mcts.getActionProb(board, temp=temp)
                
                # Ensure pi is valid
                if np.sum(pi) <= 0 or np.isnan(np.sum(pi)):
                    valid_moves = self.game.getValidMoves(board, self.curPlayer)
                    pi = valid_moves / np.sum(valid_moves)
                    
            except Exception as e:
                log.warning(f"MCTS error: {e}, using valid moves")
                valid_moves = self.game.getValidMoves(board, self.curPlayer)
                pi = valid_moves / np.sum(valid_moves)
            
            # Store transition (state, policy) - we'll add value later
            trainExamples.append([board.copy(), self.curPlayer, pi, None])
            
            # Select action based on the policy
            # With some randomness for exploration
            exploration_factor = max(0.05, 0.3 - (iteration * 0.01))
            
            if np.random.random() < exploration_factor:
                # Random valid action for exploration
                valid_moves = self.game.getValidMoves(board, self.curPlayer)
                valid_indices = np.where(valid_moves > 0)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:
                    # Default action if no valid moves (shouldn't happen)
                    action = 0
            else:
                # Choose from policy
                action = np.random.choice(len(pi), p=pi)
            
            # Execute action
            next_board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            
            # Check if game ended
            gameResult = self.game.getGameEnded(next_board, self.curPlayer)
            
            # Update board
            board = next_board
            
            # If terminal state, assign rewards
            if gameResult != 0:
                # Assign the true outcome to all states
                for i in range(len(trainExamples)):
                    trainExamples[i][3] = gameResult
                
                self.episode_steps.append(episodeStep)
                return trainExamples, gameResult
        
        # Max steps reached - small negative reward
        gameResult = -0.2
        for i in range(len(trainExamples)):
            trainExamples[i][3] = gameResult
            
        self.episode_steps.append(episodeStep)
        return trainExamples, gameResult
    
    def learn(self):
        """Main training loop"""
        log.info('Starting AlphaZero training for FrozenLake')
        
        for i in range(1, self.args.numIters+1):
            log.info(f'Starting Iteration {i}')
            
            # Self-play phase
            iteration_examples = []
            rewards = []
            
            for ep in range(self.args.numEps):
                log.info(f'Episode {ep+1}/{self.args.numEps}')
                try:
                    examples, reward = self.executeEpisode()
                    iteration_examples.extend(examples)
                    rewards.append(reward)
                except Exception as e:
                    log.error(f"Episode error: {e}")
                    continue
            
            if not rewards:
                log.error("All episodes failed, skipping iteration")
                continue
                
            # Calculate statistics - only count true wins/losses
            win_rate = sum(1 for r in rewards if r == 1.0) / len(rewards)
            loss_rate = sum(1 for r in rewards if r == -1.0) / len(rewards)
            timeout_rate = sum(1 for r in rewards if r == -0.2) / len(rewards)
            self.win_rates.append(win_rate)
            
            log.info(f'Completed {len(rewards)}/{self.args.numEps} episodes')
            log.info(f'Win rate: {win_rate:.2f}, Loss rate: {loss_rate:.2f}, Timeout rate: {timeout_rate:.2f}')
            if self.episode_steps:
                log.info(f'Average episode length: {np.mean(self.episode_steps):.2f} steps')
            
            # Train neural network
            if iteration_examples:
                # Save current examples
                self.trainExamplesHistory.append(iteration_examples)
                
                # Keep limited history for more focused learning
                if len(self.trainExamplesHistory) > 5:
                    self.trainExamplesHistory.pop(0)
                
                # Prepare training data
                trainExamples = []
                for e in self.trainExamplesHistory:
                    trainExamples.extend(e)
                random.shuffle(trainExamples)
                
                log.info(f'Training neural network with {len(trainExamples)} examples')
                training_data = [(x[0], x[2], x[3]) for x in trainExamples]
                self.nnet.train(training_data)
                
                # Save checkpoint
                os.makedirs('./checkpoints', exist_ok=True)
                self.nnet.save_checkpoint('./checkpoints', f'checkpoint_{i}.pth.tar')
                
                # Test model every 5 iterations
                if i % 5 == 0 or i == self.args.numIters:
                    eval_win_rate = self.test(num_games=20)
                    self.evaluations.append(eval_win_rate)
                    
                    log.info(f'Evaluation win rate: {eval_win_rate:.2f}')
                    
                    if eval_win_rate > 0.3:  # Lower threshold for saving good models
                        log.info(f'New best model with win rate: {eval_win_rate:.2f}')
                        self.nnet.save_checkpoint('./checkpoints', 'best.pth.tar')
            else:
                log.warning("No valid examples collected, skipping training")
            
            # Early stopping if consistently good performance
            if len(self.win_rates) >= 5 and all(rate > 0.7 for rate in self.win_rates[-5:]):
                log.info('Achieved consistent win rate > 70%. Stopping training.')
                self.nnet.save_checkpoint('./checkpoints', 'best.pth.tar')
                break
    
    def test(self, num_games=20, render=False):
        """Test the trained agent"""
        log.info(f'Testing trained agent for {num_games} games')
        wins = 0
        losses = 0
        timeouts = 0
        steps = []
        
        for game_num in range(num_games):
            board = self.game.getInitBoard()
            player = 1
            done = False
            step = 0
            max_steps = self.game.getBoardSize()[0] * self.game.getBoardSize()[1] * 5
            
            # Fresh MCTS for testing with more simulations
            args_copy = self.args
            args_copy.numMCTSSims = 150  # More planning for testing
            mcts = MCTS(self.game, self.nnet, args_copy)
            visited_states = {}
            
            while not done and step < max_steps:
                step += 1
                
                # Render if requested
                if render and game_num == 0:
                    self.game.board = board
                    self.game.render()
                    time.sleep(0.3)
                
                # Handle cycles
                board_str = self.game.stringRepresentation(board)
                visited_states[board_str] = visited_states.get(board_str, 0) + 1
                
                if visited_states[board_str] > 3:
                    # Break cycle with direct goal movement
                    pos = np.unravel_index(np.argmax(board), board.shape)
                    goal_pos = self.game.goal_pos
                    
                    # Simple directional bias
                    if pos[0] < goal_pos[0]:
                        action = 2  # Down
                    elif pos[1] < goal_pos[1]:
                        action = 1  # Right
                    elif pos[0] > goal_pos[0]:
                        action = 0  # Up
                    else:
                        action = 3  # Left
                    
                    # Check if action is valid
                    valid_moves = self.game.getValidMoves(board, player)
                    if valid_moves[action] == 0:
                        valid_indices = np.where(valid_moves > 0)[0]
                        action = np.random.choice(valid_indices) if len(valid_indices) > 0 else 0
                else:
                    # Use MCTS with low temperature for exploitation
                    try:
                        pi = mcts.getActionProb(board, temp=0.1)
                        action = np.random.choice(len(pi), p=pi)
                    except Exception as e:
                        log.warning(f"Test error: {e}")
                        # Fallback to goal-directed action
                        pos = np.unravel_index(np.argmax(board), board.shape)
                        goal_pos = self.game.goal_pos
                        
                        if pos[0] < goal_pos[0]:
                            action = 2  # Down
                        elif pos[1] < goal_pos[1]:
                            action = 1  # Right
                        elif pos[0] > goal_pos[0]:
                            action = 0  # Up
                        else:
                            action = 3  # Left
                            
                        # Check if action is valid
                        valid_moves = self.game.getValidMoves(board, player)
                        if valid_moves[action] == 0:
                            valid_indices = np.where(valid_moves > 0)[0]
                            action = np.random.choice(valid_indices) if len(valid_indices) > 0 else 0
                
                # Execute action
                board, player = self.game.getNextState(board, player, action)
                
                # Check if game ended - only terminal states
                result = self.game.getGameEnded(board, player)
                if result == 1.0:
                    done = True
                    wins += 1
                    steps.append(step)
                elif result == -1.0:
                    done = True
                    losses += 1
                    steps.append(step)
            
            # Handle max steps reached
            if not done:
                timeouts += 1
                steps.append(step)
        
        # Calculate statistics
        win_rate = wins / num_games if num_games > 0 else 0
        loss_rate = losses / num_games if num_games > 0 else 0
        timeout_rate = timeouts / num_games if num_games > 0 else 0
        
        log.info(f'Test Results: {num_games} games')
        log.info(f'Win rate: {win_rate:.2f}')
        log.info(f'Loss rate: {loss_rate:.2f}')
        log.info(f'Timeout rate: {timeout_rate:.2f}')
        if steps:
            log.info(f'Average steps: {np.mean(steps):.2f}')
        
        return win_rate


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AlphaZero for FrozenLake')
    parser.add_argument('--map_size', type=int, default=4, choices=[4, 8], help='Size of the FrozenLake map (4 or 8)')
    parser.add_argument('--iterations', type=int, default=50, help='Number of training iterations')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per iteration')
    parser.add_argument('--mcts_sims', type=int, default=100, help='Number of MCTS simulations per move')
    parser.add_argument('--load_model', action='store_true', help='Load the latest model before training')
    parser.add_argument('--test_only', action='store_true', help='Run only testing, no training')
    parser.add_argument('--render', action='store_true', help='Render the environment during testing')
    
    cmd_args = parser.parse_args()
    
    # Set up training arguments
    args = Args()
    args.numIters = cmd_args.iterations
    args.numEps = cmd_args.episodes
    args.numMCTSSims = cmd_args.mcts_sims
    
    # Create pipeline
    pipeline = TrainPipeline(args, map_size=cmd_args.map_size)
    
    # Load model if requested
    if cmd_args.load_model:
        os.makedirs('./checkpoints', exist_ok=True)
        try:
            pipeline.nnet.load_checkpoint('./checkpoints', 'best.pth.tar')
            log.info('Loaded best model')
        except:
            log.warning('Could not load best model, starting from scratch')
    
    # Run testing or training
    if cmd_args.test_only:
        pipeline.test(num_games=100, render=cmd_args.render)
    else:
        pipeline.learn()
        # Save best model and test
        pipeline.nnet.save_checkpoint('./checkpoints', 'best.pth.tar')
        pipeline.test(num_games=100, render=cmd_args.render)


if __name__ == "__main__":
    main()