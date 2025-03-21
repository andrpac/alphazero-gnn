import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from gnn_utils import GNNProcessor
from tictactoe.TicTacToeNet import TicTacToeNet, TicTacToeNNetWrapper

class TicTacToeGNNWrapper(TicTacToeNNetWrapper):
    """
    GNN-enhanced wrapper for TicTacToe that processes visited nodes in MCTS path.
    """
    def __init__(self, game, args):
        super(TicTacToeGNNWrapper, self).__init__(game, args)
        
        # Determine feature dimension for TicTacToe network
        self.feature_dim = 128 * (self.board_x-2) * (self.board_y-2)
        
        # Create GNN processor
        num_layers = getattr(args, 'gnn_layers', 2)
        self.gnn_processor = GNNProcessor(self.feature_dim, num_layers)
        self.gnn_processor.to(self.device)
    
    def extract_features(self, board_tensor):
        """Extract intermediate features from board using base network"""
        # Reshape input
        s = board_tensor.view(-1, 1, self.board_x, self.board_y)
        
        # Pass through convolutional layers (TicTacToe has 3 conv layers)
        s = F.relu(self.nnet.conv1(s))
        s = F.relu(self.nnet.conv2(s))
        s = F.relu(self.nnet.conv3(s))
        
        # Flatten
        s = s.view(-1, self.feature_dim)
        return s
    
    def process_with_policy_value_heads(self, features):
        """Apply policy and value heads to processed features"""
        # TicTacToe has separate fully connected layers before policy/value heads
        # Policy head
        pi = F.relu(self.nnet.fc1(features))
        pi = self.nnet.fc_policy(pi)
        pi = F.log_softmax(pi, dim=1)
        
        # Value head
        v = F.relu(self.nnet.fc2(features))
        v = torch.tanh(self.nnet.fc_value(v))
        
        return pi, v
    
    def predict(self, board, path_boards=None, path_depths=None):
        """
        Enhanced predict method that uses GNN to process visited MCTS nodes
        
        Args:
            board: Current board state
            path_boards: List of boards from the MCTS search path
            path_depths: List of depths from the MCTS search path (optional)
        
        Returns:
            pi: Policy vector
            v: Value prediction
        """
        # Prepare board tensor
        board_tensor = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board_tensor = board_tensor.view(1, self.board_x, self.board_y)
        
        # Set network to evaluation mode
        self.nnet.eval()
        
        with torch.no_grad():
            # Check if we should use GNN with path states
            if path_boards and len(path_boards) > 0:
                # Extract features for the target board
                target_features = self.extract_features(board_tensor)
                
                # Extract features for path states
                path_features = []
                for path_board in path_boards:
                    board_t = torch.FloatTensor(path_board.astype(np.float64)).to(self.device)
                    board_t = board_t.view(1, self.board_x, self.board_y)
                    path_features.append(self.extract_features(board_t))
                
                # Stack all features (target board first, then path boards)
                all_features = torch.cat([target_features] + path_features, dim=0)
                
                # Prepare depths tensor if provided
                depths_tensor = None
                if path_depths is not None and len(path_depths) > 0:
                    depths_tensor = torch.tensor([0] + path_depths, dtype=torch.long, device=self.device)
                
                # Process with GNN
                updated_features = self.gnn_processor(all_features, depths_tensor)
                
                # Get updated features for target board (index 0)
                updated_target_features = updated_features[0:1]
                
                # Apply policy and value heads to updated features
                log_pi, v = self.process_with_policy_value_heads(updated_target_features)
            else:
                # Use standard neural network
                log_pi, v = self.nnet(board_tensor)
            
            # Convert to numpy
            pi = torch.exp(log_pi).cpu().numpy()[0]
            v = v.cpu().numpy()[0][0]
            
            return pi, v
    
    def save_checkpoint(self, folder, filename):
        """Save checkpoint including GNN"""
        # Create folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # Save base network and GNN in the same file
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'gnn_processor': self.gnn_processor.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        """Load checkpoint including GNN"""
        filepath = os.path.join(folder, filename)
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load base network
        self.nnet.load_state_dict(checkpoint['state_dict'])
        
        # Load GNN processor if available
        if 'gnn_processor' in checkpoint:
            self.gnn_processor.load_state_dict(checkpoint['gnn_processor'])
        else:
            print(f"GNN processor state not found in {filepath}, initializing new GNN processor")