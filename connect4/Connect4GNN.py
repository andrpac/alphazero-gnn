import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from gnn_utils import PolicyValueGNN
from connect4.Connect4Net import Connect4Net, Connect4NNetWrapper

class Connect4GNNWrapper(Connect4NNetWrapper):
    """
    GNN-enhanced wrapper for Connect4 that enhances features from the base network
    to approximate deeper search results.
    """
    def __init__(self, game, args):
        super(Connect4GNNWrapper, self).__init__(game, args)
        
        # Determine feature dimension for Connect4 network
        # Connect4Net has 64 filters in conv2 layer
        self.feature_dim = 64 * self.board_x * self.board_y
        
        # Create the GNN that enhances board features
        num_layers = getattr(args, 'gnn_layers', 2)
        self.gnn = PolicyValueGNN(
            feature_dim=self.feature_dim,
            num_layers=num_layers
        )
        self.gnn.to(self.device)
    
    def extract_features(self, board_tensor):
        """Extract intermediate features from board using base network"""
        # Reshape input
        s = board_tensor.view(-1, 1, self.board_x, self.board_y)
        
        # Pass through convolutional layers
        s = F.relu(self.nnet.conv1(s))
        s = F.relu(self.nnet.conv2(s))
        
        # Flatten
        s = s.view(-1, self.feature_dim)
        
        # Apply dropout as in the original network
        s = F.dropout(s, p=self.nnet.dropout, training=self.nnet.training)
        
        return s
    
    def apply_policy_value_heads(self, features):
        """Apply policy and value heads to processed features"""
        # Policy head
        pi = self.nnet.fc_policy(features)
        pi_log = F.log_softmax(pi, dim=1)
        
        # Value head
        v = torch.tanh(self.nnet.fc_value(features))
        
        return pi_log, v
    
    def predict(self, board):
        """
        Standard prediction without GNN enhancement
        
        Args:
            board: Current board state
        
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
            log_pi, v = self.nnet(board_tensor)
            
            # Convert to numpy
            pi = torch.exp(log_pi).cpu().numpy()[0]
            v = v.cpu().numpy()[0][0]
            
            return pi, v
    
    def predict_with_gnn(self, board):
        """
        GNN-enhanced prediction that enhances features before
        applying the same policy/value heads
        
        Args:
            board: Current board state
        
        Returns:
            pi: Enhanced policy vector
            v: Enhanced value prediction
        """
        # Prepare board tensor
        board_tensor = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board_tensor = board_tensor.view(1, self.board_x, self.board_y)
        
        # Set network to evaluation mode
        self.nnet.eval()
        self.gnn.eval()
        
        with torch.no_grad():
            # Extract board features
            board_features = self.extract_features(board_tensor)
            
            # Apply GNN to enhance features
            enhanced_features = self.gnn(board_features)
            
            # Apply policy and value heads to enhanced features
            log_pi, v = self.apply_policy_value_heads(enhanced_features)
            
            # Convert to numpy
            pi = torch.exp(log_pi).cpu().numpy()[0]
            v = v.cpu().numpy()[0][0]
            
            return pi, v
    
    def train(self, examples, gnn_examples=None):
        """
        Train both the standard network and the GNN.
        The GNN is trained to enhance features to match expanded policy/value.
        
        Args:
            examples: List of standard examples (board, pi, v)
            gnn_examples: List of GNN examples (board, player, std_pi, std_v, expanded_pi, expanded_v, reward)
        """
        # Create optimizers for both networks
        nnet_optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.epochs):
            self.nnet.train()
            self.gnn.train()
            
            # Train standard network on examples
            if examples:
                batch_idx = np.random.randint(0, len(examples), min(len(examples), self.args.batch_size))
                boards, pis, vs = list(zip(*[examples[i] for i in batch_idx]))
                
                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
                
                out_pi, out_v = self.nnet(boards)
                
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = l_pi + l_v
                
                nnet_optimizer.zero_grad()
                total_loss.backward()
                nnet_optimizer.step()
            
            # Train GNN on examples from expanded tree
            if gnn_examples and len(gnn_examples) > 0:
                batch_idx = np.random.randint(0, len(gnn_examples), min(len(gnn_examples), self.args.batch_size))
                batch = [gnn_examples[i] for i in batch_idx]
                
                boards = []
                expanded_pis = []
                expanded_vs = []
                
                # Format: (board, player, std_pi, std_v, expanded_pi, expanded_v, reward)
                for b, _, _, _, expanded_pi, expanded_v, _ in batch:
                    boards.append(b)
                    expanded_pis.append(expanded_pi)
                    expanded_vs.append(expanded_v)
                
                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                expanded_pis = torch.FloatTensor(np.array(expanded_pis)).to(self.device)
                expanded_vs = torch.FloatTensor(np.array(expanded_vs).astype(np.float64)).to(self.device)
                
                # Extract features from the boards
                board_features = self.extract_features(boards)
                
                # Use GNN to enhance features
                enhanced_features = self.gnn(board_features)
                
                # Apply policy and value heads to enhanced features
                gnn_log_pi, gnn_v = self.apply_policy_value_heads(enhanced_features)
                
                # Policy loss - cross entropy between GNN policy and expanded policy
                gnn_pi_loss = -torch.sum(expanded_pis * gnn_log_pi) / expanded_pis.size()[0]
                
                # Value loss - MSE between GNN value and expanded value
                gnn_v_loss = torch.sum((expanded_vs - gnn_v.view(-1)) ** 2) / expanded_vs.size()[0]
                
                # Total loss
                gnn_total_loss = gnn_pi_loss + gnn_v_loss
                
                gnn_optimizer.zero_grad()
                gnn_total_loss.backward()
                gnn_optimizer.step()
    
    def save_checkpoint(self, folder, filename):
        """Save model checkpoint including GNN to file"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'gnn': self.gnn.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        """Load model checkpoint including GNN from file"""
        filepath = os.path.join(folder, filename)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.nnet.load_state_dict(checkpoint['state_dict'])
        
        if 'gnn' in checkpoint:
            self.gnn.load_state_dict(checkpoint['gnn'])
        else:
            print(f"GNN state not found in {filepath}, initializing new GNN")