import torch
import torch.optim as optim
import numpy as np
import os

from gnn_utils import PolicyValueGNN
from tictactoe.TicTacToeNet import TicTacToeNNetWrapper

class TicTacToeGNNWrapper(TicTacToeNNetWrapper):
    def __init__(self, game, args):
        # Initialize TicTacToe neural network with GNN enhancement
        super(TicTacToeGNNWrapper, self).__init__(game, args)
        
        self.feature_dim = 128 * (self.board_x-2) * (self.board_y-2)
        
        num_layers = getattr(args, 'gnn_layers', 2)
        self.gnn = PolicyValueGNN(
            feature_dim=self.feature_dim,
            num_layers=num_layers
        )
        self.gnn.to(self.device)
    
    def predict_with_gnn(self, board):
        # Predict policy and value for a board using GNN enhancement
        board_tensor = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board_tensor = board_tensor.view(1, self.board_x, self.board_y)
        
        self.nnet.eval()
        self.gnn.eval()
        
        with torch.no_grad():
            features = self.nnet.extract_features(board_tensor)
            enhanced_features = self.gnn(features)
            log_pi, v = self.nnet.policy_value_heads(enhanced_features)
            
            pi = torch.exp(log_pi).cpu().numpy()[0]
            v = v.cpu().numpy()[0][0]
            
            return pi, v
    
    def train(self, examples, gnn_examples=None):
        # Train both standard network and GNN components
        nnet_optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.epochs):
            self.nnet.train()
            self.gnn.train()
            
            # Train standard network
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
            
            # Train GNN
            if gnn_examples and len(gnn_examples) > 0:
                batch_idx = np.random.randint(0, len(gnn_examples), min(len(gnn_examples), self.args.batch_size))
                batch = [gnn_examples[i] for i in batch_idx]
                
                boards = []
                expanded_pis = []
                expanded_vs = []
                
                for b, _, _, _, expanded_pi, expanded_v, _ in batch:
                    boards.append(b)
                    expanded_pis.append(expanded_pi)
                    expanded_vs.append(expanded_v)
                
                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                expanded_pis = torch.FloatTensor(np.array(expanded_pis)).to(self.device)
                expanded_vs = torch.FloatTensor(np.array(expanded_vs).astype(np.float64)).to(self.device)
                
                features = self.nnet.extract_features(boards)
                enhanced_features = self.gnn(features)
                gnn_log_pi, gnn_v = self.nnet.policy_value_heads(enhanced_features)
                
                gnn_pi_loss = -torch.sum(expanded_pis * gnn_log_pi) / expanded_pis.size()[0]
                gnn_v_loss = torch.sum((expanded_vs - gnn_v.view(-1)) ** 2) / expanded_vs.size()[0]
                gnn_total_loss = gnn_pi_loss + gnn_v_loss
                
                gnn_optimizer.zero_grad()
                gnn_total_loss.backward()
                gnn_optimizer.step()
    
    def save_checkpoint(self, folder, filename):
        # Save model checkpoint including both neural network and GNN weights
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'gnn': self.gnn.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        # Load model checkpoint including GNN weights if available
        filepath = os.path.join(folder, filename)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.nnet.load_state_dict(checkpoint['state_dict'])
        
        if 'gnn' in checkpoint:
            self.gnn.load_state_dict(checkpoint['gnn'])
        else:
            print(f"GNN layer state not found in {filepath}, initializing new GNN")