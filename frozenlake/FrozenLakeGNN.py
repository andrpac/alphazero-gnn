import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from frozenlake.FrozenLakeNet import FrozenLakeNet

class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, adj):
        support = self.W(x)
        output = torch.bmm(adj, support)
        return F.relu(output)

class PolicyValueGNN(nn.Module):
    def __init__(self, feature_dim, num_layers=2):
        super(PolicyValueGNN, self).__init__()
        self.feature_dim = feature_dim
        
        self.gnn_layers = nn.ModuleList([
            GNNLayer(feature_dim, feature_dim) 
            for _ in range(num_layers)
        ])
        
        self.output_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features, adjacency):
        node_embeddings = features.unsqueeze(0)
        
        for gnn_layer in self.gnn_layers:
            node_embeddings = gnn_layer(node_embeddings, adjacency)
        
        current_embedding = node_embeddings[:, 0, :]
        enhanced_features = self.output_transform(current_embedding)
        
        return enhanced_features

class FrozenLakeGNNNet(FrozenLakeNet):
    def __init__(self, game, args):
        super(FrozenLakeGNNNet, self).__init__(game, args)
        
        self.embedding_dim = getattr(args, 'embedding_dim', 64)
        
        num_gnn_layers = getattr(args, 'gnn_layers', 2)
        self.gnn = PolicyValueGNN(
            feature_dim=self.embedding_dim,
            num_layers=num_gnn_layers
        )
        self.gnn.to(self.device)
        
        self.gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=args.lr)
    
    def predict_with_gnn(self, board):
        self.nnet.eval()
        self.gnn.eval()
        
        board_tensor = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        if len(board_tensor.shape) == 2:
            board_tensor = board_tensor.unsqueeze(0)
        
        neighbor_states = [board]
        valids = self.game.getValidMoves(board, 1)
        
        for action in range(self.action_size):
            if valids[action]:
                next_board, _ = self.game.getNextState(board, 1, action)
                canonical_next = self.game.getCanonicalForm(next_board, 1)
                neighbor_states.append(canonical_next)
        
        neighbor_tensors = torch.FloatTensor(np.array(neighbor_states)).to(self.device)
        
        adjacency = self.create_adjacency(len(neighbor_states))
            
        with torch.no_grad():
            try:
                # Extract features for all states
                neighbor_features = self.nnet.extract_features(neighbor_tensors)
                
                # Apply GNN to enhance features
                enhanced_features = self.gnn(neighbor_features, adjacency.unsqueeze(0))
                
                # Apply policy and value heads
                pi, v = self.nnet.policy_value_heads(enhanced_features)
                
                if torch.isnan(pi).any():
                    pi = torch.ones_like(pi) / pi.size(1)
                if torch.isnan(v).any():
                    v = torch.zeros_like(v)
                    
                return pi.detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]
            except Exception as e:
                print(f"Error in GNN prediction: {e}")
                return np.ones(self.action_size) / self.action_size, 0.0
    
    def train(self, examples, gnn_examples=None):
        # Train base network
        super().train(examples)
        
        # Train GNN if examples are provided
        if gnn_examples and len(gnn_examples) > 0:
            self.nnet.eval()  # Freeze base network
            self.gnn.train()
            
            print(f"Training GNN with {len(gnn_examples)} examples")
            
            for epoch in range(self.args.epochs):
                np.random.shuffle(gnn_examples)
                
                batch_size = min(len(gnn_examples), self.args.batch_size)
                batches = [gnn_examples[i:i + batch_size] for i in range(0, len(gnn_examples), batch_size)]
                
                epoch_loss = 0
                num_batches = 0
                
                for batch in batches:
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
                    
                    batch_pi_outputs = []
                    batch_v_outputs = []
                    
                    for board_idx in range(boards.shape[0]):
                        single_board = boards[board_idx].unsqueeze(0)
                        board_np = single_board.squeeze(0).cpu().numpy()
                        
                        neighbor_states = [board_np]
                        valids = self.game.getValidMoves(board_np, 1)
                        
                        for action in range(self.action_size):
                            if valids[action]:
                                next_board, _ = self.game.getNextState(board_np, 1, action)
                                canonical_next = self.game.getCanonicalForm(next_board, 1)
                                neighbor_states.append(canonical_next)
                        
                        neighbor_tensors = torch.FloatTensor(np.array(neighbor_states)).to(self.device)
                        
                        adjacency = self.create_adjacency(len(neighbor_states))
                        
                        # Extract features using base network
                        with torch.no_grad():
                            neighbor_features = self.nnet.extract_features(neighbor_tensors)
                        
                        # Apply GNN enhancement
                        enhanced_features = self.gnn(neighbor_features, adjacency.unsqueeze(0))
                        
                        # Apply policy and value heads
                        pi, v = self.nnet.policy_value_heads(enhanced_features)
                        
                        batch_pi_outputs.append(pi)
                        batch_v_outputs.append(v)
                    
                    out_pi = torch.cat(batch_pi_outputs, dim=0)
                    out_v = torch.cat(batch_v_outputs, dim=0)
                    
                    self.gnn_optimizer.zero_grad()
                    
                    gnn_pi_loss = -torch.mean(torch.sum(expanded_pis * torch.log(out_pi.clamp(min=1e-8)), dim=1))
                    gnn_v_loss = F.mse_loss(out_v.view(-1), expanded_vs)
                    gnn_total_loss = gnn_pi_loss + gnn_v_loss
                    
                    gnn_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=1.0)
                    self.gnn_optimizer.step()
                    
                    epoch_loss += gnn_total_loss.item()
                    num_batches += 1
                
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"GNN Epoch {epoch+1}/{self.args.epochs} - Loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'gnn': self.gnn.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        
        if 'gnn' in checkpoint:
            self.gnn.load_state_dict(checkpoint['gnn'])
        else:
            print(f"GNN state not found in {filepath}, initializing new GNN")