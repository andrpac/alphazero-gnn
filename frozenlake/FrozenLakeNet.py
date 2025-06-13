import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FrozenLakeNet():
    def __init__(self, game, args):
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet = FrozenLakeNNetBase(self.board_size, self.action_size, args)
        self.nnet.to(self.device)
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        
        self.game = game
        
    def create_adjacency(self, num_nodes):
        adjacency = torch.ones((num_nodes, num_nodes), device=self.device)
        
        rowsum = adjacency.sum(1)
        d_inv_sqrt = torch.pow(rowsum.clamp(min=1e-8), -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adjacency = torch.mm(torch.mm(d_mat_inv_sqrt, adjacency), d_mat_inv_sqrt)
        
        return adjacency
        
    def train(self, examples):
        self.nnet.train()
        
        if not examples or len(examples) < 4:
            print("Not enough examples for training, need at least 4")
            return
            
        print(f"Training on {len(examples)} examples")
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        
        examples = [(x[0], x[1], x[2]) for x in examples if x[2] is not None]
        
        for epoch in range(self.args.epochs):
            np.random.shuffle(examples)
            
            batch_size = min(len(examples), self.args.batch_size)
            batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
            
            epoch_loss = 0
            num_batches = 0
            
            for batch in batches:
                boards, pis, vs = list(zip(*batch))
                
                if any(np.isnan(np.sum(x)) for x in boards) or any(np.isnan(np.sum(x)) for x in pis) or any(np.isnan(x) for x in vs):
                    print("NaN values detected in input data, skipping batch")
                    continue
                    
                batch_boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
                
                out_pi, out_v = self.nnet(batch_boards)
                
                self.optimizer.zero_grad()
                
                pi_loss = -torch.mean(torch.sum(target_pis * torch.log(out_pi.clamp(min=1e-8)), dim=1))
                v_loss = F.mse_loss(out_v.view(-1), target_vs)
                total_loss = pi_loss + v_loss
                
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{self.args.epochs} - Loss: {avg_loss:.4f}")
    
    def predict(self, board):
        self.nnet.eval()
        
        board_tensor = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        if len(board_tensor.shape) == 2:
            board_tensor = board_tensor.unsqueeze(0)
            
        with torch.no_grad():
            try:
                pi, v = self.nnet(board_tensor)
                
                if torch.isnan(pi).any():
                    pi = torch.ones_like(pi) / pi.size(1)
                if torch.isnan(v).any():
                    v = torch.zeros_like(v)
                    
                return pi.detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]
            except Exception as e:
                print(f"Error in neural network prediction: {e}")
                return np.ones(self.action_size) / self.action_size, 0.0
    
    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])


class FrozenLakeNNetBase(nn.Module):
    def __init__(self, board_size, action_size, args):
        super(FrozenLakeNNetBase, self).__init__()
        
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.input_size = self.board_x * self.board_y
        self.embedding_dim = getattr(args, 'embedding_dim', 64)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(self.embedding_dim, action_size)
        self.value_head = nn.Linear(self.embedding_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        x = x.view(x.size(0), -1)
        return self.feature_extractor(x)
    
    def policy_value_heads(self, features):
        logits = self.policy_head(features)
        pi = F.softmax(logits, dim=1)
        v = torch.tanh(self.value_head(features))
        return pi, v
    
    def forward(self, x):
        features = self.extract_features(x)
        return self.policy_value_heads(features)