import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FrozenLakeNet():
    """Neural network for the FrozenLake game using PyTorch."""
    
    def __init__(self, game, args):
        """Initialize the neural network"""
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # Define the neural network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet = SimpleFrozenLakeNNet(self.board_size, self.action_size, args.num_channels)
        self.nnet.to(self.device)
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        
    def train(self, examples):
        """Train the neural network with examples from self-play"""
        self.nnet.train()
        
        # Skip empty examples
        if not examples or len(examples) < 4:
            print("Not enough examples for training, need at least 4")
            return
            
        print(f"Training on {len(examples)} examples")
        
        # Reset optimizer for each training session
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        
        # Filter out examples with None values
        examples = [(x[0], x[1], x[2]) for x in examples if x[2] is not None]
        
        # Training loop
        for epoch in range(self.args.epochs):
            # Shuffle examples
            np.random.shuffle(examples)
            
            # Process batches
            batch_size = min(len(examples), self.args.batch_size)
            batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
            
            epoch_loss = 0
            num_batches = 0
            
            for batch in batches:
                boards, pis, vs = list(zip(*batch))
                
                # Check for NaN values
                if any(np.isnan(np.sum(x)) for x in boards) or any(np.isnan(np.sum(x)) for x in pis) or any(np.isnan(x) for x in vs):
                    print("NaN values detected in input data, skipping batch")
                    continue
                    
                # Convert to PyTorch tensors
                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Get network outputs
                out_pi, out_v = self.nnet(boards)
                
                # Calculate loss with clipping to avoid NaN
                pi_loss = -torch.mean(torch.sum(target_pis * torch.log(out_pi.clamp(min=1e-8)), dim=1))
                v_loss = F.mse_loss(out_v.view(-1), target_vs)
                total_loss = pi_loss + v_loss
                
                # Backpropagate
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            # Print progress
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{self.args.epochs} - Loss: {avg_loss:.4f}")
    
    def predict(self, board):
        """Predict policy and value for a given board"""
        self.nnet.eval()
        
        # Prepare input
        board = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        if len(board.shape) == 2:
            board = board.unsqueeze(0)  # Add batch dimension
            
        # Get predictions
        with torch.no_grad():
            try:
                pi, v = self.nnet(board)
                
                # Handle NaN values with fallback strategy
                if torch.isnan(pi).any():
                    pi = torch.ones_like(pi) / pi.size(1)
                if torch.isnan(v).any():
                    v = torch.zeros_like(v)
                    
                return pi.detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]
            except Exception as e:
                print(f"Error in neural network prediction: {e}")
                # Fallback to uniform policy on error
                return np.ones(self.action_size) / self.action_size, 0.0
    
    def save_checkpoint(self, folder, filename):
        """Save the model parameters"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        """Load the model parameters"""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])


class SimpleFrozenLakeNNet(nn.Module):
    """Simplified Neural network architecture for FrozenLake"""
    
    def __init__(self, board_size, action_size, num_channels):
        """Initialize the neural network architecture"""
        super(SimpleFrozenLakeNNet, self).__init__()
        
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.input_size = self.board_x * self.board_y
        
        # Simpler architecture with fewer layers
        # First layer
        self.fc1 = nn.Linear(self.input_size, 64)
        
        # Second layer
        self.fc2 = nn.Linear(64, 64)
        
        # Policy head
        self.fc_pi = nn.Linear(64, action_size)
        
        # Value head
        self.fc_v = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small random values to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Flatten input
        x = x.view(-1, self.input_size)
        
        # Apply layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy head
        pi = F.softmax(self.fc_pi(x), dim=1)
        
        # Value head
        v = torch.tanh(self.fc_v(x))
        
        return pi, v