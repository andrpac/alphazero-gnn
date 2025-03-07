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
        self.nnet = FrozenLakeNNet(self.board_size, self.action_size, args.num_channels)
        self.nnet.to(self.device)
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=1e-4)
        
        # Add learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
    def train(self, examples):
        """Train the neural network with examples from self-play"""
        self.nnet.train()
        
        # Skip empty examples
        if not examples:
            return
            
        # Filter out examples with None values
        examples = [(x[0], x[1], x[2]) for x in examples if x[2] is not None]
        
        if not examples:
            return
        
        # Ensure we have enough examples for batch normalization
        if len(examples) < 8:
            print("Not enough examples for training, need at least 8")
            return
            
        print(f"Training on {len(examples)} examples")
        
        for epoch in range(self.args.epochs):
            # Create multiple batches for each epoch
            all_indices = np.arange(len(examples))
            np.random.shuffle(all_indices)
            
            # Ensure batch size is at least 8 for batch normalization
            batch_size = max(8, self.args.batch_size)
            batch_count = max(1, len(examples) // batch_size)
            
            epoch_loss = 0
            epoch_pi_loss = 0
            epoch_v_loss = 0
            num_batches = 0
            
            for b in range(batch_count):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, len(examples))
                batch_indices = all_indices[start_idx:end_idx]
                
                if len(batch_indices) < 8:  # Need at least 8 samples for batch norm
                    continue
                    
                samples = [examples[i] for i in batch_indices]
                boards, pis, vs = list(zip(*samples))
                
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
                
                # Check for NaN in outputs
                if torch.isnan(out_pi).any() or torch.isnan(out_v).any():
                    print("NaN values detected in model output, skipping batch")
                    continue
                    
                # Safe log calculation
                safe_log = torch.log(out_pi.clamp(min=1e-8))
                
                # Calculate loss
                pi_loss = -torch.mean(torch.sum(target_pis * safe_log, dim=1))
                v_loss = torch.mean((target_vs - out_v.view(-1)) ** 2)
                
                # Balanced loss - equal weight to policy and value
                total_batch_loss = pi_loss + v_loss
                
                # Backpropagate
                if not torch.isnan(total_batch_loss) and not torch.isinf(total_batch_loss):
                    total_batch_loss.backward()
                    
                    # Add gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    epoch_loss += total_batch_loss.item()
                    epoch_pi_loss += pi_loss.item()
                    epoch_v_loss += v_loss.item()
                    num_batches += 1
            
            # Print progress
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_pi_loss = epoch_pi_loss / num_batches
                avg_v_loss = epoch_v_loss / num_batches
                print(f"Epoch {epoch+1}/{self.args.epochs} - Loss: {avg_loss:.4f} (Policy: {avg_pi_loss:.4f}, Value: {avg_v_loss:.4f})")
                
                # Update learning rate
                self.scheduler.step(avg_loss)
    
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
                
                # Handle NaN values
                if torch.isnan(pi).any():
                    pi = torch.ones_like(pi) / pi.size(1)
                    
                if torch.isnan(v).any():
                    v = torch.zeros_like(v)
                
                return pi.detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]
            except Exception as e:
                # Fallback to uniform policy on error
                print(f"Error in neural network prediction: {e}")
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


class FrozenLakeNNet(nn.Module):
    """Neural network architecture for FrozenLake"""
    
    def __init__(self, board_size, action_size, num_channels):
        """Initialize the neural network architecture"""
        super(FrozenLakeNNet, self).__init__()
        
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.num_channels = num_channels
        
        # Enhanced architecture for FrozenLake
        self.input_size = self.board_x * self.board_y
        
        # Position encoding to help with spatial awareness
        self.pos_encoding = nn.Parameter(torch.randn(self.input_size) * 0.01)
        
        # First fully connected layer with residual connection
        self.fc1 = nn.Linear(self.input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Third fully connected layer
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Fourth fully connected layer
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        # Policy head with preprocessing
        self.fc_pi_pre = nn.Linear(64, 32)
        self.bn_pi = nn.BatchNorm1d(32)
        self.fc_pi = nn.Linear(32, action_size)
        
        # Value head with preprocessing
        self.fc_v_pre = nn.Linear(64, 32)
        self.bn_v = nn.BatchNorm1d(32)
        self.fc_v = nn.Linear(32, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small random values to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Flatten input
        if len(x.shape) == 3:  # [batch_size, board_x, board_y]
            x = x.view(-1, self.board_x * self.board_y)
        elif len(x.shape) == 4:  # [batch_size, channels, board_x, board_y]
            x = x.view(-1, x.shape[1] * self.board_x * self.board_y)
        
        # Add positional encoding to help with spatial awareness
        x = x + self.pos_encoding
            
        # Apply layers with ReLU + batch normalization + dropout for robustness
        x1 = self.fc1(x)
        if x1.size(0) > 1:  # BatchNorm requires more than 1 sample
            x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = self.fc2(x1)
        if x2.size(0) > 1:
            x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        x3 = self.fc3(x2)
        if x3.size(0) > 1:
            x3 = self.bn3(x3)
        x3 = F.relu(x3)
        # Add skip connection
        x3 = x3 + F.pad(x2, (0, x3.size(1) - x2.size(1))) if x3.size(1) >= x2.size(1) else x3
        x3 = F.dropout(x3, p=0.2, training=self.training)
        
        x4 = self.fc4(x3)
        if x4.size(0) > 1:
            x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x4 = F.dropout(x4, p=0.2, training=self.training)
        
        # Policy head with separate pathway
        pi = self.fc_pi_pre(x4)
        if pi.size(0) > 1:
            pi = self.bn_pi(pi)
        pi = F.relu(pi)
        pi = self.fc_pi(pi)
        pi = F.softmax(pi, dim=1)
        
        # Value head with separate pathway
        v = self.fc_v_pre(x4)
        if v.size(0) > 1:
            v = self.bn_v(v)
        v = F.relu(v)
        v = self.fc_v(v)
        v = torch.tanh(v)
        
        return pi, v