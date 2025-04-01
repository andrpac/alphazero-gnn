import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GNNLayer(nn.Module):
    """
    A simple Graph Neural Network layer for message passing between states.
    """
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, adj):
        """
        Forward pass for GNN layer
        
        Args:
            x: Node features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, output_dim]
        """
        # Apply linear transformation
        support = self.W(x)
        
        # Message passing: matrix multiplication between adjacency and features
        output = torch.bmm(adj, support)
        
        return F.relu(output)


class FrozenLakeNet():
    """Neural network for the FrozenLake game with integrated GNN."""
    
    def __init__(self, game, args):
        """Initialize the neural network"""
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # Define the neural network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet = EnhancedNNet(self.board_size, self.action_size, args)
        self.nnet.to(self.device)
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        
        # For generating neighbor states
        self.game = game
        
    def create_adjacency(self, num_nodes):
        """
        Create a simple fully-connected adjacency matrix for the local neighborhood
        
        Args:
            num_nodes: Number of nodes in the graph
            
        Returns:
            Normalized adjacency matrix
        """
        # Create fully connected adjacency (all states connected)
        adjacency = torch.ones((num_nodes, num_nodes), device=self.device)
        
        # Normalize adjacency matrix (symmetric normalization)
        rowsum = adjacency.sum(1)
        d_inv_sqrt = torch.pow(rowsum.clamp(min=1e-8), -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adjacency = torch.mm(torch.mm(d_mat_inv_sqrt, adjacency), d_mat_inv_sqrt)
        
        return adjacency
        
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
                batch_boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
                
                # Process each board in the batch
                batch_pi_outputs = []
                batch_v_outputs = []
                
                for board_idx in range(batch_boards.shape[0]):
                    single_board = batch_boards[board_idx].unsqueeze(0)
                    
                    # Generate neighbor states from valid moves
                    board_np = single_board.squeeze(0).cpu().numpy()
                    
                    # Generate a set of neighboring states using valid moves
                    neighbor_states = [board_np]  # Include current state
                    valids = self.game.getValidMoves(board_np, 1)
                    
                    for action in range(self.action_size):
                        if valids[action]:
                            next_board, _ = self.game.getNextState(board_np, 1, action)
                            canonical_next = self.game.getCanonicalForm(next_board, 1)
                            neighbor_states.append(canonical_next)
                    
                    # Convert neighbor states to tensor
                    neighbor_tensors = torch.FloatTensor(np.array(neighbor_states)).to(self.device)
                    
                    # Create adjacency matrix
                    adjacency = self.create_adjacency(len(neighbor_states))
                    
                    # Forward pass with GNN
                    pi, v = self.nnet(single_board, neighbor_tensors, adjacency.unsqueeze(0))
                    
                    batch_pi_outputs.append(pi)
                    batch_v_outputs.append(v)
                
                # Stack outputs
                out_pi = torch.cat(batch_pi_outputs, dim=0)
                out_v = torch.cat(batch_v_outputs, dim=0)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
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
    
    def predict(self, board, neighbor_states=None):
        """
        Predict policy and value for a given board.
        Can optionally receive precomputed neighboring states from MCTS.
        
        Args:
            board: Current board state
            neighbor_states: List of neighboring states (optional)
            
        Returns:
            policy, value: Policy vector and value prediction
        """
        self.nnet.eval()
        
        # Prepare input
        board_tensor = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        if len(board_tensor.shape) == 2:
            board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension
        
        # If neighbor states not provided, generate them
        if neighbor_states is None:
            # Generate a set of neighboring states using valid moves
            neighbor_states = [board]  # Include current state
            valids = self.game.getValidMoves(board, 1)
            
            for action in range(self.action_size):
                if valids[action]:
                    next_board, _ = self.game.getNextState(board, 1, action)
                    canonical_next = self.game.getCanonicalForm(next_board, 1)
                    neighbor_states.append(canonical_next)
        
        # Convert neighbor states to tensor
        neighbor_tensors = torch.FloatTensor(np.array(neighbor_states)).to(self.device)
        
        # Create adjacency matrix
        adjacency = self.create_adjacency(len(neighbor_states))
            
        # Get predictions
        with torch.no_grad():
            try:
                pi, v = self.nnet(board_tensor, neighbor_tensors, adjacency.unsqueeze(0))
                
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


class EnhancedNNet(nn.Module):
    """Neural network architecture with GNN component"""
    
    def __init__(self, board_size, action_size, args):
        """Initialize the neural network architecture"""
        super(EnhancedNNet, self).__init__()
        
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.input_size = self.board_x * self.board_y
        self.embedding_dim = getattr(args, 'embedding_dim', 64)  # Default to 64 if not specified
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU()
        )
        
        # GNN layers
        num_gnn_layers = getattr(args, 'gnn_layers', 2)  # Default to 2 if not specified
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.embedding_dim, self.embedding_dim) 
            for _ in range(num_gnn_layers)
        ])
        
        # Policy head
        self.policy_head = nn.Linear(self.embedding_dim, action_size)
        
        # Value head
        self.value_head = nn.Linear(self.embedding_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small random values to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, neighbor_states, adjacency):
        """
        Forward pass through the network
        
        Args:
            x: Current state [batch_size, H, W]
            neighbor_states: All neighbor states including current [num_neighbors, H, W]
            adjacency: Adjacency matrix [batch_size, num_neighbors, num_neighbors]
            
        Returns:
            Policy and value for the current state
        """
        batch_size = x.size(0)
        
        # Process all states (current + neighbors) to get embeddings
        # Reshape to 2D
        states_flattened = neighbor_states.view(neighbor_states.size(0), -1)
        
        # Extract features
        node_embeddings = self.feature_extractor(states_flattened)
        
        # Add batch dimension for GNN processing
        node_embeddings = node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_embeddings = gnn_layer(node_embeddings, adjacency)
        
        # Extract current state embedding (always the first node)
        current_embedding = node_embeddings[:, 0, :]
        
        # Policy head
        logits = self.policy_head(current_embedding)
        pi = F.softmax(logits, dim=1)
        
        # Value head
        v = torch.tanh(self.value_head(current_embedding))
        
        return pi, v