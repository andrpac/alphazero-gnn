import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TicTacToeNet(nn.Module):
    """
    Simple neural network for TicTacToe that outputs a policy and a value.
    """
    def __init__(self, game, args):
        super(TicTacToeNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # Common layers
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        
        # Policy head
        self.fc1 = nn.Linear(128 * (self.board_x-2) * (self.board_y-2), 512)
        self.fc_policy = nn.Linear(512, self.action_size)
        
        # Value head
        self.fc2 = nn.Linear(128 * (self.board_x-2) * (self.board_y-2), 512)
        self.fc_value = nn.Linear(512, 1)
        
    def forward(self, s):
        """
        Forward pass for the neural network.
        
        Args:
            s: Input board state
            
        Returns:
            pi: Policy output (probabilities for each action)
            v: Value output (evaluation of the state)
        """
        # Reshape input
        s = s.view(-1, 1, self.board_x, self.board_y)
        
        # Common layers
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        
        # Flatten
        s = s.view(-1, 128 * (self.board_x-2) * (self.board_y-2))
        
        # Policy head
        pi = F.relu(self.fc1(s))
        pi = self.fc_policy(pi)
        
        # Value head
        v = F.relu(self.fc2(s))
        v = torch.tanh(self.fc_value(v))
        
        return F.log_softmax(pi, dim=1), v

class TicTacToeNNetWrapper():
    """
    Wrapper class for the neural network to interface with the MCTS algorithm.
    """
    def __init__(self, game, args):
        self.nnet = TicTacToeNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # Initialize with CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)
        
    def train(self, examples):
        """
        Train the neural network on examples.
        
        Args:
            examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.epochs):
            self.nnet.train()
            
            # Sample batch
            batch_idx = np.random.randint(0, len(examples), min(len(examples), self.args.batch_size))
            boards, pis, vs = list(zip(*[examples[i] for i in batch_idx]))
            
            # Prepare input
            boards = torch.FloatTensor(np.array(boards)).to(self.device)
            target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
            
            # Compute output
            out_pi, out_v = self.nnet(boards)
            
            # Calculate loss
            l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
            l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
            total_loss = l_pi + l_v
            
            # Compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
    def predict(self, board, neighbor_states=None):
        """
        Predict the policy and value for the given board.
        
        Args:
            board: np array with board
            neighbor_states: optional list of neighboring states for GNN enhancement
            
        Returns:
            pi: policy vector
            v: float value
        """
        # Preparing input
        board = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        
        # Set the network to evaluation mode
        self.nnet.eval()
        
        with torch.no_grad():
            # Forward pass
            pi, v = self.nnet(board)
            
            # Return policy and value
            return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0][0]
    
    def save_checkpoint(self, folder, filename):
        """Save model checkpoint to file"""
        filepath = folder + '/' + filename
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        """Load model checkpoint from file"""
        filepath = folder + '/' + filename
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])