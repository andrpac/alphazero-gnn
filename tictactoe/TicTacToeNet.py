import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class TicTacToeNet(nn.Module):
    def __init__(self, game, args):
        super(TicTacToeNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        
        self.fc1 = nn.Linear(128 * (self.board_x-2) * (self.board_y-2), 512)
        self.fc_policy = nn.Linear(512, self.action_size)
        
        self.fc2 = nn.Linear(128 * (self.board_x-2) * (self.board_y-2), 512)
        self.fc_value = nn.Linear(512, 1)
        
    def extract_features(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = s.view(-1, 128 * (self.board_x-2) * (self.board_y-2))
        return s
        
    def policy_value_heads(self, features):
        pi = F.relu(self.fc1(features))
        pi = self.fc_policy(pi)
        pi = F.log_softmax(pi, dim=1)
        
        v = F.relu(self.fc2(features))
        v = torch.tanh(self.fc_value(v))
        
        return pi, v
        
    def forward(self, s):
        features = self.extract_features(s)
        return self.policy_value_heads(features)

class TicTacToeNNetWrapper():
    def __init__(self, game, args):
        self.nnet = TicTacToeNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)
        
    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.epochs):
            self.nnet.train()
            
            batch_idx = np.random.randint(0, len(examples), min(len(examples), self.args.batch_size))
            boards, pis, vs = list(zip(*[examples[i] for i in batch_idx]))
            
            boards = torch.FloatTensor(np.array(boards)).to(self.device)
            target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
            
            out_pi, out_v = self.nnet(boards)
            
            l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
            l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
            total_loss = l_pi + l_v
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        
        self.nnet.eval()
        
        with torch.no_grad():
            pi, v = self.nnet(board)
            return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0][0]
    
    def save_checkpoint(self, folder, filename):
        filepath = folder + '/' + filename
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, folder, filename):
        filepath = folder + '/' + filename
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])