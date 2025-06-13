import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += r
        return F.relu(out, inplace=True)

class Connect4Net(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        bx, by = game.getBoardSize()
        self.action_size = game.getActionSize()
        nc = 128
        nres = getattr(args, 'num_residual_layers', 20)

        self.conv_input = nn.Conv2d(1, nc, kernel_size=5, padding=2, bias=False)
        self.bn_input   = nn.BatchNorm2d(nc)
        self.residual_tower = nn.Sequential(*[ResidualBlock(nc) for _ in range(nres)])

        self.conv_policy = nn.Conv2d(nc, 64, kernel_size=1, bias=False)
        self.bn_policy   = nn.BatchNorm2d(64)
        self.fc_policy   = nn.Linear(64 * bx * by, self.action_size)

        self.conv_value = nn.Conv2d(nc, 64, kernel_size=1, bias=False)
        self.bn_value   = nn.BatchNorm2d(64)
        self.fc_value1  = nn.Linear(64 * bx * by, 512)
        self.fc_value2  = nn.Linear(512, 1)

        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.3))

    def extract_features(self, s):
        out = F.relu(self.bn_input(self.conv_input(s)), inplace=True)
        return self.residual_tower(out)

    def policy_value_heads(self, f):
        bx, by = f.shape[2], f.shape[3]
        p = F.relu(self.bn_policy(self.conv_policy(f)), inplace=True)
        p = p.view(-1, 64 * bx * by)
        p = self.dropout(p)
        log_pi = F.log_softmax(self.fc_policy(p), dim=1)

        v = F.relu(self.bn_value(self.conv_value(f)), inplace=True)
        v = v.view(-1, 64 * bx * by)
        v = self.dropout(v)
        v = F.relu(self.fc_value1(v), inplace=True)
        v = torch.tanh(0.5 * self.fc_value2(v))
        return log_pi, v

    def forward(self, s):
        f = self.extract_features(s)
        return self.policy_value_heads(f)

class Connect4NNetWrapper:
    def __init__(self, game, args):
        self.nnet = Connect4Net(game, args)
        bx, by = game.getBoardSize()
        self.board_x = bx
        self.board_y = by
        self.args = args

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.nnet.to(self.device)
        self._scratch_board = torch.zeros((1, 1, bx, by), dtype=torch.float32, device=self.device)
        self.nnet.eval()

    def save_checkpoint(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save({'state_dict': self.nnet.state_dict()}, path)

    def load_checkpoint(self, folder, filename):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            cp = torch.load(path, map_location=self.device)
            self.nnet.load_state_dict(cp['state_dict'], strict=False)
            self.nnet.eval()

    def board_to_tensor(self, board: np.ndarray):
        temp = torch.from_numpy(board.astype(np.float32))
        self._scratch_board[0, 0].copy_(temp, non_blocking=True)
        return self._scratch_board

    def predict_tensor(self, board_tensor: torch.Tensor):
        self.nnet.eval()
        with torch.no_grad():
            log_pi, v = self.nnet(board_tensor)
            pi = torch.exp(log_pi).cpu().numpy()[0]
            return pi, v.cpu().item()

    def predict(self, board: np.ndarray):
        b_t = torch.from_numpy(board.astype(np.float32)).to(self.device)
        b_t = b_t.view(1, 1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            log_pi, v = self.nnet(b_t)
            return torch.exp(log_pi).cpu().numpy()[0], v.cpu().item()

    def train(self, examples):
        optimizer = optim.AdamW(self.nnet.parameters(), lr=self.args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        
        use_amp = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        N = len(examples)
        if N == 0:
            return
            
        indices = np.arange(N)
        batch_size = self.args.batch_size
        
        for epoch in range(self.args.epochs):
            self.nnet.train()
            np.random.shuffle(indices)
            total_loss = 0.0
            num_batches = 0

            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                boards, pis, vs = zip(*[examples[i] for i in batch_idx])

                boards_np = np.array(boards, dtype=np.float32)
                boards_t = torch.from_numpy(boards_np).unsqueeze(1).to(self.device)
                target_pis = torch.from_numpy(np.array(pis, dtype=np.float32)).to(self.device)
                target_vs  = torch.from_numpy(np.array(vs, dtype=np.float32)).to(self.device)

                optimizer.zero_grad(set_to_none=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        log_pi, v_pred = self.nnet(boards_t)
                        v_pred = v_pred.squeeze(-1)
                        loss = F.kl_div(log_pi, target_pis, reduction='batchmean') + F.l1_loss(v_pred, target_vs)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_pi, v_pred = self.nnet(boards_t)
                    v_pred = v_pred.squeeze(-1)
                    loss = F.kl_div(log_pi, target_pis, reduction='batchmean') + F.l1_loss(v_pred, target_vs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            print(f"CNN Epoch {epoch+1}/{self.args.epochs}, Loss {avg_loss:.4f}")
        
        self.nnet.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()