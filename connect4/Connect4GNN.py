import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

from .Connect4Net import Connect4NNetWrapper
from gnn_utils import PolicyValueGNN

class Connect4GNNWrapper(Connect4NNetWrapper):
    def __init__(self, game, args):
        super().__init__(game, args)
        self.game = game
        self.args = args

        dummy = torch.zeros(1, 1, *game.getBoardSize(), device=self.device)
        with torch.no_grad():
            feat = self.nnet.extract_features(dummy)
        channels, h, w = feat.shape[1:]
        self.feature_dim = channels * h * w

        self.small_dim = getattr(args, 'gnn_feature_dim', max(1, self.feature_dim // 4))
        self.input_fc  = nn.Linear(self.feature_dim, self.small_dim).to(self.device)
        self.gnn       = PolicyValueGNN(self.small_dim, args.gnn_layers).to(self.device)
        self.output_fc = nn.Linear(self.small_dim, self.feature_dim).to(self.device)

        self.base_optimizer = optim.AdamW(self.nnet.parameters(), lr=args.lr, weight_decay=1e-4)
        gnn_params = (
            list(self.input_fc.parameters()) +
            list(self.gnn.parameters()) +
            list(self.output_fc.parameters())
        )
        self.gnn_optimizer = optim.AdamW(gnn_params, lr=0.1 * args.lr, weight_decay=1e-4)

    def save_checkpoint(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save({
            'base_state':      self.nnet.state_dict(),
            'input_fc_state':  self.input_fc.state_dict(),
            'gnn_state':       self.gnn.state_dict(),
            'output_fc_state': self.output_fc.state_dict(),
        }, path)

    def load_checkpoint(self, folder, filename):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            cp = torch.load(path, map_location=self.device)
            self.nnet.load_state_dict(cp.get('base_state', {}), strict=False)
            self.input_fc.load_state_dict(cp.get('input_fc_state', {}), strict=False)
            self.gnn.load_state_dict(cp.get('gnn_state', {}), strict=False)
            self.output_fc.load_state_dict(cp.get('output_fc_state', {}), strict=False)
            self.nnet.eval()

    def train(self, examples, gnn_examples=None):
        if examples:
            self._train_base_network(examples)
    
        if gnn_examples:
            self._train_gnn_only(gnn_examples)

    def _train_base_network(self, examples):
        for p in self.input_fc.parameters():
            p.requires_grad = False
        for p in self.gnn.parameters():
            p.requires_grad = False
        for p in self.output_fc.parameters():
            p.requires_grad = False
        for p in self.nnet.parameters():
            p.requires_grad = True

        N = len(examples)
        if N == 0:
            return

        device = self.device
        batch_size = self.args.batch_size
        indices = np.arange(N)
        
        use_amp = device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(self.args.epochs):
            self.nnet.train()
            np.random.shuffle(indices)
            total_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                boards, pis, vs = zip(*[examples[i] for i in batch_idx])
                
                boards_np = np.array(boards, dtype=np.float32)
                boards_t = torch.from_numpy(boards_np).unsqueeze(1).to(device)
                target_pis = torch.from_numpy(np.array(pis, dtype=np.float32)).to(device)
                target_vs = torch.from_numpy(np.array(vs, dtype=np.float32)).to(device)

                self.base_optimizer.zero_grad(set_to_none=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        log_pi, v_pred = self.nnet(boards_t)
                        v_pred = v_pred.squeeze(-1)
                        loss = F.kl_div(log_pi, target_pis, reduction='batchmean') + F.l1_loss(v_pred, target_vs)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.base_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
                    scaler.step(self.base_optimizer)
                    scaler.update()
                else:
                    log_pi, v_pred = self.nnet(boards_t)
                    v_pred = v_pred.squeeze(-1)
                    loss = F.kl_div(log_pi, target_pis, reduction='batchmean') + F.l1_loss(v_pred, target_vs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
                    self.base_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            print(f"Base Epoch {epoch+1}/{self.args.epochs}, Loss {avg_loss:.4f}")

        for p in self.input_fc.parameters():
            p.requires_grad = True
        for p in self.gnn.parameters():
            p.requires_grad = True
        for p in self.output_fc.parameters():
            p.requires_grad = True

    def _train_gnn_only(self, gnn_examples):
        gnn_examples = [ex for ex in gnn_examples if ex[6].get('initial_nodes')]
        if not gnn_examples:
            return

        self.nnet.eval()
        for p in self.nnet.parameters():
            p.requires_grad = False
        for p in self.input_fc.parameters():
            p.requires_grad = True
        for p in self.gnn.parameters():
            p.requires_grad = True
        for p in self.output_fc.parameters():
            p.requires_grad = True

        device = self.device
        H, W = self.game.getBoardSize()
        G = min(64, len(gnn_examples))
        
        np.random.shuffle(gnn_examples)

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(gnn_examples), G):
                chunk = gnn_examples[i:i+G]
                feats_list = []
                edges_list = []
                roots = []
                pi_list = []
                v_list = []
                offset = 0

                for ex in chunk:
                    nodes_np = np.stack(ex[6]['initial_nodes'], axis=0).astype(np.float32)
                    Ni = nodes_np.shape[0]

                    nodes_t = torch.from_numpy(nodes_np).unsqueeze(1).to(device)
                    with torch.no_grad():
                        fmap = self.nnet.extract_features(nodes_t)
                    feats = fmap.view(Ni, -1)
                    feats_list.append(feats)

                    edge_index = ex[6]['initial_edges']
                    edge_index = edge_index.to(device)
                    edges_list.append(edge_index + offset)
                    roots.append(offset)

                    pi_list.append(torch.as_tensor(ex[4], dtype=torch.float32, device=device))
                    v_list.append(torch.tensor(ex[5], dtype=torch.float32, device=device))

                    offset += Ni

                all_feats  = torch.cat(feats_list, dim=0)
                all_edges  = torch.cat(edges_list, dim=1) if edges_list else torch.empty((2, 0), dtype=torch.long, device=device)
                all_roots  = torch.tensor(roots, dtype=torch.long, device=device)
                target_pis = torch.stack(pi_list, dim=0)
                target_vs  = torch.stack(v_list, dim=0)

                self.input_fc.train()
                self.gnn.train()
                self.output_fc.train()

                small_feats    = self.input_fc(all_feats)
                enhanced_small = self.gnn(small_feats, all_edges)
                enhanced_full  = self.output_fc(enhanced_small)
                ch = enhanced_full.size(1) // (H * W)
                enhanced       = enhanced_full.view(-1, ch, H, W)

                batch_roots = enhanced[all_roots]
                log_pi, v_pred = self.nnet.policy_value_heads(batch_roots)
                v_pred = v_pred.squeeze(-1)

                loss = F.kl_div(log_pi, target_pis, reduction='batchmean') + F.l1_loss(v_pred, target_vs)

                self.gnn_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.input_fc.parameters()) +
                    list(self.gnn.parameters()) +
                    list(self.output_fc.parameters()),
                    1.0
                )
                self.gnn_optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            print(f"GNN Epoch {epoch+1}/{self.args.epochs}, Avg Loss {avg_loss:.4f}")

        for p in self.nnet.parameters():
            p.requires_grad = True
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def predict(self, board: np.ndarray):
        if not getattr(self.args, 'gnn_predict', False):
            return super().predict(board)

        scratch = self.board_to_tensor(board)
        self.nnet.eval()
        self.gnn.eval()

        with torch.no_grad():
            fmap = self.nnet.extract_features(scratch)
            
            H, W = self.game.getBoardSize()
            edges = []
            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    if j < W-1: edges.append([idx, i*W + (j+1)])
                    if i < H-1: edges.append([idx, (i+1)*W + j])
                    if j < W-1 and i < H-1: edges.append([idx, (i+1)*W + (j+1)])
                    if j > 0 and i < H-1: edges.append([idx, (i+1)*W + (j-1)])
            
            edges_bidir = edges + [[b, a] for a, b in edges]
            edge_index = torch.tensor(edges_bidir, dtype=torch.long).t().to(self.device)
            
            ch, h, w = fmap.shape[1:]
            fmap_per_pos = fmap.permute(0, 2, 3, 1).reshape(H*W, ch)
            
            small = self.input_fc(fmap_per_pos)
            enh_s = self.gnn(small, edge_index)
            enh_f = self.output_fc(enh_s)
            
            enhanced = enh_f.reshape(1, H, W, ch).permute(0, 3, 1, 2)
            log_pi, v = self.nnet.policy_value_heads(enhanced)

            pi = torch.exp(log_pi).cpu().numpy()[0]
            v_val = v.cpu().item()

            valids = self.game.getValidMoves(board, 1)
            pi *= valids
            if pi.sum() > 0:
                pi /= pi.sum()

            return pi, v_val
