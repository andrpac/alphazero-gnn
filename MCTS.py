import math
import numpy as np
import torch
from collections import defaultdict

EPS = 1e-8

class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Qsa = defaultdict(float)
        self.Nsa = defaultdict(int)
        self.Ns  = defaultdict(int)
        self.Ps  = {}
        self.Es  = {}
        self.Vs  = {}

        self.nodes = []
        self.edges = []
        self.edge_set = set()
        self.node_indices = {}

        H, W = game.getBoardSize()
        self._board_buf = torch.zeros((1, 1, H, W), dtype=torch.float32, device=nnet.device)
        self._action_size = game.getActionSize()
        self._cpuct = args.cpuct

        self._valid_moves_cache = {}
        self._canonical_cache = {}

    def _evaluate_leaf(self, board):
        tmp = torch.from_numpy(board.astype(np.float32))
        self._board_buf[0, 0].copy_(tmp, non_blocking=True)
        return self.nnet.predict_tensor(self._board_buf)

    def getActionProb(self, board, temp=1):
        for _ in range(self.args.numMCTSSims):
            self.search(board)

        key = board.tobytes()
        counts = np.zeros(self._action_size, dtype=np.float32)
        for a in range(self._action_size):
            counts[a] = self.Nsa[(key, a)]

        total = counts.sum()
        if total == 0:
            vm = self.Vs.get(key)
            if vm is None:
                vm = self.game.getValidMoves(board, 1)
                self.Vs[key] = vm
            return (vm / (vm.sum() + EPS)).tolist()

        if temp == 0:
            best = counts.max()
            acts = np.where(counts == best)[0]
            choice = np.random.choice(acts)
            probs = np.zeros(self._action_size, dtype=np.float32)
            probs[choice] = 1.0
            return probs.tolist()

        counts = np.power(counts, 1.0 / temp)
        ssum = counts.sum()
        if ssum == 0:
            vm = self.Vs.get(key)
            if vm is None:
                vm = self.game.getValidMoves(board, 1)
                self.Vs[key] = vm
            return (vm / (vm.sum() + EPS)).tolist()

        return (counts / ssum).tolist()

    def search(self, board):
        key = board.tobytes()

        if key not in self.Es:
            self.Es[key] = self.game.getGameEnded(board, 1)
        if self.Es[key] != 0:
            return -self.Es[key]

        if key not in self.Ps:
            pi, v = self._evaluate_leaf(board)

            vm_key = key
            vm = self._valid_moves_cache.get(vm_key)
            if vm is None:
                vm = self.game.getValidMoves(board, 1)
                self._valid_moves_cache[vm_key] = vm

            masked = pi * vm
            sm = masked.sum()
            if sm > 0:
                masked /= sm
            else:
                masked = vm / (vm.sum() + EPS)

            self.Ps[key] = masked
            self.Vs[key] = vm
            self.Ns[key] = 0

            if key not in self.node_indices:
                idx = len(self.nodes)
                self.node_indices[key] = idx
                self.nodes.append(board.copy())

            return -v

        vm = self.Vs[key]
        ns = self.Ns[key]
        sqrt_ns = math.sqrt(ns + EPS)
        ps = self.Ps[key]
        cpuct_factor = self._cpuct * sqrt_ns

        best_u = -float('inf')
        best_a = -1
        for a in range(self._action_size):
            if vm[a]:
                sa = (key, a)
                q = self.Qsa[sa]
                n = self.Nsa[sa]
                u = q + cpuct_factor * ps[a] / (1 + n)
                if u > best_u:
                    best_u = u
                    best_a = a

        next_b, next_player = self.game.getNextState(board, 1, best_a)

        canon_key = (next_b.tobytes(), next_player)
        canonical = self._canonical_cache.get(canon_key)
        if canonical is None:
            canonical = self.game.getCanonicalForm(next_b, next_player)
            self._canonical_cache[canon_key] = canonical

        v = self.search(canonical)

        if key not in self.node_indices:
            idx_s = len(self.nodes)
            self.node_indices[key] = idx_s
            self.nodes.append(board.copy())
        else:
            idx_s = self.node_indices[key]

        next_key = canonical.tobytes()
        if next_key not in self.node_indices:
            idx_ns = len(self.nodes)
            self.node_indices[next_key] = idx_ns
            self.nodes.append(canonical.copy())
        else:
            idx_ns = self.node_indices[next_key]

        e = (idx_s, idx_ns)
        if e not in self.edge_set:
            self.edges.append(e)
            self.edge_set.add(e)

        sa = (key, best_a)
        n = self.Nsa[sa]
        self.Qsa[sa] = (self.Qsa[sa] * n + v) / (n + 1)
        self.Nsa[sa] = n + 1

        self.Ns[key] = ns + 1

        return -v

    def expand_tree(self, board, expand_by=1):
        self.nodes = []
        self.edges = []
        self.edge_set = set()
        self.node_indices = {}

        pi0, v0 = self._evaluate_leaf(board)
        vm = self.game.getValidMoves(board, 1)
        init_pi = pi0 * vm
        sm = init_pi.sum()
        if sm > 0:
            init_pi /= sm
        else:
            init_pi = vm / (vm.sum() + EPS)
        init_v = v0

        for _ in range(expand_by):
            self.search(board)

        key = board.tobytes()
        counts = np.zeros(self._action_size, dtype=np.float32)
        q_sum = 0.0
        total_count = 0
        for a in range(self._action_size):
            count = self.Nsa[(key, a)]
            counts[a] = count
            if count > 0:
                q_sum += self.Qsa[(key, a)] * count
                total_count += count

        exp_pi = counts / (counts.sum() + EPS)
        exp_v = q_sum / (total_count + EPS) if total_count > 0 else v0

        if self.edges:
            edge_tensor = torch.tensor(self.edges, dtype=torch.long, device=self.nnet.device).t().cpu()
        else:
            edge_tensor = torch.empty((2, 0), dtype=torch.long)

        return (init_pi, init_v, exp_pi, exp_v), {
            'initial_nodes': list(self.nodes),
            'initial_edges': edge_tensor
        }
