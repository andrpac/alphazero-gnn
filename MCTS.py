import logging
import math
import numpy as np
import torch

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}   # stores game.getGameEnded ended for board s
        self.Vs = {}   # stores game.getValidMoves for board s
        
        self.standard_predictions = {}  # Format: {s: (pi, v)}
        self.gnn_predictions = {}       # Format: {s: (pi, v)}
        
        self.expanded = False    # Flag to indicate if we're in expansion mode
        self.expanded_nodes = {} # Tracking expanded nodes for sliding window training

    def getActionProb(self, canonicalBoard, temp=1):
        self.standard_predictions = {}
        self.gnn_predictions = {}
        
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [(x + EPS) ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        
        if counts_sum <= 0:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            valid_sum = np.sum(valids)
            if valid_sum > 0:
                return valids / valid_sum
            else:
                return np.ones(len(counts)) / len(counts)
            
        probs = [x / counts_sum for x in counts]
        return probs

    def expand_tree(self, canonicalBoard, expand_by=5):
        """
        Expand the current MCTS tree with additional simulations
        and return states with both initial and expanded policies/values.
        
        Args:
            canonicalBoard: The board to expand from
            expand_by: Number of additional simulations to run
            
        Returns:
            Dictionary mapping state strings to tuples of 
            (initial_policy, initial_value, expanded_policy, expanded_value)
        """
        s = self.game.stringRepresentation(canonicalBoard)
        self.expanded = True
        self.expanded_nodes = {}
        
        # Store current state of the search tree to derive initial policy
        initial_counts = {}
        for key in self.Nsa:
            if key[0] == s:
                initial_counts[key[1]] = self.Nsa[key]
        
        # Get initial policy from neural network if we haven't searched yet
        if not initial_counts:
            # Run standard number of simulations to establish initial policy
            for i in range(self.args.numMCTSSims):
                self.search(canonicalBoard)
                
            # Now collect the initial counts after standard simulations
            for key in self.Nsa:
                if key[0] == s:
                    initial_counts[key[1]] = self.Nsa[key]
        
        # Get initial policy from visit counts
        initial_policy = np.zeros(self.game.getActionSize())
        for a, count in initial_counts.items():
            initial_policy[a] = count
        
        initial_sum = np.sum(initial_policy)
        if initial_sum > 0:
            initial_policy = initial_policy / initial_sum
        else:
            # Fallback to valid moves if no counts
            valids = self.game.getValidMoves(canonicalBoard, 1)
            initial_policy = valids / np.sum(valids)
        
        # Get neural network's value prediction
        if s not in self.standard_predictions:
            with torch.no_grad():
                std_pi, std_v = self.nnet.predict(canonicalBoard)
                self.standard_predictions[s] = (std_pi, std_v)
        
        initial_value = self.standard_predictions[s][1]
        
        # Run additional simulations to expand the tree
        for i in range(expand_by):
            self.search(canonicalBoard)
        
        # Get expanded policy from updated visit counts
        expanded_policy = np.zeros(self.game.getActionSize())
        for key in self.Nsa:
            if key[0] == s:
                expanded_policy[key[1]] = self.Nsa[key]
        
        expanded_sum = np.sum(expanded_policy)
        if expanded_sum > 0:
            expanded_policy = expanded_policy / expanded_sum
        else:
            expanded_policy = initial_policy  # Fallback to initial policy
        
        # Calculate expanded value as weighted average of leaf Q-values
        expanded_value = 0
        valid_count = 0
        
        for a in range(self.game.getActionSize()):
            if (s, a) in self.Qsa and (s, a) in self.Nsa and self.Nsa[(s, a)] > 0:
                expanded_value += self.Qsa[(s, a)] * self.Nsa[(s, a)]
                valid_count += self.Nsa[(s, a)]
        
        if valid_count > 0:
            expanded_value = expanded_value / valid_count
        else:
            expanded_value = initial_value  # Fallback to initial value
        
        # Store both initial and expanded policies and values
        self.expanded_nodes[s] = (initial_policy, initial_value, expanded_policy, expanded_value)
        
        self.expanded = False
        return self.expanded_nodes

    def search(self, canonicalBoard, expansion=False):
        s = self.game.stringRepresentation(canonicalBoard)
        
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return self.Es[s]

        if expansion and self.Ns.get(s, 0) >= self.args.numMCTSSims:
            return 0

        if s not in self.Ps:
            if s not in self.Vs:
                self.Vs[s] = self.game.getValidMoves(canonicalBoard, 1)
            valids = self.Vs[s]
            
            try:
                with torch.no_grad():
                    std_pi, std_v = self.nnet.predict(canonicalBoard)
                    self.standard_predictions[s] = (std_pi, std_v)
                    
                    if hasattr(self.args, 'use_gnn') and self.args.use_gnn:
                        gnn_pi, gnn_v = self.nnet.predict_with_gnn(canonicalBoard)
                        self.gnn_predictions[s] = (gnn_pi, gnn_v)
                        
                        self.Ps[s] = gnn_pi if self.args.use_gnn else std_pi
                    else:
                        self.Ps[s] = std_pi
                
                self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s  # renormalize
                else:
                    log.warning("All valid moves were masked, using uniform policy")
                    self.Ps[s] = valids / np.sum(valids)

                self.Ns[s] = 0
                
                if hasattr(self.args, 'use_gnn') and self.args.use_gnn and s in self.gnn_predictions:
                    return self.gnn_predictions[s][1]
                else:
                    return self.standard_predictions[s][1]
                    
            except Exception as e:
                log.error(f"Error in neural network prediction: {e}")
                valids = self.Vs[s]
                self.Ps[s] = valids / np.sum(valids)
                self.Ns[s] = 0
                return 0

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        
        if a == -1:
            return 0
            
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, expansion)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        
        if hasattr(self.game, 'is_two_player') and self.game.is_two_player:
            return -v
        else:
            return v