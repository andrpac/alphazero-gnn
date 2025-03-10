import logging
import math
import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}   # stores game.getGameEnded ended for board s
        self.Vs = {}   # stores game.getValidMoves for board s
        
        # Cycle detection only
        self._path = set()

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            # Reset path for each simulation
            self._path = set()
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # Add small epsilon to each count to ensure non-zero probabilities
        counts = [(x + EPS) ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        
        # Ensure probabilities sum to 1
        if counts_sum <= 0:
            # Fallback to uniform valid moves
            valids = self.game.getValidMoves(canonicalBoard, 1)
            valid_sum = np.sum(valids)
            if valid_sum > 0:
                return valids / valid_sum
            else:
                # If no valid moves, return uniform distribution
                return np.ones(len(counts)) / len(counts)
            
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the value of the current canonicalBoard for the current player
               Note: We don't negate for single-player games
        """    
        s = self.game.stringRepresentation(canonicalBoard)
        
        # Check for cycles - if we've seen this state in current path
        if s in self._path:
            return -1.0  # Strong penalty for cycles
            
        # Add to path for cycle detection
        self._path.add(s)
        
        try:
            # Check if game ended
            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            if self.Es[s] != 0:
                # Terminal node - don't negate for single-player games
                self._path.remove(s)  # Clean up path before return
                return self.Es[s]

            # Check if we've seen this state before
            if s not in self.Ps:
                # Leaf node
                try:
                    self.Ps[s], v = self.nnet.predict(canonicalBoard)
                    valids = self.game.getValidMoves(canonicalBoard, 1)
                    self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
                    sum_Ps_s = np.sum(self.Ps[s])
                    if sum_Ps_s > 0:
                        self.Ps[s] /= sum_Ps_s  # renormalize
                    else:
                        # if all valid moves were masked make all valid moves equally probable
                        log.warning("All valid moves were masked, using uniform policy")
                        self.Ps[s] = valids / np.sum(valids)

                    self.Vs[s] = valids
                    self.Ns[s] = 0
                    
                    # Remove from path before returning
                    self._path.remove(s)
                    return v  # Don't negate for single-player games
                except Exception as e:
                    log.error(f"Error in neural network prediction: {e}")
                    # Fallback to uniform valid moves
                    valids = self.game.getValidMoves(canonicalBoard, 1)
                    self.Ps[s] = valids / np.sum(valids)
                    self.Vs[s] = valids
                    self.Ns[s] = 0
                    
                    self._path.remove(s)  # Clean up path before return
                    return 0

            # Select best action
            valids = self.Vs[s]
            cur_best = -float('inf')
            best_act = -1

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    if (s, a) in self.Qsa:
                        u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                                1 + self.Nsa[(s, a)])
                    else:
                        u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            
            # Handle edge case: no valid actions found
            if a == -1:
                self._path.remove(s)  # Clean up path before return
                return 0
                
            # Get next state
            next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
            next_s = self.game.getCanonicalForm(next_s, next_player)

            # Recursive search
            v = self.search(next_s)

            # Update Q value
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1

            self.Ns[s] += 1
            
            # Remove from path before returning
            self._path.remove(s)
            return v  # Don't negate for single-player games
            
        except Exception as e:
            # Ensure path is always cleaned up on exception
            if s in self._path:
                self._path.remove(s)
            log.error(f"Unexpected error in MCTS search: {e}")
            return 0