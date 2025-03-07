import numpy as np
import gymnasium as gym

class FrozenLakeGame:
    """FrozenLake implementation for AlphaZero-style MCTS"""

    def __init__(self, map_size=4):
        """Initialize the game environment"""
        self.map_size = map_size
        self.action_size = 4  # up, right, down, left
        
        # Create environment with a simpler map layout
        if map_size == 4:
            # Custom map with fewer holes for easier learning
            self.env = gym.make('FrozenLake-v1', desc=[
                "SFFF",  # S: start, F: frozen (safe), H: hole, G: goal
                "FHFF",  # Only one hole in row 2
                "FFFF",  # No holes in row 3
                "FFFG"   # G: goal at bottom right
            ], is_slippery=False, render_mode=None)
        else:
            # Use standard 8x8 map for larger size
            self.env = gym.make('FrozenLake8x8-v1', is_slippery=False, render_mode=None)
        
        # Reset environment and get initial state
        self.env.reset()
        
        # Store map description
        self.desc = self.env.unwrapped.desc
        
        # Find goal position
        self.goal_pos = self.find_goal_position()
        
        # Placeholder for current board (for rendering)
        self.board = None
    
    def find_goal_position(self):
        """Find the position of the goal on the map"""
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.desc[i][j] == b'G':
                    return (i, j)
        return (self.map_size-1, self.map_size-1)  # Default to bottom-right
    
    def getInitBoard(self):
        """Get initial board state (one-hot encoded)"""
        board = np.zeros((self.map_size, self.map_size))
        board[0, 0] = 1  # Initial position is top-left
        return board
    
    def getBoardSize(self):
        """Get board dimensions"""
        return (self.map_size, self.map_size)
    
    def getActionSize(self):
        """Get number of possible actions"""
        return self.action_size
    
    def getNextState(self, board, player, action):
        """Get next state after applying action"""
        # Get current position
        pos = np.unravel_index(np.argmax(board) if np.sum(board) > 0 else 0, board.shape)
        row, col = pos
        
        # Calculate new position based on action
        # 0: up, 1: right, 2: down, 3: left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = directions[action]
        new_row, new_col = row + dr, col + dc
        
        # Check boundaries - stay in place if would go out of bounds
        if not (0 <= new_row < self.map_size and 0 <= new_col < self.map_size):
            new_row, new_col = row, col
        
        # Create next board with new position
        next_board = np.zeros_like(board)
        next_board[new_row, new_col] = 1
        
        return next_board, player  # Player doesn't change in single-player games
    
    def getValidMoves(self, board, player):
        """Get valid moves from current state"""
        # All moves are valid by default
        valid_moves = np.ones(self.action_size)
        
        # If game ended, no moves are valid
        if self.getGameEnded(board, player) != 0:
            return np.zeros(self.action_size)
        
        # Get current position
        pos = np.unravel_index(np.argmax(board) if np.sum(board) > 0 else 0, board.shape)
        row, col = pos
        
        # Check boundaries - can't move outside the grid
        if row == 0: valid_moves[0] = 0  # Can't go up
        if row == self.map_size - 1: valid_moves[2] = 0  # Can't go down
        if col == 0: valid_moves[3] = 0  # Can't go left
        if col == self.map_size - 1: valid_moves[1] = 0  # Can't go right
        
        return valid_moves
    
    def getGameEnded(self, board, player):
        """
        Check if game has ended
        Returns:
        - 0 if game not ended
        - 1 if win
        - -1 if loss
        """
        if np.sum(board) == 0:
            return 0
        
        # Get current position
        pos = np.unravel_index(np.argmax(board), board.shape)
        row, col = pos
        
        # Win condition: reached goal
        if self.desc[row][col] == b'G':
            return 1.0
        
        # Loss condition: fell in hole
        if self.desc[row][col] == b'H':
            return -1.0
        
        # Non-terminal state
        return 0
    
    # This is a new function that provides reward shaping outside of the terminal states
    # Used for training guidance but not for MCTS evaluation
    def getDistanceReward(self, board):
        """Get distance-based reward for non-terminal states (for training only)"""
        if np.sum(board) == 0:
            return 0
            
        # Get current position
        pos = np.unravel_index(np.argmax(board), board.shape)
        row, col = pos
        
        # For non-terminal states, return distance-based reward
        goal_row, goal_col = self.goal_pos
        
        # Calculate Manhattan distance to goal
        distance = abs(row - goal_row) + abs(col - goal_col)
        
        # Normalize distance to [0, 1] range
        max_distance = self.map_size * 2
        normalized_distance = distance / max_distance
        
        # Return reward inversely proportional to distance
        # Higher reward when closer to goal
        return 0.1 + 0.4 * (1.0 - normalized_distance)
    
    def getCanonicalForm(self, board, player):
        """Single-player game, so canonical form is just the board"""
        return board
    
    def getSymmetries(self, board, pi):
        """No useful symmetries in FrozenLake"""
        return [(board, pi)]
    
    def stringRepresentation(self, board):
        """Unique string representation for MCTS node tracking"""
        if np.sum(board) == 0:
            return "empty"
        pos = np.unravel_index(np.argmax(board), board.shape)
        return f"{pos[0]},{pos[1]}"
    
    def render(self):
        """Render the environment (optional)"""
        try:
            renderer = gym.make(
                f'FrozenLake{"8x8-" if self.map_size == 8 else "-"}v1',
                desc=self.desc,
                is_slippery=False,
                render_mode='human'
            )
            if hasattr(self, 'board') and self.board is not None:
                pos = np.unravel_index(np.argmax(self.board), self.board.shape)
                state_idx = pos[0] * self.map_size + pos[1]
                renderer.unwrapped.s = state_idx
            renderer.render()
        except Exception as e:
            print(f"Rendering error: {e}")