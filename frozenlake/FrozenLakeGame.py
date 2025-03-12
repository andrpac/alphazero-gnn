import numpy as np
import gymnasium as gym

class FrozenLakeGame:
    """FrozenLake implementation for AlphaZero-style MCTS"""

    def __init__(self, map_size=4, custom_map=None, is_slippery=False, render_mode=None):
        """
        Initialize the game environment
        
        Args:
            map_size: Size of the map (4 or 8)
            custom_map: Optional custom map layout as a list of strings
            is_slippery: Whether the ice is slippery (stochastic transitions)
            render_mode: Render mode for gymnasium
        """
        # Must be set up for single player games
        self.is_single_player = True

        self.map_size = map_size
        self.action_size = 4  # up, right, down, left
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        
        # Create environment with appropriate map
        if custom_map is not None:
            # Use provided custom map
            self.env = gym.make('FrozenLake-v1', desc=custom_map, 
                               is_slippery=is_slippery, render_mode=render_mode)
        elif map_size == 8:
            # Use standard 8x8 map
            self.env = gym.make('FrozenLake8x8-v1', 
                               is_slippery=is_slippery, render_mode=render_mode)
        else:
            # Default 4x4 map
            self.env = gym.make('FrozenLake-v1', 
                               is_slippery=is_slippery, render_mode=render_mode)
        
        # Reset environment and get initial state
        self.env.reset()
        
        # Store map description
        self.desc = self.env.unwrapped.desc
        self.map_size = len(self.desc)  # Update map_size from actual map dimensions
        
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
        # Find the starting position 'S' in the map
        start_pos = None
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.desc[i][j] == b'S':
                    start_pos = (i, j)
                    break
            if start_pos:
                break
        
        # Default to top-left if 'S' not found
        if not start_pos:
            start_pos = (0, 0)
            
        board = np.zeros((self.map_size, self.map_size))
        board[start_pos] = 1
        return board
    
    def getBoardSize(self):
        """Get board dimensions"""
        return (self.map_size, self.map_size)
    
    def getActionSize(self):
        """Get number of possible actions"""
        return self.action_size
    
    def getNextState(self, board, player, action):
        """
        Get next state after applying action
        
        Note: In slippery mode, this only returns the deterministic outcome
        The actual gameplay should handle stochasticity separately
        """
        if np.sum(board) == 0:
            # If board is empty, create a new one with starting position
            return self.getInitBoard(), player
            
        # Get current position
        pos = np.unravel_index(np.argmax(board), board.shape)
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
        
        # Store current board for rendering
        self.board = next_board
        
        return next_board, player  # Player doesn't change in single-player games
    
    def getValidMoves(self, board, player):
        """Get valid moves from current state"""
        # All moves are valid by default
        valid_moves = np.ones(self.action_size, dtype=np.int8)
        
        # If game ended, no moves are valid
        if self.getGameEnded(board, player) != 0:
            return np.zeros(self.action_size, dtype=np.int8)
        
        # Get current position
        if np.sum(board) == 0:
            # If the board is empty, consider it's at the start position
            # This is a special case handling
            return valid_moves
        
        pos = np.unravel_index(np.argmax(board), board.shape)
        row, col = pos
        
        # Check boundaries - can't move outside the grid
        if row == 0: valid_moves[0] = 0  # Can't go up
        if row == self.map_size - 1: valid_moves[2] = 0  # Can't go down
        if col == 0: valid_moves[3] = 0  # Can't go left
        if col == self.map_size - 1: valid_moves[1] = 0  # Can't go right
        
        # Discourage but don't forbid moving into holes
        # This makes learning easier while keeping the game rules intact
        for a in range(self.action_size):
            if valid_moves[a]:
                # Calculate new position
                dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
                new_row, new_col = row + dr, col + dc
                
                # Check if in bounds
                if 0 <= new_row < self.map_size and 0 <= new_col < self.map_size:
                    # Check if it's a hole - don't forbid but will learn to avoid
                    if self.desc[new_row][new_col] == b'H':
                        # Keep valid but will learn negative reward
                        pass
        
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
        """Render the environment"""
        try:
            # If we already have a render_mode, use the existing environment
            if self.render_mode:
                if hasattr(self, 'board') and self.board is not None:
                    pos = np.unravel_index(np.argmax(self.board), self.board.shape)
                    state_idx = pos[0] * self.map_size + pos[1]
                    self.env.unwrapped.s = state_idx
                self.env.render()
            else:
                # Create a temporary environment with human rendering
                renderer = gym.make(
                    'FrozenLake-v1' if self.map_size <= 4 else 'FrozenLake8x8-v1',
                    desc=self.desc,
                    is_slippery=self.is_slippery,
                    render_mode='human'
                )
                if hasattr(self, 'board') and self.board is not None:
                    pos = np.unravel_index(np.argmax(self.board), self.board.shape)
                    state_idx = pos[0] * self.map_size + pos[1]
                    renderer.unwrapped.s = state_idx
                renderer.render()
        except Exception as e:
            print(f"Rendering error: {e}")