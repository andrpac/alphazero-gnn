from __future__ import print_function
import numpy as np

"""
Connect4 game implementation including:
1. Board Logic
2. Game Rules
"""

###############################
# Connect4 Board Logic
###############################

class Board():
    """
    Board class for Connect4.
    Board data: 1=X (first player), -1=O (second player), 0=empty
    First dim is column, 2nd is row:
       pieces[0][0] is the bottom left square,
       pieces[board_size-1][0] is the bottom right square,
       pieces[0][board_size-1] is the top left square
    """
    def __init__(self, board_size=7):
        "Set up initial board configuration."
        # Inform MCTS that this is a 2 player game!
        self.is_two_player = True
        
        self.board_size = board_size
        # Create the empty board array.
        self.pieces = [None]*self.board_size
        for i in range(self.board_size):
            self.pieces[i] = [0]*self.board_size

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color=None):
        """Returns all the legal moves for the given color.
        (1 for X, -1 for O)
        @param color not used and came from previous version.
        """
        moves = set()  # stores the legal moves.

        # Get all the non-full columns
        for x in range(self.board_size):
            if self[x][self.board_size-1] == 0:
                moves.add(x)
        
        return list(moves)

    def has_legal_moves(self):
        """Returns True if has legal move else False"""
        # Check if any column is not full
        for x in range(self.board_size):
            if self[x][self.board_size-1] == 0:
                return True
        return False
    
    def get_drop_position(self, column):
        """Get the position where the piece would land if dropped in the column"""
        for y in range(self.board_size):
            if self[column][y] == 0:
                return y
        return -1  # Column is full
    
    def is_win(self, color):
        """Check whether the given player has four in a row; 
        @param color (1=X,-1=O)
        @return: bool
        """
        # Number of pieces needed in a row to win (standard Connect4 is 4)
        win_length = min(4, self.board_size)
        
        # Check horizontal
        for y in range(self.board_size):
            for x in range(self.board_size - win_length + 1):
                if all(self[x+i][y] == color for i in range(win_length)):
                    return True
                    
        # Check vertical
        for x in range(self.board_size):
            for y in range(self.board_size - win_length + 1):
                if all(self[x][y+i] == color for i in range(win_length)):
                    return True
                    
        # Check diagonal /
        for x in range(self.board_size - win_length + 1):
            for y in range(win_length - 1, self.board_size):
                if all(self[x+i][y-i] == color for i in range(win_length)):
                    return True
                
        # Check diagonal \
        for x in range(self.board_size - win_length + 1):
            for y in range(self.board_size - win_length + 1):
                if all(self[x+i][y+i] == color for i in range(win_length)):
                    return True
                
        return False

    def execute_move(self, column, color):
        """Perform the given move on the board; 
        color gives the color of the piece to play (1=X,-1=O)
        """
        # Find the position where the piece will land in the column
        y = self.get_drop_position(column)
        
        # Add the piece to the board
        assert y != -1, "Column is full!"
        self[column][y] = color

###############################
# Game Class
###############################

class Connect4Game():
    """
    Game class implementation for Connect4.
    """
    # Flag to indicate this is a 2-player game for MCTS
    is_two_player = True
    
    def __init__(self, board_size=7):
        """
        Initialize with a square board of the given size.
        Standard Connect4 uses a 7x6 board, but we use a square board for simplicity.
        """
        self.board_size = board_size
        
    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.board_size)
        return np.array(b.pieces)
        
    def getBoardSize(self):
        # (a,b) tuple
        return (self.board_size, self.board_size)
        
    def getActionSize(self):
        # return number of actions
        return self.board_size + 1  # All columns plus pass
        
    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.board_size:  # pass move
            return (board, -player)
            
        b = Board(self.board_size)
        b.pieces = np.copy(board)
        b.execute_move(action, player)
        return (b.pieces, -player)
        
    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*(self.getActionSize())
        b = Board(self.board_size)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves()
        if len(legalMoves) == 0:
            valids[-1] = 1  # Pass move if no valid moves
            return np.array(valids)
            
        for column in legalMoves:
            valids[column] = 1
            
        return np.array(valids)
        
    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.board_size)
        b.pieces = np.copy(board)
        
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
            
        # draw has a very little value
        return 1e-4
        
    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board
        
    def getSymmetries(self, board, pi):
        """
        Mirror symmetry for Connect4 - the game is symmetric about the vertical axis.
        No rotational symmetry due to gravity.
        
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()
            
        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector.
        """
        assert(len(pi) == self.board_size + 1)  # action size = columns + 1 pass move
        
        # Original form
        forms = [(board, pi)]
        
        # Mirror form - note that we need to mirror the policy vector too
        mirror_board = np.fliplr(board)
        mirror_pi = np.copy(pi)
        # Mirror all moves except the pass move
        for i in range(self.board_size):
            mirror_pi[i] = pi[self.board_size - 1 - i]
            
        forms.append((mirror_board, mirror_pi))
        return forms
        
    def stringRepresentation(self, board):
        # numpy array (canonical board)
        return board.tobytes()
        
    @staticmethod
    def display(board):
        board_size = board.shape[1]
        
        print("  ", end="")
        for j in range(board_size):
            print(f"{j} ", end="")
        print("")
        
        print(" +", end="")
        for j in range(board_size):
            print("--", end="")
        print("+")
        
        for i in range(board_size-1, -1, -1):  # Start from top row
            print(f"{i}|", end="")
            for j in range(board_size):
                piece = board[j][i]    # get the piece to print
                if piece == -1: print("O ", end="")
                elif piece == 1: print("X ", end="")
                else: print(". ", end="")
            print("|")
            
        print(" +", end="")
        for j in range(board_size):
            print("--", end="")
        print("+")
        
        print("  ", end="")
        for j in range(board_size):
            print(f"{j} ", end="")
        print("")