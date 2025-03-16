from __future__ import print_function
import numpy as np

"""
TicTacToe game implementation including:
1. Board Logic
2. Game Rules
"""

###############################
# TicTacToe Board Logic
###############################

class Board():
    """
    Board class for TicTacToe.
    Board data: 1=X, -1=O, 0=empty
    First dim is column, 2nd is row:
       pieces[0][0] is the top left square,
       pieces[2][0] is the bottom left square,
    """
    def __init__(self, n=3):
        "Set up initial board configuration."
        # Inform MCTS that this is a 2 player game!
        self.is_two_player = True
        
        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for X, -1 for O)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.add((x,y))
        return list(moves)

    def has_legal_moves(self):
        """Returns True if has legal move else False"""
        # Get all the empty squares (color==0)
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False
    
    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=X,-1=O)
        @return: bool
        """
        # Check horizontal sequences
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
                else:
                    count = 0
                if count == self.n:
                    return True
                    
        # Check vertical sequences
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self[x][y] == color:
                    count += 1
                else:
                    count = 0
                if count == self.n:
                    return True
                    
        # Check main diagonal
        count = 0
        for i in range(self.n):
            if self[i][i] == color:
                count += 1
            else:
                count = 0
            if count == self.n:
                return True
                
        # Check other diagonal
        count = 0
        for i in range(self.n):
            if self[i][self.n-i-1] == color:
                count += 1
            else:
                count = 0
            if count == self.n:
                return True
                
        return False

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color of the piece to play (1=X,-1=O)
        """
        (x,y) = move
        # Add the piece to the empty square.
        assert self[x][y] == 0
        self[x][y] = color

###############################
# Game Class
###############################

class TicTacToeGame():
    """
    Game class implementation for TicTacToe.
    """
    # Flag to indicate this is a 2-player game for MCTS
    is_two_player = True
    
    def __init__(self, n=3):
        self.n = n
        
    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)
        
    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)
        
    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1
        
    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)
        
    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)
        
    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
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
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l
        
    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tobytes()
        
    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == -1: print("O ", end="")
                elif piece == 1: print("X ", end="")
                else:
                    if x==n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")