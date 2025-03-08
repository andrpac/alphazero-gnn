import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class Arena():
    """
    An Arena class adapted for single-player environments like FrozenLake.
    Each agent plays independent games, and we compare their performance.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player1, player2: two functions that take board as input, return action
            game: Game object
            display: a function that takes board as input and prints it
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
    
    def playGame(self, player, verbose=False):
        """
        Executes one episode with the given player.
        Returns:
            result: 1 for win, -1 for loss, or any other value for draw/timeout
        """
        board = self.game.getInitBoard()
        curPlayer = 1  # Always 1 in single-player environments
        it = 0
        
        # Set a reasonable maximum number of steps to prevent infinite loops
        max_steps = self.game.getBoardSize()[0] * self.game.getBoardSize()[1] * 5
        
        if hasattr(player, "startGame"):
            player.startGame()
            
        while self.game.getGameEnded(board, curPlayer) == 0 and it < max_steps:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
                
            # Get action from player
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            action = player(canonicalBoard)
            
            # Check if action is valid
            valids = self.game.getValidMoves(canonicalBoard, 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                # Find valid actions if possible
                valid_indices = [i for i, v in enumerate(valids) if v == 1]
                if valid_indices:
                    import random
                    action = random.choice(valid_indices)
                    log.warning(f'Selecting random valid action {action} instead')
                else:
                    assert valids[action] > 0
            
            # Make the move
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            
        if hasattr(player, "endGame"):
            player.endGame()
            
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, curPlayer)))
            self.display(board)
        
        # Check for timeout (max steps reached)
        if it >= max_steps and self.game.getGameEnded(board, curPlayer) == 0:
            return 0  # Draw for timeout
            
        return self.game.getGameEnded(board, curPlayer)
    
    def playGames(self, num, verbose=False):
        """
        Each player plays num/2 independent games.
        
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws: games that ended in draw/timeout
        """
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        
        # Player 1 plays their games
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(self.player1, verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                # Losses count as draws for comparison purposes in FrozenLake
                draws += 1
            else:
                draws += 1
        
        # Player 2 plays their games
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(self.player2, verbose=verbose)
            if gameResult == 1:
                twoWon += 1
            elif gameResult == -1:
                # Losses count as draws for comparison purposes in FrozenLake
                draws += 1
            else:
                draws += 1
        
        return oneWon, twoWon, draws