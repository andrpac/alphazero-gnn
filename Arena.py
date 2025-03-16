import logging
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

class Arena:
    """
    An Arena class that works for both single-player and two-player games.
    - For single-player games, it compares which model solves the puzzle more effectively.
    - For two-player games, it pits two players against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player1: function that takes board as input, returns action
            player2: function that takes board as input, returns action
            game: Game object
            display: a function that takes board as input and prints it
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        
        # Detect if the game is single-player or two-player
        self.is_single_player = hasattr(self.game, 'is_two_player') and not self.game.is_two_player
    
    def playGameForSinglePlayer(self, player, board_state=None, verbose=False):
        """
        Executes one episode with the given player from a given board state.
        Used for single-player games like FrozenLake.
        
        Returns:
            (result, steps): 
                result is 1 for win, -1 for loss, 0 for draw/timeout
                steps is the number of steps taken
        """
        # Make copy to avoid modifying original
        board = np.copy(board_state) if board_state is not None else self.game.getInitBoard()
        it = 0
        
        # Set a reasonable maximum number of steps to prevent infinite loops
        # More lenient max steps for larger boards
        max_steps = self.game.getBoardSize()[0] * self.game.getBoardSize()[1] * 5
        
        # Initialize player if necessary
        if hasattr(player, "startGame"):
            player.startGame()
            
        # Game loop
        while True:
            # Check for termination conditions
            game_result = self.game.getGameEnded(board, 1)
            if game_result != 0 or it >= max_steps:
                break
                
            it += 1
            if verbose:
                assert self.display
                print(f"Turn {it}, Player 1")
                self.display(board)
                
            # Get action from player
            canonicalBoard = self.game.getCanonicalForm(board, 1)
            
            try:
                action = player(canonicalBoard)
                
                # Check if action is valid
                valids = self.game.getValidMoves(canonicalBoard, 1)
                
                if valids[action] == 0:
                    log.error(f'Action {action} is not valid!')
                    log.debug(f'valids = {valids}')
                    # Choose a random valid action as fallback
                    valid_actions = np.where(valids == 1)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        # If no valid actions, game over
                        log.error('No valid actions available!')
                        break
                
                # Make the move
                board, _ = self.game.getNextState(board, 1, action)
            except Exception as e:
                log.error(f'Error during gameplay: {e}')
                break
            
        # End game for player if necessary
        if hasattr(player, "endGame"):
            player.endGame()
            
        if verbose:
            assert self.display
            print(f"Game over: Turn {it}, Result {self.game.getGameEnded(board, 1)}")
            self.display(board)
        
        # Check for timeout
        if it >= max_steps and self.game.getGameEnded(board, 1) == 0:
            return 0, it  # Draw for timeout
            
        return self.game.getGameEnded(board, 1), it
    
    def playGameForTwoPlayer(self, verbose=False):
        """
        Executes one episode of a two-player game.
        
        Returns:
            winner: player who won the game (1 if player1, -1 if player2)
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in [players[0], players[2]]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        for player in [players[0], players[2]]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)
    
    def playGame(self, verbose=False):
        """
        Wrapper that calls the appropriate game playing method based on game type.
        """
        if self.is_single_player:
            # For single-player games, we only care about player1's result
            result, _ = self.playGameForSinglePlayer(self.player1, verbose=verbose)
            return result
        else:
            # For two-player games, use the existing method
            return self.playGameForTwoPlayer(verbose=verbose)
    
    def playGamesForSinglePlayer(self, num, verbose=False):
        """
        Plays num games, with both players playing on the same map each time.
        For single-player games like FrozenLake.
        
        Returns:
            oneWon: games where player1 (old model) did better
            twoWon: games where player2 (new model) did better
            draws: games where they performed equally
        """
        oneWon = 0
        twoWon = 0
        draws = 0
        
        total_steps1 = 0
        total_steps2 = 0
        success1 = 0
        success2 = 0
        
        for i in tqdm(range(num), desc="Arena.playGames (Single-Player)"):
            # Generate a fresh board using standard map
            board = self.game.getInitBoard()
            
            # Compare which player does better on this board
            result1, steps1 = self.playGameForSinglePlayer(self.player1, board)
            result2, steps2 = self.playGameForSinglePlayer(self.player2, board)
            
            # Track success metrics
            total_steps1 += steps1
            total_steps2 += steps2
            
            if result1 > 0:
                success1 += 1
                
            if result2 > 0:
                success2 += 1
            
            # Determine winner based on success and efficiency
            if result1 > 0 and result2 <= 0:
                # Only player1 succeeded
                oneWon += 1
            elif result2 > 0 and result1 <= 0:
                # Only player2 succeeded
                twoWon += 1
            elif result1 > 0 and result2 > 0:
                # Both succeeded - compare steps
                if steps1 < steps2:
                    oneWon += 1
                elif steps2 < steps1:
                    twoWon += 1
                else:
                    draws += 1
            elif result1 < 0 and result2 < 0:
                # Both failed - compare how quickly they failed
                # Quickly failing is worse, as it means the agent made bad decisions
                if steps1 > steps2:
                    oneWon += 1  # Player1 survived longer
                elif steps2 > steps1:
                    twoWon += 1  # Player2 survived longer
                else:
                    draws += 1
            else:
                # Other cases (both timeout or draw)
                draws += 1
        
        # Log the results
        log.info(f"Results - Player1 wins: {oneWon}, Player2 wins: {twoWon}, Draws: {draws}")
        
        if num > 0:
            log.info(f"Success rates - Player1: {success1/num:.2f}, Player2: {success2/num:.2f}")
            
            if success1 > 0:
                log.info(f"Avg steps for success - Player1: {total_steps1/max(1,success1):.2f}")
            if success2 > 0:
                log.info(f"Avg steps for success - Player2: {total_steps2/max(1,success2):.2f}")
        
        # Calculate win rates
        total_games = oneWon + twoWon + draws
        if total_games > 0:
            log.info(f"Win rates - Player1: {oneWon/total_games:.2f}, Player2: {twoWon/total_games:.2f}, Draws: {draws/total_games:.2f}")
        
        return oneWon, twoWon, draws
    
    def playGamesForTwoPlayer(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games. For two-player games like TicTacToe.
        
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (Two-Player) (1)"):
            gameResult = self.playGameForTwoPlayer(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (Two-Player) (2)"):
            gameResult = self.playGameForTwoPlayer(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
    
    def playGames(self, num, verbose=False):
        """
        Wrapper that calls the appropriate games playing method based on game type.
        """
        if self.is_single_player:
            return self.playGamesForSinglePlayer(num, verbose)
        else:
            return self.playGamesForTwoPlayer(num, verbose)