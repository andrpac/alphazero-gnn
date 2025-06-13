import logging
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

class Arena:
    """
    Supports playing games between two players (functions/lambdas returning actions).
    """
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game    = game
        self.display = display
        self.is_single_player = hasattr(self.game, 'is_two_player') and not self.game.is_two_player

    def playGameForSinglePlayer(self, player, board_state=None, verbose=False):
        board = np.copy(board_state) if board_state is not None else self.game.getInitBoard()
        it = 0
        max_steps = self.game.getBoardSize()[0] * self.game.getBoardSize()[1] * 5
        if hasattr(player, "startGame"): player.startGame()
        while True:
            result = self.game.getGameEnded(board, 1)
            if result != 0 or it >= max_steps:
                break
            it += 1
            if verbose:
                assert self.display
                print(f"Turn {it}, Player 1")
                self.display(board)
            canonicalBoard = self.game.getCanonicalForm(board, 1)
            action = player(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            if valids[action] == 0:
                log.error(f'Invalid action {action}!')
                valid_actions = np.where(valids==1)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    log.error('No valid moves!')
                    break
            board, _ = self.game.getNextState(board, 1, action)
        if hasattr(player, "endGame"): player.endGame()
        if verbose:
            assert self.display
            print(f"Game over: Turn {it}, Result {self.game.getGameEnded(board,1)}")
            self.display(board)
        if it >= max_steps and self.game.getGameEnded(board,1)==0:
            return 0, it
        return self.game.getGameEnded(board, 1), it

    def playGameForTwoPlayer(self, verbose=False):
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        for p in [players[0], players[2]]:
            if hasattr(p, "startGame"): p.startGame()
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print(f"Turn {it}, Player {curPlayer}")
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Invalid action {action}!')
                assert valids[action] > 0
            opponent = players[-curPlayer+1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        for p in [players[0], players[2]]:
            if hasattr(p, "endGame"): p.endGame()
        if verbose:
            assert self.display
            print(f"Game over: Turn {it}, Result {self.game.getGameEnded(board,1)}")
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGame(self, verbose=False):
        if self.is_single_player:
            result, _ = self.playGameForSinglePlayer(self.player1, verbose=verbose)
            return result
        else:
            return self.playGameForTwoPlayer(verbose=verbose)

    def playGamesForSinglePlayer(self, num, verbose=False):
        oneWon = twoWon = draws = 0
        total_steps1 = total_steps2 = 0
        success1 = success2 = 0
        for _ in tqdm(range(num), desc="Arena (Single-Player)"):
            board = self.game.getInitBoard()
            result1, steps1 = self.playGameForSinglePlayer(self.player1, board)
            result2, steps2 = self.playGameForSinglePlayer(self.player2, board)
            total_steps1 += steps1
            total_steps2 += steps2
            if result1 > 0: success1 += 1
            if result2 > 0: success2 += 1
            if result1 > 0 and result2 <= 0:
                oneWon += 1
            elif result2 > 0 and result1 <= 0:
                twoWon += 1
            elif result1 > 0 and result2 > 0:
                if steps1 < steps2:
                    oneWon += 1
                elif steps2 < steps1:
                    twoWon += 1
                else:
                    draws += 1
            elif result1 < 0 and result2 < 0:
                if steps1 > steps2:
                    oneWon += 1
                elif steps2 > steps1:
                    twoWon += 1
                else:
                    draws += 1
            else:
                draws += 1
        log.info(f"Results - P1: {oneWon}, P2: {twoWon}, Draws: {draws}")
        if num > 0:
            log.info(f"Success - P1: {success1/num:.2f}, P2: {success2/num:.2f}")
            if success1 > 0:
                log.info(f"Avg steps P1: {total_steps1/max(1,success1):.2f}")
            if success2 > 0:
                log.info(f"Avg steps P2: {total_steps2/max(1,success2):.2f}")
        total_games = oneWon + twoWon + draws
        if total_games > 0:
            log.info(f"Win rates - P1: {oneWon/total_games:.2f}, P2: {twoWon/total_games:.2f}, Draws: {draws/total_games:.2f}")
        return oneWon, twoWon, draws

    def playGamesForTwoPlayer(self, num, verbose=False):
        num = int(num/2)
        oneWon = twoWon = draws = 0
        for _ in tqdm(range(num), desc="Arena (Two-Player) (1)"):
            result = self.playGameForTwoPlayer(verbose=verbose)
            if result == 1: oneWon += 1
            elif result == -1: twoWon += 1
            else: draws += 1
        self.player1, self.player2 = self.player2, self.player1
        for _ in tqdm(range(num), desc="Arena (Two-Player) (2)"):
            result = self.playGameForTwoPlayer(verbose=verbose)
            if result == -1: oneWon += 1
            elif result == 1:  twoWon += 1
            else: draws += 1
        return oneWon, twoWon, draws

    def playGames(self, num, verbose=False):
        if self.is_single_player:
            return self.playGamesForSinglePlayer(num, verbose)
        else:
            return self.playGamesForTwoPlayer(num, verbose)
