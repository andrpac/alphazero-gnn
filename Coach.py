import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        trainExamples = []
        gnnExamples = []  # For GNN training with sliding window
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            # Standard MCTS search
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            
            # Collect standard training examples
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            
            # If using GNN, perform tree expansion for sliding window training
            if hasattr(self.args, 'use_gnn') and self.args.use_gnn:
                # Expand tree to get improved policy and value estimates
                expanded_nodes = self.mcts.expand_tree(canonicalBoard, 
                                                     expand_by=getattr(self.args, 'expand_by', 5))
                
                # Process expanded nodes for GNN training
                for s, (initial_pi, initial_v, expanded_pi, expanded_v) in expanded_nodes.items():
                    # Find matching board state in symmetries
                    for b, _ in sym:
                        if self.game.stringRepresentation(b) == s:
                            # Add example: board, player, initial_pi, initial_v, expanded_pi, expanded_v
                            gnnExamples.append([b, self.curPlayer, initial_pi, initial_v, expanded_pi, expanded_v, None])
                            break

            # Choose action and make move
            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                # Game ended, return examples with proper rewards
                std_examples = [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
                
                # Add GNN examples if using GNN
                if hasattr(self.args, 'use_gnn') and self.args.use_gnn and gnnExamples:
                    # Format: (board, player, initial_pi, initial_v, expanded_pi, expanded_v, reward)
                    gnn_examples = [(x[0], x[1], x[2], x[3], x[4], x[5], r * ((-1) ** (x[1] != self.curPlayer))) 
                                  for x in gnnExamples]
                    return std_examples, gnn_examples
                
                return std_examples, []

    def getCheckpointFile(self, iteration):
        base_name = f'checkpoint_{iteration}'
        if hasattr(self.args, 'use_gnn') and self.args.use_gnn:
            base_name += '_gnn'
        return base_name + '.pth.tar'

    def learn(self):
        for i in range(1, self.args.numIters + 1):
            log.info(f'Starting Iter #{i} ...')

            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                iterationGnnExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    std_examples, gnn_examples = self.executeEpisode()
                    iterationTrainExamples += std_examples
                    if gnn_examples:
                        iterationGnnExamples += gnn_examples

                # save the iteration examples to the history 
                self.trainExamplesHistory.append((iterationTrainExamples, iterationGnnExamples))

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
                
            # backup history to a file
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            gnnExamples = []
            for std_ex, gnn_ex in self.trainExamplesHistory:
                trainExamples.extend(std_ex)
                if gnn_ex:
                    gnnExamples.extend(gnn_ex)
            shuffle(trainExamples)
            if gnnExamples:
                shuffle(gnnExamples)

            # training new network, keeping a copy of the old one
            temp_filename = 'temp.pth.tar'
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=temp_filename)
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=temp_filename)
            pmcts = MCTS(self.game, self.pnet, self.args)

            # Train with standard examples and GNN examples if available
            if hasattr(self.args, 'use_gnn') and self.args.use_gnn and gnnExamples:
                log.info(f"Training with {len(trainExamples)} standard examples and {len(gnnExamples)} GNN examples")
                self.nnet.train(trainExamples, gnnExamples)
            else:
                self.nnet.train(trainExamples)
                
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            
            # MODIFICATION: Always save the best model for the first iteration
            if i == 1:
                log.info('FIRST ITERATION: SAVING AS BEST MODEL')
                accept_model = True
            else:
                # Normal acceptance logic
                accept_model = (pwins + nwins > 0) and (float(nwins) / (pwins + nwins) >= self.args.updateThreshold)
            
            if not accept_model:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=temp_filename)
            else:
                log.info('ACCEPTING NEW MODEL')
                
                # Determine checkpoint naming based on whether this is GNN or not
                is_using_gnn = hasattr(self.args, 'use_gnn') and self.args.use_gnn
                
                if is_using_gnn:
                    best_filename = 'best_gnn.pth.tar'
                    iter_filename = f'checkpoint_{i}_gnn.pth.tar'
                else:
                    best_filename = 'best.pth.tar'
                    iter_filename = f'checkpoint_{i}.pth.tar'
                
                # Save iteration-specific checkpoint
                log.info(f'Saving iteration checkpoint to {self.args.checkpoint}/{iter_filename}')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=iter_filename)
                
                # Save as best model
                log.info(f'Saving best model to {self.args.checkpoint}/{best_filename}')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=best_filename)

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True