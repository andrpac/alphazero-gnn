import logging
import numpy as np
import os
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import shuffle
from tqdm import tqdm

from connect4.Connect4GNN import Connect4GNNWrapper
from MCTS import MCTS
from Arena import Arena

log = logging.getLogger(__name__)

def execute_episode_worker(args_tuple):
    game, state_dict_cpu, args = args_tuple
    device = torch.device('cpu')
    nnet = Connect4GNNWrapper(game, args)
    nnet.device = device
    nnet.nnet.to(device)
    nnet.nnet.load_state_dict(state_dict_cpu)
    nnet.nnet.eval()
    mcts = MCTS(game, nnet, args)
    trainExamples = []
    gnnExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    position_counter = 0

    while True:
        episodeStep += 1
        position_counter += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        temp = 1 if episodeStep < args.tempThreshold else 0
        pi = np.array(mcts.getActionProb(canonicalBoard, temp=temp), dtype=np.float64)
        total_pi = pi.sum()
        if total_pi <= 0:
            vm = game.getValidMoves(canonicalBoard, 1)
            pi = vm / (vm.sum() + 1e-10)
        else:
            pi /= total_pi
        symm = game.getSymmetries(canonicalBoard, pi)
        for b_sym, p_sym in symm:
            trainExamples.append((b_sym, curPlayer, p_sym, None))
        if (args.gnn_accumulate or args.gnn_train) and (position_counter % args.gnn_collect_interval == 0):
            (init_pi, init_v, exp_pi, exp_v), tree_data = mcts.expand_tree(canonicalBoard, expand_by=args.expand_by)
            for idx, (b_sym, p_sym) in enumerate(symm):
                if idx == 0:
                    gnnExamples.append((b_sym, curPlayer, init_pi, init_v, exp_pi, exp_v, tree_data))
                else:
                    mirrored_nodes = [np.fliplr(node) for node in tree_data['initial_nodes']]
                    edge_tensor = tree_data['initial_edges']
                    if edge_tensor.numel() > 0:
                        node_count = len(tree_data['initial_nodes'])
                        node_mapping = {}
                        for i in range(node_count):
                            orig_board = tree_data['initial_nodes'][i]
                            mirrored_board = mirrored_nodes[i]
                            for j, mb in enumerate(mirrored_nodes):
                                if np.array_equal(orig_board, mb):
                                    node_mapping[i] = j
                                    break
                        if len(node_mapping) == node_count:
                            new_edges = []
                            for i in range(edge_tensor.shape[1]):
                                src, dst = edge_tensor[0, i].item(), edge_tensor[1, i].item()
                                if src in node_mapping and dst in node_mapping:
                                    new_edges.append([node_mapping[src], node_mapping[dst]])
                            if new_edges:
                                new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
                            else:
                                new_edge_tensor = torch.empty((2, 0), dtype=torch.long)
                        else:
                            new_edge_tensor = edge_tensor
                    else:
                        new_edge_tensor = edge_tensor
                    mirrored_tree = {'initial_nodes': mirrored_nodes, 'initial_edges': new_edge_tensor}
                    gnnExamples.append((b_sym, curPlayer, init_pi, init_v, exp_pi, exp_v, mirrored_tree))
        result = game.getGameEnded(board, curPlayer)
        if result != 0:
            standardized = []
            for i, (b, player, p_sym, _) in enumerate(trainExamples):
                val = result * ((-1) ** (i % 2))
                standardized.append((b, p_sym, val))
            return standardized, gnnExamples
        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(board, curPlayer, action)


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = nnet.__class__(self.game, args)
        self.args = args
        self.pnet.nnet.to(self.nnet.device)
        self.trainExamplesHistory = []
        self.position_counter = 0
        if args.gnn_accumulate:
            run_id = torch.randint(0, 10000, ()).item()
            self.gnn_data_dir = os.path.join(args.checkpoint, f"gnn_data_{run_id}")
            os.makedirs(self.gnn_data_dir, exist_ok=True)
        self.mcts = None
        self.use_parallel = True
        self.num_workers = min(mp.cpu_count() - 1, 4)

    def executeEpisode(self):
        trainExamples = []
        gnnExamples = []
        game = self.game
        args = self.args
        mcts = self.mcts
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            self.position_counter += 1
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = 1 if episodeStep < args.tempThreshold else 0
            pi = np.array(mcts.getActionProb(canonicalBoard, temp=temp), dtype=np.float64)
            total_pi = pi.sum()
            if total_pi <= 0:
                vm = game.getValidMoves(canonicalBoard, 1)
                pi = vm / (vm.sum() + 1e-10)
            else:
                pi /= total_pi
            symm = game.getSymmetries(canonicalBoard, pi)
            for b_sym, p_sym in symm:
                trainExamples.append((b_sym, curPlayer, p_sym, None))
            if (args.gnn_accumulate or args.gnn_train) and (self.position_counter % args.gnn_collect_interval == 0):
                (init_pi, init_v, exp_pi, exp_v), tree_data = mcts.expand_tree(canonicalBoard, expand_by=args.expand_by)
                for idx, (b_sym, p_sym) in enumerate(symm):
                    if idx == 0:
                        gnnExamples.append((b_sym, curPlayer, init_pi, init_v, exp_pi, exp_v, tree_data))
                    else:
                        mirrored_nodes = [np.fliplr(node) for node in tree_data['initial_nodes']]
                        edge_tensor = tree_data['initial_edges']
                        if edge_tensor.numel() > 0:
                            node_count = len(tree_data['initial_nodes'])
                            node_mapping = {}
                            for i in range(node_count):
                                orig_board = tree_data['initial_nodes'][i]
                                mirrored_board = mirrored_nodes[i]
                                for j, mb in enumerate(mirrored_nodes):
                                    if np.array_equal(orig_board, mb):
                                        node_mapping[i] = j
                                        break
                            if len(node_mapping) == node_count:
                                new_edges = []
                                for i in range(edge_tensor.shape[1]):
                                    src, dst = edge_tensor[0, i].item(), edge_tensor[1, i].item()
                                    if src in node_mapping and dst in node_mapping:
                                        new_edges.append([node_mapping[src], node_mapping[dst]])
                                if new_edges:
                                    new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
                                else:
                                    new_edge_tensor = torch.empty((2, 0), dtype=torch.long)
                            else:
                                new_edge_tensor = edge_tensor
                        else:
                            new_edge_tensor = edge_tensor
                        mirrored_tree = {'initial_nodes': mirrored_nodes, 'initial_edges': new_edge_tensor}
                        gnnExamples.append((b_sym, curPlayer, init_pi, init_v, exp_pi, exp_v, mirrored_tree))
            result = game.getGameEnded(board, curPlayer)
            if result != 0:
                standardized = []
                for i, (b, player, p_sym, _) in enumerate(trainExamples):
                    val = result * ((-1) ** (i % 2))
                    standardized.append((b, p_sym, val))
                return standardized, gnnExamples
            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)

    def getCheckpointFile(self, iteration):
        base = f"checkpoint_{iteration}"
        if self.args.gnn_train:
            base += "_gnn"
        return base + ".pth.tar"

    def getBestFilename(self, iteration):
        base = f"checkpoint_best_{iteration}"
        if self.args.gnn_train:
            base += "_gnn"
        return base + ".pth.tar"

    def learn(self):
        game = self.game
        nnet = self.nnet
        pnet = self.pnet
        args = self.args

        for it in range(1, args.numIters + 1):
            log.info(f"------ ITER {it} ------")
            self.position_counter = 0
            iterationStd = []
            iterationGnn = []
            state_dict_cpu = {name: param.cpu() for name, param in nnet.nnet.state_dict().items()}

            if self.use_parallel and self.num_workers > 1:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    for _ in range(args.numEps):
                        futures.append(executor.submit(execute_episode_worker, (game, state_dict_cpu, args)))
                    for future in tqdm(as_completed(futures), total=args.numEps, desc="Self Play (Parallel)"):
                        std_ex, gnn_ex = future.result()
                        iterationStd.extend(std_ex)
                        iterationGnn.extend(gnn_ex)
            else:
                for _ in tqdm(range(args.numEps), desc="Self Play"):
                    self.mcts = MCTS(game, nnet, args)
                    std_ex, gnn_ex = self.executeEpisode()
                    iterationStd.extend(std_ex)
                    iterationGnn.extend(gnn_ex)

            if args.gnn_accumulate:
                path = os.path.join(self.gnn_data_dir, f"iter_{it}.pt")
                torch.save(iterationGnn, path)
                log.info(f"Saved {len(iterationGnn)} GNN examples to {path}")

            self.trainExamplesHistory.append((iterationStd, iterationGnn))
            if len(self.trainExamplesHistory) > args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)

            trainExamples = []
            gnnExamples = []
            for std_hist, gnn_hist in self.trainExamplesHistory:
                trainExamples += std_hist
                gnnExamples += gnn_hist
            shuffle(trainExamples)

            log.info(f"Training base network with {len(trainExamples)} examples")
            nnet.train(trainExamples)

            if args.gnn_train and (it % args.gnn_train_interval == 0) and gnnExamples:
                log.info(f"Training GNN branch with {len(gnnExamples)} examples")
                nnet.train(None, gnnExamples)

            fname_current = self.getCheckpointFile(it)
            nnet.save_checkpoint(args.checkpoint, fname_current)

            if it == 1:
                pnet.load_checkpoint(args.checkpoint, fname_current)
                best_fn = self.getBestFilename(it)
                pnet.save_checkpoint(args.checkpoint, best_fn)
                continue

            prev_best_fn = self.getBestFilename(it - 1)
            pnet.load_checkpoint(args.checkpoint, prev_best_fn)

            mcts_new = MCTS(game, nnet, args)
            mcts_best = MCTS(game, pnet, args)

            def policy_new(b): return np.argmax(mcts_new.getActionProb(b, temp=0))
            def policy_best(b): return np.argmax(mcts_best.getActionProb(b, temp=0))

            arena = Arena(policy_new, policy_best, game)
            new_wins, prev_wins, draws = arena.playGames(args.arenaCompare)
            total = new_wins + prev_wins
            win_rate = new_wins / total if total > 0 else 0

            if win_rate < args.updateThreshold:
                log.info(f"REJECTING NEW MODEL (win_rate={win_rate:.2f})")
                nnet.load_checkpoint(args.checkpoint, prev_best_fn)
            else:
                log.info(f"ACCEPTING NEW MODEL (win_rate={win_rate:.2f})")
                pnet.load_checkpoint(args.checkpoint, fname_current)

            best_fn = self.getBestFilename(it)
            pnet.save_checkpoint(args.checkpoint, best_fn)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        log.info("LEARNING COMPLETE")
