#!/usr/bin/env python3
import os
import re
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from register import get_game
from Arena import Arena
from MCTS import MCTS

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, val):
        self[name] = val
    def copy(self):
        return dotdict(dict.copy(self))

def load_config(path_to_yaml):
    with open(path_to_yaml, "r") as f:
        return yaml.safe_load(f)

def config_to_args(cfg_dict):
    args = dotdict({})
    for section in cfg_dict.values():
        for k, v in section.items():
            args[k] = v
    if "checkpoint_path" in args and "checkpoint" not in args:
        args.checkpoint = args.checkpoint_path
    elif "checkpoint" in args and "checkpoint_path" not in args:
        args.checkpoint_path = args.checkpoint
    return args

def find_available_iterations(checkpoint_folder):
    normal_pattern = re.compile(r"^checkpoint_best_(\d+)\.pth\.tar$")
    gnn_pattern    = re.compile(r"^checkpoint_best_(\d+)_gnn\.pth\.tar$")
    normal_iters = set()
    gnn_iters    = set()
    for fname in os.listdir(checkpoint_folder):
        m_norm = normal_pattern.match(fname)
        if m_norm:
            normal_iters.add(int(m_norm.group(1)))
            continue
        m_gnn = gnn_pattern.match(fname)
        if m_gnn:
            gnn_iters.add(int(m_gnn.group(1)))
            continue
    return sorted(normal_iters), sorted(gnn_iters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--output", type=str, default="heatmap.png")
    args_cli = parser.parse_args()

    cfg_dict = load_config(args_cli.config)
    config_args = config_to_args(cfg_dict)
    config_args.game = "connect4"
    checkpoint_base = args_cli.checkpoint.strip()
    if not os.path.isdir(checkpoint_base):
        raise FileNotFoundError(f"Cannot find checkpoint folder: {checkpoint_base!r}")

    normal_iters_all, gnn_iters_all = find_available_iterations(checkpoint_base)
    if not normal_iters_all:
        files = os.listdir(checkpoint_base)
        raise RuntimeError(f"No normal checkpoints found in {checkpoint_base!r}, contents: {files}")
    if not gnn_iters_all:
        files = os.listdir(checkpoint_base)
        raise RuntimeError(f"No GNN checkpoints found in {checkpoint_base!r}, contents: {files}")

    step = args_cli.step
    normal_iters = normal_iters_all[::step]
    gnn_iters    = gnn_iters_all[::step]

    print("Normal iterations to test:", normal_iters)
    print("GNN iterations to test:   ", gnn_iters)

    GameClass_norm, BaseNetClass = get_game("connect4", use_gnn=False)
    _,            GnnNetClass   = get_game("connect4", use_gnn=True)
    game = GameClass_norm(board_size=config_args.board_size)

    n_norm = len(normal_iters)
    n_gnn  = len(gnn_iters)
    win_rates = np.zeros((n_norm, n_gnn), dtype=np.float32)

    for i_idx, it_norm in enumerate(normal_iters):
        fname_norm = f"checkpoint_best_{it_norm}.pth.tar"
        fullpath_norm = os.path.join(checkpoint_base, fname_norm)
        if not os.path.isfile(fullpath_norm):
            raise FileNotFoundError(f"Missing normal checkpoint: {fullpath_norm!r}")
        print(f"Loading normal model: {fname_norm}")
        normal_net = BaseNetClass(game, config_args)
        normal_net.load_checkpoint(checkpoint_base, fname_norm)
        normal_net.nnet.eval()
        if torch.cuda.is_available():
            normal_net.nnet.to(normal_net.device)
        else:
            normal_net.device = torch.device("cpu")
            normal_net.nnet.to(normal_net.device)

        for j_idx, it_gnn in enumerate(gnn_iters):
            fname_gnn = f"checkpoint_best_{it_gnn}_gnn.pth.tar"
            fullpath_gnn = os.path.join(checkpoint_base, fname_gnn)
            if not os.path.isfile(fullpath_gnn):
                raise FileNotFoundError(f"Missing GNN checkpoint: {fullpath_gnn!r}")
            print(f"  Loading GNN model: {fname_gnn}")
            gnn_net = GnnNetClass(game, config_args)
            gnn_net.load_checkpoint(checkpoint_base, fname_gnn)
            gnn_net.nnet.eval()
            if torch.cuda.is_available():
                gnn_net.nnet.to(gnn_net.device)
            else:
                gnn_net.device = torch.device("cpu")
                gnn_net.nnet.to(gnn_net.device)

            print(f"    Pitting normal@{it_norm} vs gnn@{it_gnn}")
            mcts_norm = MCTS(game, normal_net,   config_args)
            mcts_gnn  = MCTS(game, gnn_net,      config_args)

            def policy_norm(board):
                return np.argmax(mcts_norm.getActionProb(board, temp=0))

            def policy_gnn(board):
                return np.argmax(mcts_gnn.getActionProb(board, temp=0))

            arena = Arena(policy_gnn, policy_norm, game)
            new_wins, prev_wins, draws = arena.playGames(config_args.arenaCompare)

            total = new_wins + prev_wins
            win_rate = new_wins / total if total > 0 else 0.0
            win_rates[i_idx, j_idx] = win_rate

            print(f"      result: new_wins={new_wins}, prev_wins={prev_wins}, draws={draws}, win_rate={win_rate:.2f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        win_rates,
        cmap="RdYlGn",
        origin="lower",
        vmin=args_cli.vmin,
        vmax=args_cli.vmax,
        aspect="equal",
    )

    ax.set_xticks(np.arange(n_gnn))
    ax.set_yticks(np.arange(n_norm))
    ax.set_xticklabels(gnn_iters)
    ax.set_yticklabels(normal_iters)
    ax.set_xlabel("GNN Model Iteration", fontsize=12)
    ax.set_ylabel("Normal Model Iteration", fontsize=12)

    for i in range(n_norm):
        for j in range(n_gnn):
            text = f"{win_rates[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Win Rate", rotation=-90, va="bottom", fontsize=12)

    ax.set_title(f"Win-Rate Heatmap: GNN vs Normal (step={step})", fontsize=14)
    plt.tight_layout()
    plt.savefig(args_cli.output, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
