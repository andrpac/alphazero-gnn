#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import numpy._core.multiarray as multiarray
import torch.nn.functional as F

from main import load_config, config_to_args, get_checkpoint_path, create_game_instance
from register import get_game
from MCTS import MCTS
from Arena import Arena

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_all_gnn_data(folder):
    """Load all .pt files in folder as pickles, returning a flat list of GNN examples."""
    examples = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.pt'):
            continue
        path = os.path.join(folder, fn)
        with torch.serialization.safe_globals([multiarray._reconstruct]):
            chunk = torch.load(path, map_location='cpu', weights_only=False)
        examples.extend(chunk)
    return examples


def strip_edges(gnn_data):
    """Replace every example's edge_index with an empty tensor (no edges)."""
    stripped = []
    for b, pl, ip, iv, ep, ev, tree_data in gnn_data:
        no_edges = torch.empty((2, 0), dtype=torch.long)
        new_tree = {
            'initial_nodes': tree_data['initial_nodes'],
            'initial_edges': no_edges
        }
        stripped.append([b, pl, ip, iv, ep, ev, new_tree])
    return stripped


def train_gnn_model(model, gnn_data, args):
    """Train the GNN, returning train/val π-loss and v-loss per epoch,
    plus baseline initial-policy KL and initial-value error metrics."""
    device = next(model.gnn.parameters()).device
    h, w = model.game.getBoardSize()
    channels = model.feature_dim // (h * w)

    # Precompute features and prepare tuples: feats, edges, root, init_pi, init_v, exp_pi, exp_v
    precomputed = []
    for ex in gnn_data:
        nodes = ex[6]['initial_nodes']
        raw_edges = ex[6]['initial_edges']
        unique_edges = torch.unique(raw_edges, dim=1).to(device)
        with torch.no_grad():
            t_nodes = torch.FloatTensor(np.array(nodes)).unsqueeze(1).to(device)
            feats = model.nnet.extract_features(t_nodes).view(t_nodes.size(0), -1)
        init_pi = torch.FloatTensor(ex[2]).to(device)
        init_v  = torch.tensor(ex[3], dtype=torch.float, device=device)
        exp_pi  = torch.FloatTensor(ex[4]).to(device)
        exp_v   = torch.tensor(ex[5], dtype=torch.float, device=device)
        precomputed.append((feats, unique_edges, 0, init_pi, init_v, exp_pi, exp_v))

    # Split into train/val
    num = len(precomputed)
    split = max(1, int(0.1 * num))
    perm = np.random.permutation(num)
    val_data = [precomputed[i] for i in perm[:split]]
    train_data = [precomputed[i] for i in perm[split:]]

    # Compute baseline initial metrics
    init_pis_train = torch.stack([b[3] for b in train_data])
    exp_pis_train  = torch.stack([b[5] for b in train_data])
    init_kl_train  = F.kl_div(torch.log(init_pis_train + 1e-8), exp_pis_train, reduction='batchmean').item()
    init_vs_train  = torch.stack([b[4] for b in train_data]).view(-1)
    exp_vs_train   = torch.stack([b[6] for b in train_data]).view(-1)
    init_val_err_train = F.l1_loss(init_vs_train, exp_vs_train, reduction='mean').item()

    init_pis_val = torch.stack([b[3] for b in val_data])
    exp_pis_val  = torch.stack([b[5] for b in val_data])
    init_kl_val  = F.kl_div(torch.log(init_pis_val + 1e-8), exp_pis_val, reduction='batchmean').item()
    init_vs_val  = torch.stack([b[4] for b in val_data]).view(-1)
    exp_vs_val   = torch.stack([b[6] for b in val_data]).view(-1)
    init_val_err_val = F.l1_loss(init_vs_val, exp_vs_val, reduction='mean').item()

    # Freeze base network, train only GNN
    model.nnet.eval()
    for p in model.nnet.parameters(): p.requires_grad = False
    model.gnn.train()
    for p in model.gnn.parameters(): p.requires_grad = True
    model.input_fc.train()
    model.output_fc.train()

    optimizer = model.gnn_optimizer
    bs, epochs = args.batch_size, args.epochs

    pi_tr, v_tr, pi_val, v_val = [], [], [], []

    for epoch in range(epochs):
        # Training
        tot_pi, tot_v = 0.0, 0.0
        batches = max(1, len(train_data) // bs)
        for _ in range(batches):
            batch = [train_data[i] for i in np.random.randint(0, len(train_data), bs)]
            feats_list = [b[0] for b in batch]
            edges_list = [b[1] for b in batch]
            roots_l    = [b[2] for b in batch]
            exp_pis    = [b[5] for b in batch]
            v_tgts     = [b[6] for b in batch]

            all_feats = torch.cat(feats_list, dim=0)
            sizes = [f.size(0) for f in feats_list]
            offs = np.cumsum([0] + sizes[:-1])

            if edges_list:
                edge_index = torch.cat([e + offs[i] for i, e in enumerate(edges_list)], dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            roots = torch.tensor([roots_l[i] + offs[i] for i in range(len(roots_l))], dtype=torch.long, device=device)

            small_feats    = model.input_fc(all_feats)
            enhanced_small = model.gnn(small_feats, edge_index)
            enhanced_full  = model.output_fc(enhanced_small)
            enhanced       = enhanced_full.view(-1, channels, h, w)[roots]

            log_pi, v_pred = model.nnet.policy_value_heads(enhanced)
            v_pred = v_pred.squeeze(-1)

            loss_pi = F.kl_div(log_pi, torch.stack(exp_pis), reduction='batchmean')
            loss_v  = F.l1_loss(v_pred, torch.stack(v_tgts), reduction='mean')
            loss    = loss_pi + loss_v

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.input_fc.parameters()) +
                list(model.gnn.parameters()) +
                list(model.output_fc.parameters()), 1.0
            )
            optimizer.step()

            tot_pi += loss_pi.item()
            tot_v  += loss_v.item()

        pi_tr.append(tot_pi / batches)
        v_tr.append(tot_v / batches)

        # Validation
        model.gnn.eval()
        with torch.no_grad():
            feats_list = [b[0] for b in val_data]
            edges_list = [b[1] for b in val_data]
            roots_l    = [b[2] for b in val_data]
            exp_pis    = [b[5] for b in val_data]
            v_tgts     = [b[6] for b in val_data]

            all_feats = torch.cat(feats_list, dim=0)
            sizes = [f.size(0) for f in feats_list]
            offs = np.cumsum([0] + sizes[:-1])

            if edges_list:
                edge_index = torch.cat([e + offs[i] for i, e in enumerate(edges_list)], dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            roots = torch.tensor([roots_l[i] + offs[i] for i in range(len(roots_l))], dtype=torch.long, device=device)

            small_feats    = model.input_fc(all_feats)
            enhanced_small = model.gnn(small_feats, edge_index)
            enhanced_full  = model.output_fc(enhanced_small)
            enhanced       = enhanced_full.view(-1, channels, h, w)[roots]

            log_pi_v, v_val_pred = model.nnet.policy_value_heads(enhanced)
            v_val_pred = v_val_pred.squeeze(-1)

            loss_pi_val = F.kl_div(log_pi_v, torch.stack(exp_pis), reduction='batchmean')
            loss_v_val  = F.l1_loss(v_val_pred, torch.stack(v_tgts), reduction='mean')

        pi_val.append(loss_pi_val.item())
        v_val.append(loss_v_val.item())
        model.gnn.train()

        log.warning(f"[Epoch {epoch+1}/{epochs}] π-tr={pi_tr[-1]:.4f}, v-tr={v_tr[-1]:.4f} | π-val={pi_val[-1]:.4f}, v-val={v_val[-1]:.4f}")

    return pi_tr, v_tr, pi_val, v_val, init_kl_train, init_kl_val, init_val_err_train, init_val_err_val


def plot_loss(curves, ylabel, title, output_path, epochs):
    plt.figure()
    x = np.arange(1, epochs + 1)
    for label, vals in curves.items():
        plt.plot(x, vals, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train & Plot two GNN variants: with and without edges")
    parser.add_argument('--data_folder', '-d', required=True, help="Folder with GNN data (.pt)")
    parser.add_argument('--config', '-c', default=None, help="Path to config.yaml")
    args = parser.parse_args()

    game_dir = os.path.dirname(os.path.abspath(args.data_folder))
    game_name = os.path.basename(game_dir)
    cfg_path = args.config or os.path.join(game_name, 'config.yaml')
    config = load_config(cfg_path)
    cfg_args = config_to_args(config)

    # Force GNN wrapper
    cfg_args.gnn_train   = True
    cfg_args.gnn_predict = True
    cfg_args.game        = game_name

    GameClass, _ = get_game(game_name, use_gnn=False)
    _, GNNClass = get_game(game_name, use_gnn=True)
    game = create_game_instance(GameClass, cfg_args)

    log.warning("Training from scratch (no checkpoint will be loaded)")
    with_model = GNNClass(game, cfg_args)
    no_model   = GNNClass(game, cfg_args)

    all_data = load_all_gnn_data(args.data_folder)
    data_no  = strip_edges(all_data)

    log.warning("== Training WITH edges ==")
    pi_tr_w, v_tr_w, pi_val_w, v_val_w, init_kl_tr, init_kl_val, init_val_err_tr, init_val_err_val = \
        train_gnn_model(with_model, all_data, cfg_args)

    log.warning("== Training WITHOUT edges ==")
    pi_tr_n, v_tr_n, pi_val_n, v_val_n, init_kl_tr2, init_kl_val2, init_val_err_tr2, init_val_err_val2 = \
        train_gnn_model(no_model, data_no, cfg_args)

    # Save checkpoints
    ckpt_folder, _ = get_checkpoint_path(game_name, 'best', use_gnn=True, base_path=cfg_args.checkpoint_path)
    os.makedirs(ckpt_folder, exist_ok=True)
    with_model.save_checkpoint(ckpt_folder, 'final_gnn_with_edges.pth.tar')
    no_model.save_checkpoint(ckpt_folder,   'final_gnn_no_edges.pth.tar')

    plots_dir = os.path.join('plots', game_name)
    os.makedirs(plots_dir, exist_ok=True)
    epochs = cfg_args.epochs

    variants = [
        ('with_edge_index',    (pi_tr_w, v_tr_w, pi_val_w, v_val_w, init_kl_tr, init_kl_val, init_val_err_tr, init_val_err_val)),
        ('without_edge_index', (pi_tr_n, v_tr_n, pi_val_n, v_val_n, init_kl_tr2, init_kl_val2, init_val_err_tr2, init_val_err_val2))
    ]

    # Plot losses
    for label, (pi_tr, v_tr, pi_val, v_val, _, _, _, _) in variants:
        curves = {
            'π Train Loss': pi_tr,
            'v Train Loss': v_tr,
            'π Valid Loss': pi_val,
            'v Valid Loss': v_val
        }
        plot_loss(curves, 'Loss', f"{game_name} {label} Loss", os.path.join(plots_dir, f"{label}_loss.png"), epochs)

    # Plot KL divergence comparison
    for label, (pi_tr, _, pi_val, _, init_kl_t, init_kl_v, _, _) in variants:
        kl_curves = {
            'Initial π Train': [init_kl_t] * epochs,
            'Predicted π Train': pi_tr,
            'Initial π Valid': [init_kl_v] * epochs,
            'Predicted π Valid': pi_val
        }
        plot_loss(kl_curves, 'KL Divergence', f"{game_name} {label} KL Comparison", os.path.join(plots_dir, f"{label}_kl_comparison.png"), epochs)

    # Plot Value error comparison
    for label, (_, v_tr, _, v_val, _, _, init_val_e_t, init_val_e_v) in variants:
        val_curves = {
            'Initial v Train': [init_val_e_t] * epochs,
            'Predicted v Train': v_tr,
            'Initial v Valid': [init_val_e_v] * epochs,
            'Predicted v Valid': v_val
        }
        plot_loss(val_curves, 'Mean Absolute Error', f"{game_name} {label} Value Comparison", os.path.join(plots_dir, f"{label}_value_comparison.png"), epochs)


if __name__ == '__main__':
    main()
