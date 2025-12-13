"""
Visualize the step-by-step algorithm used to compute path betweenness
as implemented in `spc.graph.TreeUtils.path_betweenness_counts`.

This script builds a synthetic Euclidean MST from random 2D points and
captures intermediate algorithm states:
  1) DFS ordering / parent pointers (discovery)
  2) Bottom-up subtree size accumulation (reverse DFS)
  3) Per-node component-size extraction and betweenness calculation

The script saves a zero-padded PNG sequence illustrating each step.
Optionally attempts to write an animated GIF (requires Pillow).

Usage examples:
    python scripts/visualize_path_betweeness.py --n 40 --out outputs/visualizations/path_betweeness_demo.png --save-sequence
    python scripts/visualize_path_betweeness.py --n 40 --out outputs/visualizations/path_betweeness_demo.gif --save-sequence --make-gif

Note: filename extension for `--out` is only used for base name; frames
are saved as `basename_0001.png` etc and GIF is emitted to `--out` when
`--make-gif` is specified and Pillow is available.
"""

from __future__ import annotations
from pathlib import Path
import sys
import argparse
import math
import random
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import deque

# We'll import the TreeUtils implementation for verification if needed
from spc.graph import TreeUtils


def build_mst_from_points(points: Dict[int, Tuple[float, float]]):
    """Kruskal on Euclidean distances returning adjacency dict (undirected).

    Returns: node -> list[(nbr, dist)].
    """
    nodes = list(points.keys())
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            x1, y1 = points[u]
            x2, y2 = points[v]
            d = math.hypot(x1 - x2, y1 - y2)
            edges.append((d, u, v))
    edges.sort(key=lambda x: x[0])

    parent = {u: u for u in nodes}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[rv] = ru
        return True

    mst_edges = []
    for d, u, v in edges:
        if union(u, v):
            mst_edges.append((u, v, d))

    adj: Dict[int, List[Tuple[int, float]]] = {u: [] for u in nodes}
    for u, v, d in mst_edges:
        adj[u].append((v, d))
        adj[v].append((u, d))
    return adj


def compute_dfs_parent_and_events(adj: Dict[Any, List[Tuple[Any, float]]]):
    """Run the DFS used in the algorithm and record discovery events.

    Returns:
      parent: mapping node->parent (root parent None)
      order: list in discovery order (root first)
      events: list of ('discover', node) in the order nodes were appended to order
    """
    nodes = list(adj.keys())
    if not nodes:
        return {}, [], []
    root = nodes[0]
    parent = {root: None}
    order: List[Any] = [root]
    events = [("discover", root)]
    stack = [root]
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v == parent.get(u):
                continue
            parent[v] = u
            stack.append(v)
            order.append(v)
            events.append(("discover", v))
    return parent, order, events


def subtree_accumulation_snapshots(
    adj: Dict[Any, List[Tuple[Any, float]]], parent: Dict[Any, Any], order: List[Any]
):
    """Simulate the reverse-DFS subtree accumulation and capture a snapshot after processing each node.

    Returns list of (processed_node, snapshot_subtree_dict_copy)
    """
    nodes = list(adj.keys())
    # initialize subtree sizes to 1
    subtree = {u: 1 for u in nodes}
    snapshots = []
    # process in reversed discovery order (children before parent)
    for u in reversed(order):
        # accumulate sizes of children
        for v, _ in adj[u]:
            if parent.get(v) == u:
                subtree[u] += subtree[v]
        snapshots.append((u, subtree.copy()))
    return snapshots


def betweenness_snapshots(
    adj: Dict[Any, List[Tuple[Any, float]]],
    parent: Dict[Any, Any],
    subtree: Dict[Any, int],
):
    """Compute betweenness step-by-step, returning snapshots after each node's betweenness is computed.

    Returns list of (node, current_betw_dict_copy, comp_sizes_for_node)
    """
    nodes = list(adj.keys())
    n = len(nodes)
    total = n - 1
    betw: Dict[Any, int] = {}
    snapshots = []
    for u in nodes:
        comp_sizes = [subtree[v] for v, _ in adj[u] if parent.get(v) == u]
        if parent.get(u) is not None:
            comp_sizes.append(total - subtree[u])
        sum_sq = sum(s**2 for s in comp_sizes)
        betw_val = (total * total - sum_sq) // 2
        betw[u] = betw_val
        snapshots.append((u, betw.copy(), comp_sizes))
    return snapshots


def render_and_save_frames(
    points: Dict[int, Tuple[float, float]],
    adj: Dict[int, List[Tuple[int, float]]],
    dfs_events: List[Tuple[str, Any]],
    subtree_snapshots: List[Tuple[Any, Dict[Any, int]]],
    betw_snapshots: List[Tuple[Any, Dict[Any, int], List[int]]],
    out_path: Path,
):
    """Render a sequence of frames illustrating each algorithm step and save PNGs.

    Frames are grouped: discovery frames, subtree accumulation frames, betweenness frames.
    """
    nodes = list(points.keys())
    xs = [points[u][0] for u in nodes]
    ys = [points[u][1] for u in nodes]

    # Prepare output directory and base name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.with_suffix("")

    # Helper to draw edges
    def draw_edges(ax):
        for u in adj:
            for v, _ in adj[u]:
                if u < v:
                    ax.plot(
                        [points[u][0], points[v][0]],
                        [points[u][1], points[v][1]],
                        color="#cccccc",
                        zorder=1,
                    )

    frame_i = 1

    def save_frame(fig):
        nonlocal frame_i
        fname = base.with_name(f"{base.name}_{frame_i:04d}.png")
        fig.tight_layout()
        fig.savefig(fname)
        frame_i += 1

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Stage 1: DFS discovery - progressively reveal discovered nodes
    discovered = set()
    for ev, node in dfs_events:
        if ev == "discover":
            discovered.add(node)
        ax[0].clear()
        ax[1].clear()
        draw_edges(ax[0])
        colors = [("#8ecae6" if u not in discovered else "#ffdd57") for u in nodes]
        sizes = [80 if u not in discovered else 200 for u in nodes]
        ax[0].scatter(xs, ys, s=sizes, c=colors, edgecolors="#222222", zorder=2)
        for u in nodes:
            ax[0].text(
                points[u][0], points[u][1], str(u), fontsize=8, ha="center", va="center"
            )
        ax[0].set_title("DFS discovery (yellow = discovered)")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Right: show discovery order as bar where discovered bars are filled
        order_list = [n for t, n in dfs_events]
        # compute indices and mark discovered
        vals = [1 if u in discovered else 0 for u in order_list]
        ax[1].bar(
            range(len(order_list)),
            vals,
            color=["#ffdd57" if v else "#dddddd" for v in vals],
        )
        ax[1].set_title("Discovery order (left->right)")
        ax[1].set_xlabel("Discovery step")
        save_frame(fig)

    # Stage 2: subtree accumulation snapshots (process reversed order)
    processed = set()
    for proc_u, snapshot in subtree_snapshots:
        processed.add(proc_u)
        ax[0].clear()
        ax[1].clear()
        draw_edges(ax[0])
        # color processed nodes differently
        colors = [("#88c0d0" if u not in processed else "#2b9eb3") for u in nodes]
        sizes = [
            80 if u not in processed else 120 + 20 * snapshot.get(u, 1) for u in nodes
        ]
        ax[0].scatter(xs, ys, s=sizes, c=colors, edgecolors="#222222", zorder=2)
        for u in nodes:
            ax[0].text(
                points[u][0],
                points[u][1],
                f"{u}\n{snapshot.get(u, 1)}",
                fontsize=8,
                ha="center",
                va="center",
            )
        ax[0].set_title(f"Subtree accumulation (processed highlighted)")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Right: bar chart of current subtree sizes
        ysizes = [snapshot.get(u, 1) for u in nodes]
        ax[1].bar(range(len(nodes)), ysizes, color="#2b9eb3")
        ax[1].set_title("Current subtree sizes")
        ax[1].set_xlabel("Node (index order)")
        save_frame(fig)

    # Stage 3: betweenness computation snapshots
    final_betw = {}
    for u, betw_dict, comp_sizes in betw_snapshots:
        final_betw.update(betw_dict)
        ax[0].clear()
        ax[1].clear()
        draw_edges(ax[0])
        # highlight the node being computed
        colors = [("#bbbbbb" if v != u else "#ff6b6b") for v in nodes]
        sizes = [80 + 3 * final_betw.get(v, 0) for v in nodes]
        ax[0].scatter(xs, ys, s=sizes, c=colors, edgecolors="#222222", zorder=2)
        for v in nodes:
            label = str(v)
            if v == u:
                label = f"{v}\ncomp:{comp_sizes}\nbetw:{final_betw.get(v,0)}"
            else:
                if final_betw.get(v) is not None:
                    label = f"{v}\n{final_betw.get(v)}"
            ax[0].text(
                points[v][0], points[v][1], label, fontsize=7, ha="center", va="center"
            )
        ax[0].set_title(f"Betweenness calc: computing node {u}")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Right: bar chart of current betweenness
        betw_vals = [final_betw.get(v, 0) for v in nodes]
        ax[1].bar(range(len(nodes)), betw_vals, color="#ff6b6b")
        ax[1].set_title("Accumulated betweenness")
        save_frame(fig)

    plt.close(fig)
    print(f"Saved sequence frames to: {out_path.parent} as {base.name}_XXXX.png")

    # Try to make a GIF if Pillow is available
    try:
        from PIL import Image

        # collect frames
        frames = []
        for i in range(1, frame_i):
            fname = base.with_name(f"{base.name}_{i:04d}.png")
            frames.append(Image.open(fname))
        gif_path = out_path.with_suffix(".gif")
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=600, loop=0
        )
        print(f"Saved GIF animation to: {gif_path}")
    except Exception:
        # Pillow not available or GIF creation failed; that's fine
        pass


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40, help="number of nodes")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out", type=str, default="outputs/visualizations/path_betweeness_demo.png"
    )
    p.add_argument(
        "--save-sequence",
        action="store_true",
        help="save a sequence of PNG frames illustrating algorithm steps",
    )
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    points = {i: (random.random(), random.random()) for i in range(args.n)}
    adj = build_mst_from_points(points)

    # Compute algorithm traces
    parent, order, dfs_events = compute_dfs_parent_and_events(adj)
    subtree_snaps = subtree_accumulation_snapshots(adj, parent, order)

    # The subtree snapshot list contains (u, snapshot) in reversed-order processing sequence
    # For betweenness snapshots we need the final subtree mapping; take last snapshot if exists
    final_subtree = subtree_snaps[-1][1] if subtree_snaps else {u: 1 for u in points}
    betw_snaps = betweenness_snapshots(adj, parent, final_subtree)

    out_path = Path(args.out)

    if args.save_sequence:
        render_and_save_frames(
            points, adj, dfs_events, subtree_snaps, betw_snaps, out_path
        )
    else:
        # fallback: just save a static visualization of final betweenness
        betw = TreeUtils.path_betweenness_counts(adj)
        # reuse the simpler plot from other script
        xs = [points[u][0] for u in points]
        ys = [points[u][1] for u in points]
        nodes = list(points.keys())
        vals = np.array([betw.get(u, 0) for u in nodes], dtype=float)
        if vals.max() > 0:
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        else:
            norm_vals = vals
        cmap = plt.get_cmap("viridis")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for u in adj:
            for v, d in adj[u]:
                if u < v:
                    ax.plot(
                        [points[u][0], points[v][0]],
                        [points[u][1], points[v][1]],
                        color="#cccccc",
                    )
        colors = [cmap(0.2 + 0.8 * norm_vals[i]) for i in range(len(nodes))]
        sizes = [80 + 220 * norm_vals[i] for i in range(len(nodes))]
        ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="#222222")
        for u in nodes:
            ax.text(
                points[u][0],
                points[u][1],
                f"{u}\n{betw.get(u,0)}",
                fontsize=8,
                ha="center",
                va="center",
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        print(f"Saved static betweenness visualization to: {out_path}")


if __name__ == "__main__":
    main()
