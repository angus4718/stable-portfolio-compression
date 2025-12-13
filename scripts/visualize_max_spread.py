"""Visualize how `select_max_spread_weighted` picks basis nodes on an MST.

This script builds a synthetic Euclidean MST from random 2D points,
runs the selector, and plots the tree with nodes colored by selection order.

Usage:
    python scripts/visualize_max_spread.py --n 100 --k 10 --alpha 0.5 --seed 4718 --annotate-nearest --animate
    python scripts/visualize_max_spread.py --n 100 --k 10 --alpha 0.5 --seed 4718 --out outputs/visualizations/max_spread_demo_sequence.png --save-sequence
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Make sure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spc.portfolio_construction import BasisSelector


def build_mst_from_points(points: Dict[int, Tuple[float, float]]):
    """Build MST edges (Kruskal) from points.

    Returns adjacency dict: node -> list[(nbr, dist)].
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


def selection_trace(
    selector: BasisSelector,
    k: int,
    weights: Dict[int, float],
    alpha: float,
    stickiness: float = 0.0,
    prev_basis=None,
):
    """Run the selector using the latest max-spread implementation and capture per-iteration state.

    This function embeds a local copy of the max-spread algorithm (the
    algorithm used by `BasisSelector.select_max_spread`) so the visualization
    is self-contained and mirrors the selector's behavior exactly.

    Returns:
        (selection_order, trace)
    """

    def select_max_spread_impl(
        selector: BasisSelector,
        k: int,
        weights: Dict[int, float],
        alpha: float = 0.5,
        stickiness: float = 0.0,
        prev_basis=None,
    ) -> List[int]:
        nodes = selector.nodes
        n = len(nodes)
        if k <= 0 or n == 0:
            return []
        if k >= n:
            return list(nodes)

        normalized_weights = selector._normalize_dict(weights or {})
        avg_dist = selector._avg_dist()

        init_score = {
            u: alpha * avg_dist[u] + (1 - alpha) * normalized_weights[u] for u in nodes
        }

        # prepare previous-basis pre-seed and deduplicate
        prev_list: List[int] = []
        if prev_basis:
            seen = set()
            for p in prev_basis:
                if p in nodes and p not in seen:
                    prev_list.append(p)
                    seen.add(p)
            prev_list = prev_list[:k]

        bonus = (
            float(stickiness) * (max(init_score.values()) if init_score else 0.0)
            if stickiness and prev_list
            else 0.0
        )

        if prev_list:
            B = list(prev_list)
        else:
            # seed with highest initial score
            b1 = max(init_score, key=init_score.get)
            B = [b1]

        nearest_dist = {v: selector._dist_uv(v, B[0]) for v in nodes}

        while len(B) < k:
            candidate_scores: Dict[int, float] = {}
            for v in nodes:
                if v in B:
                    continue
                base = alpha * nearest_dist[v] + (1 - alpha) * normalized_weights[v]
                if bonus and prev_basis and v in prev_basis:
                    base += bonus
                candidate_scores[v] = base

            if not candidate_scores:
                break

            v_star = max(candidate_scores, key=candidate_scores.get)
            B.append(v_star)
            for v in nodes:
                dvb = selector._dist_uv(v, v_star)
                if dvb < nearest_dist[v]:
                    nearest_dist[v] = dvb

        return B

    # Use the local implementation to get the final selection order, then
    # reconstruct the per-iteration trace (candidate scores and nearest-dist
    # snapshots) so the visualization can animate or display the stepwise
    # behavior.
    nodes = selector.nodes
    normalized_weights = selector._normalize_dict(weights or {})
    avg_dist = selector._avg_dist()

    # Build bonus consistent with selection logic
    init_score = {
        u: alpha * avg_dist[u] + (1 - alpha) * normalized_weights[u] for u in nodes
    }
    prev_list = []
    if prev_basis:
        seen = set()
        for p in prev_basis:
            if p in nodes and p not in seen:
                prev_list.append(p)
                seen.add(p)
        prev_list = prev_list[:k]

    bonus = (
        float(stickiness) * (max(init_score.values()) if init_score else 0.0)
        if stickiness and prev_list
        else 0.0
    )

    B = select_max_spread_impl(selector, k, weights, alpha, stickiness, prev_basis)

    if not B:
        return B, []

    # Reconstruct trace by simulating the greedy choices up to k using the
    # same scoring function so we capture candidate states for each step.
    nearest_dist = {v: selector._dist_uv(v, B[0]) for v in nodes}
    trace = []
    current_selection = [B[0]]

    for chosen in B[1:]:
        candidate_scores: Dict[int, float] = {}
        for v in nodes:
            if v in current_selection:
                continue
            base = alpha * nearest_dist[v] + (1 - alpha) * normalized_weights[v]
            if bonus and prev_basis and v in prev_basis:
                base += bonus
            candidate_scores[v] = base

        if not candidate_scores:
            break

        score_v = candidate_scores.get(chosen, None)
        trace.append(
            {
                "chosen": chosen,
                "score": score_v,
                "candidates": candidate_scores.copy(),
                "nearest_dist": nearest_dist.copy(),
            }
        )

        current_selection.append(chosen)
        for v in nodes:
            dvb = selector._dist_uv(v, chosen)
            if dvb < nearest_dist[v]:
                nearest_dist[v] = dvb

    return B, trace


def _draw_base_mst(ax, points, adj):
    for u in adj:
        for v, d in adj[u]:
            if u < v:
                ax.plot(
                    [points[u][0], points[v][0]],
                    [points[u][1], points[v][1]],
                    color="#999999",
                    zorder=1,
                )


def _nearest_basis_map(nodes, points, selection):
    # return dict node -> nearest selected basis
    mapping = {}
    if not selection:
        return mapping
    for u in nodes:
        best = None
        bestd = float("inf")
        x1, y1 = points[u]
        for b in selection:
            xb, yb = points[b]
            d = math.hypot(x1 - xb, y1 - yb)
            if d < bestd:
                bestd = d
                best = b
        mapping[u] = best
    return mapping


def plot_mst_selection(
    points,
    adj,
    selection_order: List[int],
    trace,
    out_path: Path,
    annotate_nearest: bool = False,
    animate: bool = False,
    save_sequence: bool = False,
):
    xs = [points[u][0] for u in points]
    ys = [points[u][1] for u in points]
    nodes = list(points.keys())

    # Colors and sizes helper
    def node_color_and_size(u, selection_order):
        cmap = plt.get_cmap("Reds")
        if u in selection_order:
            order = selection_order.index(u) + 1
            max_order = max(1, len(selection_order))
            color = cmap(0.3 + 0.7 * (order / max_order))
            size = 120 + 80 * (1.0 - (order / max_order))
            return color, size
        return "#8ecae6", 80

    # Static plot path
    def render_frame(ax_left, ax_right, current_selection, show_labels=True):
        ax_left.clear()
        _draw_base_mst(ax_left, points, adj)
        node_colors = []
        node_sizes = []
        for u in nodes:
            c, s = node_color_and_size(u, current_selection)
            node_colors.append(c)
            node_sizes.append(s)
        ax_left.scatter(
            xs, ys, s=node_sizes, c=node_colors, edgecolors="#222222", zorder=2
        )
        if show_labels:
            for u in nodes:
                ax_left.text(
                    points[u][0],
                    points[u][1],
                    f"{u}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="#000000",
                    zorder=3,
                )
        ax_left.set_title("MST & Selection Order (numbers are node ids)")
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        if annotate_nearest and current_selection:
            mapping = _nearest_basis_map(nodes, points, current_selection)
            for u, b in mapping.items():
                if b is None:
                    continue
                ax_left.plot(
                    [points[u][0], points[b][0]],
                    [points[u][1], points[b][1]],
                    color="#023047",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=1,
                )

        # Right panel: chosen scores up to current step
        ax_right.clear()
        chosen_steps = (
            [step for step in trace[: len(current_selection) - 1]] if trace else []
        )
        chosen_scores = [step["score"] for step in chosen_steps]
        chosen_nodes = [step["chosen"] for step in chosen_steps]
        if chosen_scores:
            ax_right.bar(
                range(1, len(chosen_scores) + 1), chosen_scores, color="#ef476f"
            )
            for i, (n, s) in enumerate(zip(chosen_nodes, chosen_scores), start=1):
                ax_right.text(
                    i,
                    s + 0.01 * max(1.0, max(chosen_scores)),
                    str(n),
                    ha="center",
                    va="bottom",
                )
        ax_right.set_xlabel("Selection step")
        ax_right.set_ylabel("Chosen node score")
        ax_right.set_title("Per-step chosen scores (node id labelled)")

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if animate:
        # Build frames: start with first selected node
        frames = []
        # include initial (first pick) plus each subsequent pick
        selections = []
        if selection_order:
            for i in range(1, len(selection_order) + 1):
                selections.append(selection_order[:i])
        else:
            selections = [[]]

        from matplotlib import animation

        try:

            def make_frame(i):
                cur_sel = selections[i]
                render_frame(ax[0], ax[1], cur_sel)
                plt.suptitle("Max-Spread-Weighted Selection on Synthetic MST")
                return (fig,)

            anim = animation.FuncAnimation(
                fig, make_frame, frames=len(selections), interval=700, blit=False
            )
            # Save as GIF (or as mp4 fallback)
            try:
                writer = animation.PillowWriter(fps=1)
                anim.save(str(out_path), writer=writer)
                print(f"Saved animated GIF to: {out_path}")
            except Exception:
                # fallback: save as mp4 if pillow unavailable
                writer = animation.FFMpegWriter(fps=1)
                anim.save(str(out_path.with_suffix(".mp4")), writer=writer)
                print(f"Saved animation to: {out_path.with_suffix('.mp4')}")
            # Optionally also save a sequence of PNG frames
            if save_sequence:
                base = out_path.with_suffix("")
                for i, sel in enumerate(selections):
                    render_frame(ax[0], ax[1], sel)
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fname = base.with_name(f"{base.name}_{i+1:04d}.png")
                    fig.savefig(fname)
                print(
                    f"Saved sequence frames to: {out_path.parent} as {base.name}_XXXX.png"
                )
        except Exception as e:
            print(f"Animation failed: {e}")
            # fallback to static render of final selection
            render_frame(ax[0], ax[1], selection_order)
            plt.suptitle("Max-Spread-Weighted Selection on Synthetic MST")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(out_path)
            print(f"Saved fallback static visualization to: {out_path}")
    else:
        render_frame(ax[0], ax[1], selection_order, show_labels=True)
        plt.suptitle("Max-Spread-Weighted Selection on Synthetic MST")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save single final image
        fig.savefig(out_path)
        print(f"Saved visualization to: {out_path}")
        # Optionally save individual frames per selection step
        if save_sequence:
            base = out_path.with_suffix("")
            selections = []
            if selection_order:
                for i in range(1, len(selection_order) + 1):
                    selections.append(selection_order[:i])
            else:
                selections = [[]]
            for i, sel in enumerate(selections):
                render_frame(ax[0], ax[1], sel)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fname = base.with_name(f"{base.name}_{i+1:04d}.png")
                fig.savefig(fname)
            print(
                f"Saved sequence frames to: {out_path.parent} as {base.name}_XXXX.png"
            )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="number of nodes")
    p.add_argument("--k", type=int, default=5, help="number of basis nodes to select")
    p.add_argument(
        "--alpha", type=float, default=0.6, help="alpha tradeoff for selector"
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out", type=str, default="outputs/visualizations/max_spread_demo.png"
    )
    p.add_argument(
        "--annotate-nearest",
        action="store_true",
        help="Draw dashed lines from each node to its nearest selected basis (final selection)",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Save an animated GIF showing the selection steps (requires pillow or ffmpeg)",
    )
    p.add_argument(
        "--save-sequence",
        action="store_true",
        help="Save a sequence of PNG frames for each selection step named with a zero-padded suffix (e.g. _0001.png)",
    )
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate random 2D points in unit square
    points = {i: (random.random(), random.random()) for i in range(args.n)}
    adj = build_mst_from_points(points)

    selector = BasisSelector(adj)

    # create simple weights (higher at center)
    weights = {}
    for u, (x, y) in points.items():
        # weight: distance to center (higher -> more important)
        cx, cy = 0.5, 0.5
        weights[u] = math.hypot(x - cx, y - cy)

    sel_order, trace = selection_trace(selector, args.k, weights, args.alpha)

    out_path = Path(args.out)
    plot_mst_selection(
        points,
        adj,
        sel_order,
        trace,
        out_path,
        annotate_nearest=args.annotate_nearest,
        animate=args.animate,
        save_sequence=args.save_sequence,
    )


if __name__ == "__main__":
    main()
