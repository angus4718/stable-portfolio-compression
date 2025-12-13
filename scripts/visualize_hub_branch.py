"""Visualize the `hub-and-branch` basis selection on an MST.

This script builds a synthetic MST from random 2D points, computes hub
centrality, runs `select_hub_branch`, and draws the MST with hubs and
branch representatives highlighted. It can also animate the selection steps.

Usage:
    python scripts/visualize_hub_branch.py --n 100 --k 20 --alpha 0.5 --seed 4718 --animate
    python scripts/visualize_hub_branch.py --n 100 --k 20 --alpha 0.5 --seed 4718 --out outputs/visualizations/hub_branch_demo_sequence.png --save-sequence
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import math
import random
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# expose project root for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spc.portfolio_construction import BasisSelector
from spc.graph import TreeUtils


def build_mst_from_points(points: Dict[int, Tuple[float, float]]):
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


def compute_hubs(
    selector: BasisSelector,
    k: int,
    h: Optional[int],
    alpha: float,
    weights: Optional[Dict[int, float]] = None,
    weight_gamma: float = 0.0,
):
    # replicate the hub scoring logic from BasisSelector.select_hub_branch
    if h is None:
        h = max(1, min(k - 1, int(round(0.2 * k))))

    centrality = {
        u: alpha * float(selector._deg.get(u, 0))
        + (1.0 - alpha) * float(selector._betw.get(u, 0))
        for u in selector.nodes
    }
    norm_centrality = selector._normalize_dict(centrality)
    norm_weights = selector._normalize_dict(weights or {})
    score = {
        u: (1.0 - weight_gamma) * norm_centrality[u] + weight_gamma * norm_weights[u]
        for u in selector.nodes
    }
    hubs = sorted(selector.nodes, key=lambda u: score[u], reverse=True)[:h]
    return hubs, score


def select_hub_branch_impl(
    selector: BasisSelector,
    k: int,
    h: Optional[int] = None,
    alpha: float = 0.5,
    weights: Optional[Dict[int, float]] = None,
    weight_gamma: float = 0.0,
    rep_alpha: float = 0.5,
    stickiness: float = 0.0,
    prev_basis: Optional[List[int]] = None,
) -> List[int]:
    """Local copy of `BasisSelector.select_hub_branch` for standalone visualization.

    Returns a list of selected basis node ids (length <= k).
    """
    nodes = selector.nodes
    n = len(nodes)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(nodes)

    if h is None:
        h = max(1, min(k - 1, int(round(0.2 * k))))

    # centrality mix
    centrality = {
        u: alpha * float(selector._deg.get(u, 0))
        + (1.0 - alpha) * float(selector._betw.get(u, 0))
        for u in nodes
    }
    norm_centrality = selector._normalize_dict(centrality)
    norm_weights = selector._normalize_dict(weights or {})
    score = {
        u: (1.0 - weight_gamma) * norm_centrality[u] + weight_gamma * norm_weights[u]
        for u in nodes
    }

    # previous-basis pre-seed + stickiness bonus
    prev_list: List[int] = []
    if prev_basis:
        seen = set()
        for p in prev_basis:
            if p in nodes and p not in seen:
                prev_list.append(p)
                seen.add(p)
        prev_list = prev_list[:k]

    bonus = (
        float(stickiness) * (max(score.values()) if score else 0.0)
        if stickiness and prev_list
        else 0.0
    )
    if bonus and prev_list:
        for p in prev_list:
            if p in score:
                score[p] = score.get(p, 0.0) + bonus

    hubs = sorted(nodes, key=lambda u: score[u], reverse=True)[:h]

    B: List[int] = []
    for hub in hubs:
        if hub not in B:
            B.append(hub)
        if len(B) >= k:
            return B[:k]

    # remove hubs and compute components (branches)
    comps = TreeUtils.remove_nodes_and_components(selector.adj, set(hubs))

    def rep_for_comp(comp: List[int]):
        best_u = None
        best_s = -float("inf")
        for u in comp:
            # distance to nearest hub
            d_to_h = min(selector._dist_uv(u, hh) for hh in hubs)
            s = rep_alpha * float(d_to_h) + (1.0 - rep_alpha) * float(
                norm_weights.get(u, 0.0)
            )
            if s > best_s:
                best_s = s
                best_u = u
        return (best_s, best_u) if best_u is not None else None

    reps = [rep_for_comp(c) for c in comps]
    reps = [r for r in reps if r is not None]
    reps.sort(key=lambda x: x[0], reverse=True)

    for s_u in reps:
        s, u = s_u
        if u not in B:
            B.append(u)
        if len(B) >= k:
            break

    # fallback: fill by highest score if still short
    if len(B) < k:
        for u in sorted(nodes, key=lambda uu: score.get(uu, 0.0), reverse=True):
            if u not in B:
                B.append(u)
            if len(B) >= k:
                break

    return B[:k]


def assign_nodes_to_hub(selector: BasisSelector, nodes: List[int], hubs: List[int]):
    mapping = {}
    for u in nodes:
        best = None
        bestd = float("inf")
        for h in hubs:
            d = selector._dist_uv(u, h)
            if d < bestd:
                bestd = d
                best = h
        mapping[u] = best
    return mapping


def plot_hub_branch(
    points,
    adj,
    hubs,
    basis_selected,
    hub_map,
    out_path: Path,
    animate: bool = False,
    save_sequence: bool = False,
):
    nodes = list(points.keys())
    xs = [points[u][0] for u in nodes]
    ys = [points[u][1] for u in nodes]

    def draw_frame(ax_left, ax_right, cur_hubs, cur_selected):
        ax_left.clear()
        # draw MST
        for u in adj:
            for v, d in adj[u]:
                if u < v:
                    ax_left.plot(
                        [points[u][0], points[v][0]],
                        [points[u][1], points[v][1]],
                        color="#cccccc",
                        zorder=1,
                    )

        # draw nodes
        for u in nodes:
            if u in cur_selected:
                if u in cur_hubs:
                    ax_left.scatter(
                        points[u][0],
                        points[u][1],
                        s=220,
                        color="#2b9348",
                        edgecolors="#111111",
                        zorder=4,
                    )
                else:
                    ax_left.scatter(
                        points[u][0],
                        points[u][1],
                        s=160,
                        color="#f9844a",
                        edgecolors="#111111",
                        zorder=4,
                    )
            else:
                ax_left.scatter(
                    points[u][0],
                    points[u][1],
                    s=70,
                    color="#8ecae6",
                    edgecolors="#222222",
                    zorder=3,
                )

        # draw lines to hub for selected nodes
        for u in nodes:
            hub = hub_map.get(u)
            if hub is None:
                continue
            color = "#023047" if (hub in cur_hubs) else "#666666"
            lw = 1.2 if u in cur_selected else 0.6
            ax_left.plot(
                [points[u][0], points[hub][0]],
                [points[u][1], points[hub][1]],
                color=color,
                linestyle=("-" if u in cur_selected else "--"),
                linewidth=lw,
                alpha=0.6,
                zorder=2,
            )

        ax_left.set_title("Hubs (green) and Representatives (orange)")
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        # right panel: simple counts per hub
        ax_right.clear()
        hub_counts = {h: sum(1 for u in nodes if hub_map.get(u) == h) for h in hubs}
        hub_labels = [str(h) for h in hubs]
        counts = [hub_counts[h] for h in hubs]
        ax_right.bar(hub_labels, counts, color="#4cc9f0")
        ax_right.set_title("Nodes per hub (component sizes)")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if animate:
        from matplotlib import animation

        # frames: first show hubs, then add representatives one-by-one
        selections = []
        # initial: only hubs visible
        selections.append((hubs, list(hubs)))
        reps = [b for b in basis_selected if b not in hubs]
        cur = list(hubs)
        for r in reps:
            cur = cur + [r]
            selections.append((hubs, list(cur)))

        def make_frame(i):
            cur_hubs, cur_sel = selections[i]
            draw_frame(ax[0], ax[1], cur_hubs, cur_sel)
            plt.suptitle("Hub-and-Branch Selection")
            return (fig,)

        anim = animation.FuncAnimation(
            fig, make_frame, frames=len(selections), interval=700, blit=False
        )
        try:
            writer = animation.PillowWriter(fps=1)
            anim.save(str(out_path), writer=writer)
            print(f"Saved animated GIF to: {out_path}")
        except Exception:
            try:
                writer = animation.FFMpegWriter(fps=1)
                fname = out_path.with_suffix(".mp4")
                anim.save(str(fname), writer=writer)
                print(f"Saved animation to: {fname}")
            except Exception as e:
                print(f"Animation failed: {e}. Saving static final frame instead.")
                draw_frame(ax[0], ax[1], hubs, basis_selected)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(out_path)
                print(f"Saved static visualization to: {out_path}")
        # Optionally save per-step PNG frames
        if save_sequence:
            base = out_path.with_suffix("")
            for i, sel in enumerate(selections):
                cur_sel = sel
                render_hubs = cur_sel[0] if isinstance(cur_sel, tuple) else cur_sel
                draw_frame(
                    ax[0],
                    ax[1],
                    render_hubs,
                    cur_sel[1] if isinstance(cur_sel, tuple) else cur_sel,
                )
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fname = base.with_name(f"{base.name}_{i+1:04d}.png")
                fig.savefig(fname)
            print(
                f"Saved sequence frames to: {out_path.parent} as {base.name}_XXXX.png"
            )
    else:
        draw_frame(ax[0], ax[1], hubs, basis_selected)
        plt.suptitle("Hub-and-Branch Selection")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_path)
        print(f"Saved visualization to: {out_path}")
        # If save_sequence requested, write per-step PNG frames for the
        # static (non-animated) mode as well.
        if save_sequence:
            base = out_path.with_suffix("")
            selections = []
            selections.append((hubs, list(hubs)))
            reps = [b for b in basis_selected if b not in hubs]
            cur = list(hubs)
            for r in reps:
                cur = cur + [r]
                selections.append((hubs, list(cur)))
            for i, sel in enumerate(selections):
                cur_hubs, cur_sel = sel
                draw_frame(ax[0], ax[1], cur_hubs, cur_sel)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fname = base.with_name(f"{base.name}_{i+1:04d}.png")
                fig.savefig(fname)
            print(
                f"Saved sequence frames to: {out_path.parent} as {base.name}_XXXX.png"
            )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--h", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--weight-gamma", type=float, default=0.0)
    p.add_argument("--rep-alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--out", type=str, default="outputs/visualizations/hub_branch_demo.png"
    )
    p.add_argument("--animate", action="store_true")
    p.add_argument(
        "--save-sequence",
        action="store_true",
        help="Save a sequence of PNG frames for each selection step named with a zero-padded suffix (e.g. _0001.png)",
    )
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    points = {i: (random.random(), random.random()) for i in range(args.n)}
    adj = build_mst_from_points(points)

    selector = BasisSelector(adj)

    # simple weight map (distance to center)
    weights = {u: math.hypot(points[u][0] - 0.5, points[u][1] - 0.5) for u in points}

    hubs, score = compute_hubs(
        selector, args.k, args.h, args.alpha, weights, args.weight_gamma
    )

    basis_selected = select_hub_branch_impl(
        selector,
        args.k,
        h=args.h,
        alpha=args.alpha,
        weights=weights,
        weight_gamma=args.weight_gamma,
        rep_alpha=args.rep_alpha,
    )

    hub_map = assign_nodes_to_hub(selector, list(points.keys()), hubs)

    out_path = Path(args.out)
    plot_hub_branch(
        points,
        adj,
        hubs,
        basis_selected,
        hub_map,
        out_path,
        animate=args.animate,
        save_sequence=args.save_sequence,
    )


if __name__ == "__main__":
    main()
