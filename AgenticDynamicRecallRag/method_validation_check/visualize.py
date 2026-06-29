"""Visualization of how the fusion alpha shifts answer placement.

We answer the readme's two questions for each sample/query:
  1. Does a given alpha push the answer-containing doc out of the top-k window?
  2. If not, how does alpha move the answer's best (smallest) position?

The figure has three panels:
  - Heatmap: a single aggregated row, cols = alpha. Cell color = mean best
    answer position across all hitting samples (1 = top, darker = better).
  - Hit-rate line: fraction of samples keeping the answer in top-k, per alpha.
  - Mean-position line: mean best position among samples that still hit, per alpha.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import Config


def _alpha_key(alpha: float) -> str:
    return f"alpha_{alpha:g}"


def _best_position(positions: List[int]) -> int:
    """Smallest 1-based position, or -1 when the answer is outside top-k."""
    valid = [p for p in positions if p > 0]
    return min(valid) if valid else -1


def build_report(rows: List[dict], config: Config, report_path: str) -> None:
    alphas = config.alphas
    topk = config.topk
    keys = [_alpha_key(a) for a in alphas]

    n = len(rows)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no valid samples collected", ha="center", va="center")
        ax.axis("off")
        fig.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # best position matrix [n_samples, n_alpha]; -1 == pushed out
    mat = np.full((n, len(alphas)), -1, dtype=float)
    for i, row in enumerate(rows):
        for j, key in enumerate(keys):
            mat[i, j] = _best_position(row.get(key, [-1]))

    hit_mask = mat > 0
    hit_rate = hit_mask.mean(axis=0)
    mean_pos = np.array([
        mat[hit_mask[:, j], j].mean() if hit_mask[:, j].any() else np.nan
        for j in range(len(alphas))
    ])

    # --- 新增：计算 Best Alpha Probability ---
    # 排除 -1 (无效位置)，将其视为 np.inf 以便寻找最小值
    mat_valid = np.where(mat > 0, mat, np.inf)
    min_vals = mat_valid.min(axis=1, keepdims=True)

    # 命中最小值的位置标记为 True（注意：全为 inf 的行会导致 min_vals 也为 inf，
    # 但因为限定了 mat > 0，所以全 -1 的行不会产生任何 best key）
    is_best_mask = (mat > 0) & (mat == min_vals)

    # 计算每个 alpha 成为 best key 的概率（占总样本的比例）
    best_prob = is_best_mask.mean(axis=0)
    # ----------------------------------------

    # aggregated single-row heatmap: mean best position per alpha (NaN -> grey)
    disp = mean_pos.reshape(1, -1)

    # 扩展了 figsize 的宽度，以适应第三个子图
    fig = plt.figure(figsize=(max(11, len(alphas) * 1.2), 6))
    # 将底部的图表拆分为 3 列
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1], hspace=0.4, wspace=0.35)

    # --- heatmap ---
    ax0 = fig.add_subplot(gs[0, :])
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="#d9d9d9")  # alphas with no hitting samples
    im = ax0.imshow(disp, aspect="auto", cmap=cmap, vmin=1, vmax=topk, interpolation="nearest")
    ax0.set_title(f"Mean best answer position in top-{topk} per alpha (n={n} samples)")
    ax0.set_xlabel("alpha (bm25 weight)")
    ax0.set_yticks([0])
    ax0.set_yticklabels(["mean position"])
    ax0.set_xticks(range(len(alphas)))
    ax0.set_xticklabels([f"{a:g}" for a in alphas], rotation=45, ha="right")
    for j in range(len(alphas)):
        if not np.isnan(mean_pos[j]):
            ax0.text(j, 0, f"{mean_pos[j]:.2f}", ha="center", va="center",
                     color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.01)
    cbar.set_label("mean position (1 = best)")

    # --- hit rate ---
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(alphas, hit_rate, marker="o", color="#1f77b4")
    ax1.set_title(f"Fraction keeping answer in top-{topk}")
    ax1.set_xlabel("alpha (bm25 weight)")
    ax1.set_ylabel("hit rate")
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)

    # --- mean position ---
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(alphas, mean_pos, marker="s", color="#d62728")
    ax2.set_title("Mean best position (hits only)")
    ax2.set_xlabel("alpha (bm25 weight)")
    ax2.set_ylabel("mean position")
    ax2.invert_yaxis()  # position 1 (best) at the top
    ax2.grid(True, alpha=0.3)

    # --- 新增图表：best probability ---
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(alphas, best_prob, marker="^", color="#2ca02c")
    ax3.set_title("Prob of being the best alpha")
    ax3.set_xlabel("alpha (bm25 weight)")
    ax3.set_ylabel("probability")
    ax3.set_ylim(0, 1.02)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"Effect of fusion alpha on answer placement (n={n} valid samples)", y=0.995)
    fig.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_stats(rows: List[dict], config: Config) -> Dict[str, dict]:
    """Per-alpha aggregate stats, handy for logging alongside the figure."""
    keys = [(a, _alpha_key(a)) for a in config.alphas]
    n_rows = len(rows)

    # --- 新增逻辑：提前统计所有行中最优的 alpha 频次 ---
    best_counts = {k: 0 for _, k in keys}
    for r in rows:
        # 提取当前样本下所有的 best position
        positions = [_best_position(r.get(k, [-1])) for _, k in keys]

        # 过滤掉 -1，只保留有效位置来竞争“最小值”
        valid_positions = [p for p in positions if p > 0]
        if valid_positions:
            min_pos = min(valid_positions)
            for i, (_, k) in enumerate(keys):
                if positions[i] == min_pos:
                    best_counts[k] += 1
    # ---------------------------------------------------

    out: Dict[str, dict] = {}
    for alpha, key in keys:
        best = [_best_position(r.get(key, [-1])) for r in rows]
        hits = [b for b in best if b > 0]
        out[key] = {
            "alpha": alpha,
            "hit_rate": round(len(hits) / n_rows, 4) if n_rows else 0.0,
            "mean_position": round(sum(hits) / len(hits), 4) if hits else None,
            "pushed_out": len(best) - len(hits),
            "best_prob": round(best_counts[key] / n_rows, 4) if n_rows else 0.0, # 新增
        }
    return out