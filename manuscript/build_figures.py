"""Generate publication figures for the DFU IEEE manuscript from verified evidence.

Quantitative figures read the authoritative evaluation artifacts; no values are
invented. Diagram figures describe the implemented MobileDFU-Net modules.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

DARK = "#22314A"
MID = "#4E6E9E"
LIGHT = "#A9BDD8"
GREY = "#6B7280"
PALE = "#E8EDF4"
SANS = {"family": "DejaVu Sans"}


def box(ax, xy, w, h, text, fc="white", ec=DARK, lw=0.9, fs=7.0, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.035",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#111111", zorder=3,
            fontweight=("bold" if bold else "normal"), **SANS)


def arrow(ax, p0, p1, style="-|>", ls="-", lw=0.9, color=DARK, rad=0.0,
          shrinkA=0, shrinkB=0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1,
                                 shrinkA=shrinkA, shrinkB=shrinkB))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def load_hist(name):
    with open(Path(__file__).resolve().parent / f"_hist_{name}.json", encoding="utf-8") as f:
        return json.load(f)


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.1, 1.35))
    blank(ax, (0, 7.1), (0, 1.35))
    stages = [
        "Data acquisition\nand lineage audit",
        "Cohort reconcile\n927 / 186 / 2,187",
        "Source-only\ntraining (3 seeds)",
        "Native-target\nevaluation",
        "Uncertainty and\nfailure analysis",
        "Ablation and\nrecovery studies",
    ]
    w, gap = 1.03, 0.155
    x = 0.06
    for i, s in enumerate(stages):
        box(ax, (x, 0.24), w, 0.80, s, fc="white", fs=5.8)
        if i < len(stages) - 1:
            arrow(ax, (x + w + 0.008, 0.64), (x + w + gap - 0.008, 0.64))
        ax.text(x + w / 2, 1.18, f"({chr(97 + i)})", ha="center", va="center",
                fontsize=6.8, color=GREY, **SANS)
        x += w + gap
    fig.savefig(OUT / "fig1_workflow.png")
    plt.close(fig)


def fig_architecture():
    """Architecture diagram, traced from src/models/mobile_dfu_net/.

    Encoder: stem 24 ch; S1..S4 = 24/32/64/128 ch, 2/3/4/3 inverted-residual
    blocks, strides 2, at 256/128/64/32 squared. RFB: 160 ch at 32 squared.
    Decoder: Up4=96 at 64 squared, Up3=48 at 128 squared, Up2=24 at 256 squared,
    Up1=24 at 512 squared, then a stem-fusion block D1=16 at 512 squared.
    Heads both consume D1; the segmentation head is boundary-gated.
    """
    fig, ax = plt.subplots(figsize=(7.1, 3.95))
    blank(ax, (0, 7.1), (0, 3.95))
    bw, bh = 1.42, 0.40
    ex, dx, hx = 0.08, 3.65, 5.70
    ys = [3.25, 2.65, 2.05, 1.45, 0.85]
    enc = [("DCS-Stem", "24 ch $\\cdot$ 512$^2$"),
           ("Encoder S1 (IR$\\times$2)", "24 ch $\\cdot$ 256$^2$"),
           ("Encoder S2 (IR$\\times$3)", "32 ch $\\cdot$ 128$^2$"),
           ("Encoder S3 (IR$\\times$4)", "64 ch $\\cdot$ 64$^2$"),
           ("Encoder S4 (IR$\\times$3)", "128 ch $\\cdot$ 32$^2$")]
    dec = [("D1 stem fusion", "16 ch $\\cdot$ 512$^2$"),
           ("Up1 + CAG", "24 ch $\\cdot$ 512$^2$"),
           ("Up2 + CAG", "24 ch $\\cdot$ 256$^2$"),
           ("Up3 + CAG", "48 ch $\\cdot$ 128$^2$"),
           ("Up4 + CAG", "96 ch $\\cdot$ 64$^2$")]
    ax.text(ex + bw / 2, 3.79, "Encoder", ha="center", va="center",
            fontsize=6.6, color=GREY, **SANS)
    ax.text(dx + bw / 2, 3.79, "Decoder", ha="center", va="center",
            fontsize=6.6, color=GREY, **SANS)
    for (t1, t2), y in zip(enc, ys):
        box(ax, (ex, y), bw, bh, f"{t1}\n{t2}", fs=6.3)
    for i in range(4):
        arrow(ax, (ex + bw / 2, ys[i]), (ex + bw / 2, ys[i + 1] + bh))
    for (t1, t2), y in zip(dec, ys):
        box(ax, (dx, y), bw, bh, f"{t1}\n{t2}", fs=6.3)
    for i in range(4, 0, -1):
        arrow(ax, (dx + bw / 2, ys[i] + bh), (dx + bw / 2, ys[i - 1]))
    # encoder-to-decoder skip connections (dashed)
    for i, lab in zip(range(1, 5), ["s1 $\\cdot$ 24 ch", "s2 $\\cdot$ 32 ch",
                                    "s3 $\\cdot$ 64 ch", "s4 $\\cdot$ 128 ch"]):
        y = ys[i] + bh / 2
        arrow(ax, (ex + bw, y), (dx, y), ls=(0, (3, 2)), lw=0.8, color=GREY)
        ax.text((ex + bw + dx) / 2, y + 0.03, lab, fontsize=5.5, color=GREY,
                ha="center", va="bottom", **SANS)
    y = ys[0] + bh / 2
    arrow(ax, (ex + bw, y), (dx, y), ls=(0, (3, 2)), lw=0.8, color=GREY)
    ax.text((ex + bw + dx) / 2, y + 0.03, "f0 $\\cdot$ 24 ch", fontsize=5.5,
            color=GREY, ha="center", va="bottom", **SANS)
    # RFB bottleneck and the deepest-resolution flow
    box(ax, (1.80, 0.14), 1.55, 0.40, "RFB bottleneck\n160 ch $\\cdot$ 32$^2$", fc=PALE, fs=6.3)
    cx = ex + bw / 2
    ax.plot([cx, cx], [ys[-1], 0.34], color=DARK, lw=0.9)
    arrow(ax, (cx, 0.34), (1.80, 0.34))
    ax.plot([3.35, dx + bw / 2], [0.34, 0.34], color=DARK, lw=0.9)
    arrow(ax, (dx + bw / 2, 0.34), (dx + bw / 2, ys[-1]))
    # heads: both consume D1 (stem fusion output)
    box(ax, (hx, 3.12), 1.38, 0.42, "Boundary head\n1 ch $\\cdot$ $\\sigma$", fc=PALE, fs=6.2)
    box(ax, (hx, 2.36), 1.38, 0.42, "Seg. head\n1 ch, gated", fc=PALE, fs=6.2)
    ax.plot([dx + bw, 5.40], [ys[0] + bh / 2, ys[0] + bh / 2], color=DARK, lw=0.9)
    ax.plot([5.40, 5.40], [ys[0] + bh / 2, 2.57], color=DARK, lw=0.9)
    ax.plot([5.40], [ys[0] + bh / 2], marker="o", ms=2.6, color=DARK, zorder=4)
    arrow(ax, (5.40, 3.33), (hx, 3.33))
    arrow(ax, (5.40, 2.57), (hx, 2.57))
    arrow(ax, (6.39, 3.12), (6.39, 2.78))
    ax.text(6.46, 2.95, "gate", fontsize=5.3, color=GREY, ha="left", va="center", **SANS)
    ax.text(0.08, 0.02,
            "IR$\\times$n: n inverted-residual blocks;  CAG: channel-attention-gated skip;  "
            "dashed: skip connection;  $\\sigma$: sigmoid",
            fontsize=5.0, color=GREY, ha="left", va="bottom", **SANS)
    fig.savefig(OUT / "fig2_architecture.png")
    plt.close(fig)


def fig_dcs_stem():
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    blank(ax, (0, 3.45), (0, 2.45))
    box(ax, (1.16, 2.13), 1.13, 0.24, "Raw RGB input\nfloat32 in [0, 1]", fc=PALE, fs=6.3)
    box(ax, (0.06, 1.54), 1.52, 0.34, "BT.601 luminance\n+ Sobel magnitude", fs=6.2)
    box(ax, (0.06, 0.98), 1.52, 0.34, "3$\\times$3 convolution\n$\\rightarrow$ 8 channels", fs=6.2)
    box(ax, (1.87, 1.54), 1.52, 0.34, "ImageNet normalization\n(chromatic branch only)", fs=6.2)
    box(ax, (1.87, 0.98), 1.52, 0.34, "depthwise + pointwise\n$\\rightarrow$ 16 channels", fs=6.2)
    box(ax, (0.06, 0.48), 1.52, 0.28, "structural features $F_s$\n(8 ch) + sigmoid gate $g$", fs=5.8)
    box(ax, (1.87, 0.48), 1.52, 0.28, "gated colour features\n$F_c \\odot g$", fs=6.0)
    box(ax, (0.72, 0.05), 2.00, 0.26, "concatenate $[F_s, F_c\\odot g] \\rightarrow F_0$", fc=PALE, fs=6.0)
    arrow(ax, (1.72, 2.13), (0.82, 1.88))
    arrow(ax, (1.72, 2.13), (2.63, 1.88))
    arrow(ax, (0.82, 1.54), (0.82, 1.32))
    arrow(ax, (2.63, 1.54), (2.63, 1.32))
    arrow(ax, (0.82, 0.98), (0.82, 0.76))
    arrow(ax, (2.63, 0.98), (2.63, 0.76))
    arrow(ax, (1.58, 0.62), (1.87, 0.62))
    ax.text(1.725, 0.65, "$g$", fontsize=5.6, color=GREY, ha="center", **SANS)
    arrow(ax, (0.82, 0.48), (1.30, 0.31))
    arrow(ax, (2.63, 0.48), (2.15, 0.31))
    ax.text(1.60, 1.99, "split", fontsize=5.5, color=GREY, ha="center", **SANS)
    fig.savefig(OUT / "fig3_dcs_stem.png")
    plt.close(fig)


def fig_boundary():
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    blank(ax, (0, 3.45), (0, 2.65))
    box(ax, (0.78, 2.33), 1.90, 0.25, "binary ground-truth mask", fc=PALE, fs=6.2)
    box(ax, (0.78, 1.92), 1.90, 0.25, "3$\\times$3 erosion $\\rightarrow$ contour", fs=6.0)
    box(ax, (0.78, 1.51), 1.90, 0.25, "Euclidean distance transform\n(tolerance $\\tau=2$, cap 20)", fs=5.8)
    box(ax, (0.18, 1.06), 1.48, 0.28, "$\\phi_{fp}$: exterior field", fs=6.0)
    box(ax, (1.80, 1.06), 1.48, 0.28, "$\\phi_{fn}$: interior field", fs=6.0)
    box(ax, (0.62, 0.63), 2.22, 0.25, "$\\phi_{signed} = \\phi_{fp} - \\phi_{fn}$", fc=PALE, fs=6.4)
    box(ax, (0.25, 0.12), 2.96, 0.34,
        "$L_{boundary} = \\frac{1}{|\\Omega|}\\sum_{x} \\phi_{signed}(x)\\,\\sigma(z(x))$", fs=6.6)
    arrow(ax, (1.73, 2.33), (1.73, 2.17))
    arrow(ax, (1.73, 1.92), (1.73, 1.76))
    arrow(ax, (1.40, 1.51), (0.92, 1.34))
    arrow(ax, (2.06, 1.51), (2.54, 1.34))
    arrow(ax, (0.92, 1.06), (1.30, 0.88))
    arrow(ax, (2.54, 1.06), (2.16, 0.88))
    arrow(ax, (1.73, 0.63), (1.73, 0.46))
    fig.savefig(OUT / "fig4_boundary.png")
    plt.close(fig)


def fig_uncertainty():
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    blank(ax, (0, 3.45), (0, 2.55))
    box(ax, (0.95, 2.24), 1.55, 0.24, "input image $x$", fc=PALE, fs=6.4)
    box(ax, (0.42, 1.80), 2.61, 0.30,
        "$T=15$ stochastic forward passes\nDropout active; BatchNorm in eval mode", fs=5.9)
    box(ax, (0.42, 1.38), 2.61, 0.28,
        "horizontal-flip TTA within each pass\n$\\rightarrow$ per-pass probabilities $p_t$", fs=5.9)
    box(ax, (0.16, 0.86), 1.48, 0.38,
        "mean probability\n$\\bar{p} = \\frac{1}{T}\\sum_t p_t$", fs=6.1)
    box(ax, (1.81, 0.86), 1.48, 0.38,
        "expected entropy\n$\\frac{1}{T}\\sum_t H[p_t]$", fs=6.1)
    box(ax, (0.16, 0.40), 1.48, 0.36,
        "predictive entropy\n$H[\\bar{p}]$", fs=6.1)
    box(ax, (1.81, 0.40), 1.48, 0.36,
        "mutual information\n$H[\\bar{p}]-\\frac{1}{T}\\sum_t H[p_t]$", fs=5.7)
    box(ax, (0.62, 0.04), 2.21, 0.24, "uncertainty maps for failure analysis", fc=PALE, fs=6.0)
    arrow(ax, (1.72, 2.24), (1.72, 2.10))
    arrow(ax, (1.72, 1.80), (1.72, 1.66))
    arrow(ax, (1.30, 1.38), (0.92, 1.24))
    arrow(ax, (2.15, 1.38), (2.53, 1.24))
    arrow(ax, (0.90, 0.86), (0.90, 0.76))
    arrow(ax, (2.55, 0.86), (2.55, 0.76))
    arrow(ax, (0.90, 0.40), (1.25, 0.28))
    arrow(ax, (2.55, 0.40), (2.20, 0.28))
    fig.savefig(OUT / "fig5_uncertainty.png")
    plt.close(fig)


INTERNAL = [0.8370467168028637, 0.8321766840514316, 0.8325559490470475]
NATIVE = {
    "42": dict(dice=0.24850680047022988, iou=0.16580480733499134, sens=0.20975430544143606),
    "2024": dict(dice=0.23653303035654205, iou=0.1566303185687821, sens=0.19330971754129841),
    "1337": dict(dice=0.2202505067443418, iou=0.14693602212906295, sens=0.17421391932804572),
}


def fig_performance():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1),
                             gridspec_kw={"width_ratios": [1, 1.3, 1.05]})
    order = ["42", "1337", "2024"]
    vals = [INTERNAL[0], INTERNAL[1], INTERNAL[2]]
    x = np.arange(3)
    ax = axes[0]
    ax.bar(x, vals, color=MID, width=0.60, edgecolor=DARK, linewidth=0.5)
    ax.axhline(0.83392645, color=GREY, ls="--", lw=0.8)
    ax.text(2.44, 0.8350, "mean 0.833927", fontsize=5.4, color=GREY, ha="right")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.0007, f"{v:.4f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x, order)
    ax.set_ylim(0.828, 0.840)
    ax.set_yticks([0.828, 0.830, 0.832, 0.834, 0.836, 0.838, 0.840])
    ax.set_ylabel("macro Dice")
    ax.set_xlabel("seed")
    ax.set_title("(a) Internal validation", fontsize=7.5)

    ax = axes[1]
    w = 0.25
    for j, (k, lab, c) in enumerate(zip(["dice", "iou", "sens"],
                                        ["Dice", "IoU", "Sensitivity"],
                                        [MID, LIGHT, "#C9D3E2"])):
        ax.bar(x + (j - 1) * w, [NATIVE[s][k] for s in order], width=w,
               color=c, edgecolor=DARK, linewidth=0.4, label=lab)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 0.32)
    ax.set_ylabel("native-target score")
    ax.set_xlabel("seed")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), handlelength=0.9,
              columnspacing=0.7, fontsize=6.2)
    ax.set_title("(b) Native target (0.5)", fontsize=7.5)

    ax = axes[2]
    for i, s in enumerate(order):
        ax.plot([NATIVE[s]["dice"], INTERNAL[i]], [i, i], color=LIGHT, lw=2.0,
                solid_capstyle="round", zorder=1)
        ax.plot(NATIVE[s]["dice"], i, "o", color=DARK, ms=4.2, zorder=3)
        ax.plot(INTERNAL[i], i, "o", color=MID, ms=4.2, zorder=3)
        ax.text(NATIVE[s]["dice"] - 0.018, i + 0.19, f"{NATIVE[s]['dice']:.3f}",
                fontsize=5.6, color=DARK, ha="right")
        ax.text(INTERNAL[i] + 0.010, i + 0.19, f"{INTERNAL[i]:.3f}",
                fontsize=5.6, color=MID, ha="left")
    ax.set_yticks(range(3), order)
    ax.set_ylim(-0.5, 2.7)
    ax.set_xlim(0.15, 0.90)
    ax.set_xlabel("macro Dice")
    ax.set_title("(c) Domain gap", fontsize=7.5)
    ax.text(0.57, 2.48, "native", fontsize=5.8, color=DARK, ha="center")
    ax.text(0.86, 2.48, "internal", fontsize=5.8, color=MID, ha="center")
    fig.tight_layout(w_pad=0.9)
    fig.savefig(OUT / "fig6_performance.png")
    plt.close(fig)


def fig_training():
    h = load_hist("seed2024")
    ep = [r["epoch"] + 1 for r in h]
    fig, ax1 = plt.subplots(figsize=(3.45, 2.15))
    l1, = ax1.plot(ep, [r["train_loss"] for r in h], color=MID, label="training loss")
    l2, = ax1.plot(ep, [r["val_loss"] for r in h], color=GREY, ls="--", label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 4.2)
    ax2 = ax1.twinx()
    l3, = ax2.plot(ep, [r["val_dice"] for r in h], color=DARK, lw=1.3, label="validation Dice")
    ax2.set_ylabel("validation Dice")
    ax2.set_ylim(0.60, 0.86)
    ax2.spines["top"].set_visible(False)
    best_ep, best = 87, 0.8325559490470475
    ax2.plot([best_ep], [best], marker="*", color=DARK, ms=6.5, zorder=5)
    ax2.annotate(f"best {best:.4f} (epoch {best_ep})", xy=(best_ep, best),
                 xytext=(38, 0.665), fontsize=5.6, color=DARK,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY,
                                 shrinkA=0, shrinkB=2))
    ax1.legend([l1, l2, l3], [x.get_label() for x in (l1, l2, l3)],
               frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.3,
               columnspacing=0.8, fontsize=5.7, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_training.png")
    plt.close(fig)


def fig_ablation():
    base = {r["record_id"]: float(r["dice"]) for r in
            csv.DictReader(open(ROOT / "reports/external_evaluation/seed_42/per_image_metrics.csv"))}
    quart = {r["record_id"]: r["quartile"] for r in
             csv.DictReader(open(ROOT / "experiments/final_research/wound_quartiles_native_target.csv"))}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.15),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    metrics = {
        "Dice": [0.24850680047022988, 0.23981183174387316, 0.20670944422564175],
        "IoU": [0.16580480733499134, 0.15899125854149038, 0.13554602667837193],
        "Sensitivity": [0.20975430544143606, 0.18730254625143242, 0.16487091343275254],
    }
    x = np.arange(3)
    w = 0.26
    for j, (k, v) in enumerate(metrics.items()):
        ax.bar(x + (j - 1) * w, v, width=w, label=k,
               color=[DARK, MID, LIGHT][j], edgecolor=DARK, linewidth=0.4)
    ax.set_xticks(x, ["Base", "E1", "E2"])
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("native-target score")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.03), handlelength=0.9,
              columnspacing=0.7, fontsize=6.2)
    ax.set_title("(a) Overall ablation", fontsize=7.5)
    ax = axes[1]
    e1 = list(csv.DictReader(open(ROOT / "experiments/final_research/ablation_e1_colour/evaluation/per_image_native_target.csv")))
    e2 = list(csv.DictReader(open(ROOT / "experiments/final_research/ablation_e2_boundary/evaluation/per_image_native_target.csv")))
    qs = ["Q1", "Q2", "Q3", "Q4"]
    d1 = [statistics.mean(float(r["dice"]) - base[r["record_id"]] for r in e1 if quart[r["record_id"]] == q) for q in qs]
    d2 = [statistics.mean(float(r["dice"]) - base[r["record_id"]] for r in e2 if quart[r["record_id"]] == q) for q in qs]
    x = np.arange(4)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.bar(x - 0.18, d1, width=0.34, color=MID, label="E1 colour", edgecolor=DARK, linewidth=0.4)
    ax.bar(x + 0.18, d2, width=0.34, color=LIGHT, label="E2 boundary", edgecolor=DARK, linewidth=0.4)
    ax.set_xticks(x, ["Q1\n(≤8,995)", "Q2\n(≤22,237)", "Q3\n(≤42,706)", "Q4\n(≤185,768)"])
    ax.set_ylabel("mean per-image ΔDice")
    ax.set_ylim(-0.065, 0.014)
    ax.set_xlabel("wound-area quartile (pixels)")
    ax.legend(frameon=False, loc="lower left", handlelength=0.9)
    ax.set_title("(b) Effect by wound size", fontsize=7.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig8_ablation.png")
    plt.close(fig)


def fig_recovery():
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    metrics = ["Internal\nDice", "Target\nDice", "Target\nsensitivity"]
    base = [0.8325559490470475, 0.23653303035654205, 0.19330971754129841]
    new = [0.8225076698487804, 0.24356042337571945, 0.21095986974069159]
    x = np.arange(3)
    ax.bar(x - 0.18, base, width=0.34, color=LIGHT, label="pos weight 2.0",
           edgecolor=DARK, linewidth=0.4)
    ax.bar(x + 0.18, new, width=0.34, color=MID, label="pos weight 4.0",
           edgecolor=DARK, linewidth=0.4)
    for xi, b, n in zip(x, base, new):
        ax.text(xi - 0.18, b + 0.012, f"{b:.3f}", ha="center", fontsize=5.6)
        ax.text(xi + 0.18, n + 0.012, f"{n:.3f}", ha="center", fontsize=5.6)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.legend(frameon=False, loc="upper right", handlelength=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_recovery.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_workflow()
    fig_architecture()
    fig_dcs_stem()
    fig_boundary()
    fig_uncertainty()
    fig_performance()
    fig_training()
    fig_ablation()
    fig_recovery()
    print("figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name, p.stat().st_size)
