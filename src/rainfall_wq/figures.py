"""Annotated figures that state the rainfall–pollution finding in plain language."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_SAMPLE, OUTPUTS

ORDER = ["sunny", "light", "medium", "heavy", "rainstorm", "extreme"]
LABELS = {
    "sunny": "Sunny",
    "light": "Light",
    "medium": "Medium",
    "heavy": "Heavy",
    "rainstorm": "Rainstorm",
    "extreme": "Extreme",
}


def _callout(ax, text: str, xy, xytext, ha="left", va="center") -> None:
    ax.annotate(
        text,
        xy=xy,
        xycoords="data",
        xytext=xytext,
        textcoords="axes fraction",
        fontsize=8,
        color="0.15",
        ha=ha,
        va=va,
        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", linewidth=0.6),
    )


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def plot_bacteria_by_class(path: Path | None = None) -> Path:
    """Harbor Survey N4: sewage indicators rise once rain is medium/heavy."""
    _style()
    csv = path or DATA_PROCESSED / "harbor_pathogens_by_rain_class.csv"
    if not csv.exists():
        csv = DATA_SAMPLE / "harbor_pathogens_by_rain_class.csv"
    raw = pd.read_csv(csv)
    n4 = raw[raw["station"] == "N4"].set_index("rain_class")
    cats = [c for c in ORDER if c in n4.index and c != "rainstorm"]
    x = np.arange(len(cats))
    entero = [n4.loc[c, "enterococci_median"] for c in cats]
    fecal = [n4.loc[c, "fecal_coliform_median"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    ax.bar(x - w / 2, entero, w, color="#E45756", label="Enterococci")
    ax.bar(x + w / 2, fecal, w, color="#4C78A8", label="Fecal coliform")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in cats])
    ax.set_ylabel("Median cells / 100 mL (N4, W. 42nd St)")
    ax.set_xlabel("Central Park rainfall class that day")
    ax.set_title("The sewage pulse is clearest in bacteria")
    ax.legend(fontsize=8, loc="upper left")
    sunny_i = cats.index("sunny")
    heavy_i = cats.index("heavy")
    _callout(
        ax,
        "Dry weather:\nlittle sewage signal",
        xy=(sunny_i + w / 2, fecal[sunny_i]),
        xytext=(0.04, 0.55),
    )
    _callout(
        ax,
        "Above a small rain threshold,\nuntreated sewage reaches the Hudson\n(CSO). Heavy: 204 vs 51 fecal.",
        xy=(heavy_i + w / 2, fecal[heavy_i]),
        xytext=(0.42, 0.78),
    )
    fig.tight_layout()
    out = OUTPUTS / "harbor_bacteria_by_rain_class.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_turbidity_by_class(df: pd.DataFrame) -> Path:
    """Everyday rain barely moves turbidity; extreme storms do."""
    _style()
    present = [c for c in ORDER if c in set(df["rain_class"]) and df.loc[df["rain_class"] == c, "turbidity_fnu"].notna().any()]
    data = [df.loc[df["rain_class"] == c, "turbidity_fnu"].dropna() for c in present]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    for i, (box, cls) in enumerate(zip(bp["boxes"], present)):
        box.set_facecolor("#E45756" if cls == "extreme" else "#9ecae1")
        box.set_edgecolor("0.3")
    ax.set_xticklabels([LABELS[c] for c in present])
    ax.set_ylabel("Daily-mean turbidity (FNU)")
    ax.set_xlabel("Central Park rainfall class")
    ax.set_title("The pulse is strongest in heavy / extreme storms")
    if "extreme" in present:
        xi = present.index("extreme") + 1
        med = float(np.median(data[xi - 1]))
        _callout(
            ax,
            "Extreme rain (≥100 mm):\ncloudier water — runoff + overflow",
            xy=(xi, med),
            xytext=(0.52, 0.82),
        )
    if "sunny" in present and "medium" in present:
        _callout(
            ax,
            "Sunny through heavy look alike.\nTides hide everyday rain in turbidity.",
            xy=(1, float(np.median(data[0]))),
            xytext=(0.02, 0.78),
        )
    fig.tight_layout()
    out = OUTPUTS / "turbidity_by_rain_class.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_timeseries(df: pd.DataFrame) -> Path:
    """Mark extreme storms; show that oxygen/turbidity are mostly seasonal/tidal."""
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True)
    axes[0].bar(df.index, df["precip_mm"], color="#4C78A8", width=1.0)
    axes[0].set_ylabel("Precip (mm)")
    axes[0].set_title("Rain delivers a pulse — easy to miss in everyday turbidity and oxygen")
    if "rain_class" in df.columns:
        extreme = df.index[df["rain_class"] == "extreme"]
        for i, t in enumerate(extreme):
            axes[0].axvline(t, color="#E45756", lw=1.2, alpha=0.85)
        if len(extreme):
            t_mark = extreme[min(2, len(extreme) - 1)]
            y_mark = df.loc[t_mark, "precip_mm"]
            if not np.isfinite(y_mark):
                y_mark = float(np.nanmax(df["precip_mm"]))
            _callout(
                axes[0],
                "Extreme storms (red):\nrunoff + untreated sewage",
                xy=(t_mark, y_mark),
                xytext=(0.55, 0.72),
            )
    if "turbidity_fnu" in df.columns:
        axes[1].plot(df.index, df["turbidity_fnu"], color="#F58518", lw=0.8)
        axes[1].set_ylabel("Turbidity (FNU)")
        _callout(
            axes[1],
            "Everyday cloudiness is noisy.\nA typical rain day does not jump out.",
            xy=(df.index[len(df) // 3], float(df["turbidity_fnu"].median())),
            xytext=(0.02, 0.78),
        )
    if "dissolved_oxygen_mg_l" in df.columns:
        axes[2].plot(df.index, df["dissolved_oxygen_mg_l"], color="#54A24B", lw=0.8)
        axes[2].set_ylabel("DO (mg/L)")
        mid = df.index[len(df) // 2]
        y_mid = float(df["dissolved_oxygen_mg_l"].median())
        _callout(
            axes[2],
            "Oxygen is mostly the seasons\n(high in winter, low in summer),\nnot the rain calendar.",
            xy=(mid, y_mid),
            xytext=(0.52, 0.28),
        )
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    out = OUTPUTS / "timeseries_rain_turbidity_do.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_story_triptych(df: pd.DataFrame) -> Path:
    """One figure, three claims: bacteria, extreme storms, easy to miss."""
    _style()
    csv = DATA_PROCESSED / "harbor_pathogens_by_rain_class.csv"
    if not csv.exists():
        csv = DATA_SAMPLE / "harbor_pathogens_by_rain_class.csv"
    n4 = pd.read_csv(csv)
    n4 = n4[n4["station"] == "N4"].set_index("rain_class")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    cats = [c for c in ["sunny", "light", "medium", "heavy"] if c in n4.index]
    x = np.arange(len(cats))
    fecal = [n4.loc[c, "fecal_coliform_median"] for c in cats]
    axes[0].bar(x, fecal, color=["#9ecae1" if c != "heavy" else "#E45756" for c in cats])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[c] for c in cats])
    axes[0].set_ylabel("Median fecal coliform (cells/100 mL)")
    axes[0].set_title("Clearest in bacteria")
    _callout(
        axes[0],
        "Heavy rain:\nuntreated sewage\n(CSO)",
        xy=(len(cats) - 1, fecal[-1]),
        xytext=(0.08, 0.78),
    )

    present = [c for c in ORDER if c in set(df["rain_class"]) and df.loc[df["rain_class"] == c, "turbidity_fnu"].notna().any()]
    data = [df.loc[df["rain_class"] == c, "turbidity_fnu"].dropna() for c in present]
    bp = axes[1].boxplot(data, showfliers=False, patch_artist=True)
    for box, cls in zip(bp["boxes"], present):
        box.set_facecolor("#E45756" if cls == "extreme" else "#9ecae1")
        box.set_edgecolor("0.3")
    axes[1].set_xticklabels([LABELS[c] for c in present], fontsize=8)
    axes[1].set_ylabel("Turbidity (FNU)")
    axes[1].set_title("Strongest in extreme storms")
    if "extreme" in present:
        xi = present.index("extreme") + 1
        _callout(
            axes[1],
            "Only the rarest\nstorms jump",
            xy=(xi, float(np.median(data[xi - 1]))),
            xytext=(0.05, 0.78),
        )

    if "dissolved_oxygen_mg_l" in df.columns:
        axes[2].plot(df.index, df["dissolved_oxygen_mg_l"], color="#54A24B", lw=0.7)
        axes[2].set_ylabel("Dissolved oxygen (mg/L)")
        axes[2].set_title("Easy to miss in oxygen")
        _callout(
            axes[2],
            "The big swing is\nwinter vs summer,\nnot rain vs dry",
            xy=(df.index[len(df) // 2], float(df["dissolved_oxygen_mg_l"].median())),
            xytext=(0.08, 0.22),
        )
    axes[2].tick_params(axis="x", labelrotation=30)
    fig.suptitle(
        "Rain does not just dilute the Hudson — it delivers runoff and, above a small threshold, untreated sewage",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    out = OUTPUTS / "story_rain_pollution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_epoch_annotated(epoch: pd.DataFrame) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels = {
        "below_cso_threshold": "Below ~0.1 in (little/no CSO)",
        "cso_proxy": "Above ~0.1 in (CSO-likely)",
    }
    colors = {"below_cso_threshold": "#4C78A8", "cso_proxy": "#E45756"}
    for group, color in colors.items():
        g = epoch[epoch["group"] == group].sort_values("lead_h")
        if g.empty:
            continue
        ax.plot(g["lead_h"], g["mean"], color=color, lw=1.6, label=labels[group])
        ax.fill_between(g["lead_h"], g["q25"], g["q75"], color=color, alpha=0.15)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xlabel("Hours from rain-event start")
    ax.set_ylabel("Turbidity residual after tides (FNU)")
    ax.set_title("Above a small rain threshold, a wet-weather pulse")
    ax.legend(fontsize=8, loc="upper right")
    _callout(
        ax,
        "CSO-scale storms sit slightly\nhigher after the rain starts.\nThe everyday tidal river still\nswamps most of the signal.",
        xy=(8, 0.5),
        xytext=(0.48, 0.18),
    )
    fig.tight_layout()
    out = OUTPUTS / "superposed_epoch_turbidity.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_first_flush_annotated(mv: pd.DataFrame) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1, label="Even mix of dirt through the storm")
    ax.plot(mv["V"], mv["M_median"], color="#E45756", lw=2, label="What we see (turbidity proxy)")
    ax.fill_between(mv["V"], mv["M_q25"], mv["M_q75"], color="#E45756", alpha=0.2)
    ax.axvline(0.3, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel("Share of the rain that has fallen so far (V)")
    ax.set_ylabel("Share of extra cloudiness so far (M)")
    n = int(mv["n_events"].iloc[0])
    ax.set_title(f"Street-runoff pulse at the start of CSO-scale storms (n={n})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    m30 = float(mv.loc[np.isclose(mv["V"], 0.3), "M_median"].iloc[0])
    _callout(
        ax,
        f"First 30% of rain carries\n~{m30:.0%} of extra turbidity\n(runoff pulse, not dilution)",
        xy=(0.3, m30),
        xytext=(0.42, 0.22),
    )
    fig.tight_layout()
    out = OUTPUTS / "first_flush_mv.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def annotate_all(daily: pd.DataFrame) -> list[Path]:
    paths = [
        plot_story_triptych(daily),
        plot_bacteria_by_class(),
        plot_turbidity_by_class(daily),
        plot_timeseries(daily),
    ]
    epoch = DATA_PROCESSED / "epoch_turbidity.csv"
    mv = DATA_PROCESSED / "first_flush_mv.csv"
    if epoch.exists():
        paths.append(plot_epoch_annotated(pd.read_csv(epoch)))
    if mv.exists():
        paths.append(plot_first_flush_annotated(pd.read_csv(mv)))
    return paths
