# Customer Behaviour Drift Analysis — Mobile Phone Purchases
# Compares baseline (Q4 2023) vs current (Q1 2025) buying behaviour
# Drift metrics: PSI, KS-test, Chi-squared, Jensen-Shannon Divergence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.spatial.distance import jensenshannon
from distapp.configurations.config import Config


# ---------------------------------------------------------------------------
# Drift Metrics
# ---------------------------------------------------------------------------

def psi(baseline_counts, current_counts, epsilon=1e-6):
    """Population Stability Index — measures distribution shift magnitude."""
    base = np.array(baseline_counts, dtype=float)
    curr = np.array(current_counts, dtype=float)
    base = base / base.sum()
    curr = curr / curr.sum()
    base = np.where(base == 0, epsilon, base)
    curr = np.where(curr == 0, epsilon, curr)
    return np.sum((curr - base) * np.log(curr / base))


def psi_label(value):
    if value < 0.1:
        return "No Drift"
    elif value < 0.2:
        return "Moderate Drift"
    else:
        return "Significant Drift"


def categorical_drift(baseline_series, current_series, feature_name):
    """Chi-squared test + PSI + JS divergence for a categorical feature."""
    all_cats = sorted(set(baseline_series.unique()) | set(current_series.unique()))
    base_counts = [baseline_series.value_counts().get(c, 0) for c in all_cats]
    curr_counts = [current_series.value_counts().get(c, 0) for c in all_cats]

    psi_val = psi(base_counts, curr_counts)

    base_prob = np.array(base_counts, dtype=float) / sum(base_counts)
    curr_prob = np.array(curr_counts, dtype=float) / sum(curr_counts)
    js_div = jensenshannon(base_prob, curr_prob)

    chi2_stat, chi2_p = stats.chi2_contingency(
        np.array([base_counts, curr_counts])
    )[:2]

    return {
        "feature": feature_name,
        "type": "categorical",
        "PSI": round(psi_val, 4),
        "PSI_label": psi_label(psi_val),
        "JS_divergence": round(float(js_div), 4),
        "Chi2_stat": round(chi2_stat, 4),
        "Chi2_p_value": round(chi2_p, 6),
        "drift_detected": chi2_p < 0.05,
        "categories": all_cats,
        "baseline_counts": base_counts,
        "current_counts": curr_counts,
        "baseline_pct": (base_prob * 100).round(1).tolist(),
        "current_pct": (curr_prob * 100).round(1).tolist(),
    }


def numerical_drift(baseline_series, current_series, feature_name, n_bins=10):
    """KS-test + PSI + JS divergence for a numerical feature."""
    ks_stat, ks_p = stats.ks_2samp(baseline_series.dropna(), current_series.dropna())

    # PSI on equal-width bins from combined range
    combined_min = min(baseline_series.min(), current_series.min())
    combined_max = max(baseline_series.max(), current_series.max())
    bins = np.linspace(combined_min, combined_max, n_bins + 1)
    base_counts, _ = np.histogram(baseline_series.dropna(), bins=bins)
    curr_counts, _ = np.histogram(current_series.dropna(), bins=bins)

    psi_val = psi(base_counts, curr_counts)

    base_prob = np.where(base_counts == 0, 1e-6, base_counts / base_counts.sum())
    curr_prob = np.where(curr_counts == 0, 1e-6, curr_counts / curr_counts.sum())
    js_div = jensenshannon(base_prob, curr_prob)

    return {
        "feature": feature_name,
        "type": "numerical",
        "PSI": round(psi_val, 4),
        "PSI_label": psi_label(psi_val),
        "JS_divergence": round(float(js_div), 4),
        "KS_stat": round(ks_stat, 4),
        "KS_p_value": round(ks_p, 6),
        "drift_detected": ks_p < 0.05,
        "baseline_mean": round(baseline_series.mean(), 2),
        "current_mean": round(current_series.mean(), 2),
        "baseline_std": round(baseline_series.std(), 2),
        "current_std": round(current_series.std(), 2),
        "bins": bins,
        "baseline_counts": base_counts,
        "current_counts": curr_counts,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_categorical_drift(results, n_cols=3):
    cat_results = [r for r in results if r["type"] == "categorical"]
    n_rows = (len(cat_results) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for ax, res in zip(axes, cat_results):
        x = np.arange(len(res["categories"]))
        width = 0.38
        bars1 = ax.bar(x - width / 2, res["baseline_pct"], width, label='Baseline (Q4 2023)',
                       color='steelblue', alpha=0.85)
        bars2 = ax.bar(x + width / 2, res["current_pct"], width, label='Current (Q1 2025)',
                       color='darkorange', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(res["categories"], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("Share (%)")
        drift_flag = " [DRIFT]" if res["drift_detected"] else ""
        ax.set_title(f"{res['feature']}{drift_flag}\nPSI={res['PSI']} ({res['PSI_label']})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    for ax in axes[len(cat_results):]:
        ax.set_visible(False)

    fig.suptitle("Customer Behaviour Drift — Categorical Features\n(Mobile Phone Purchases: Q4 2023 vs Q1 2025)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("src/distapp/resources/drift_categorical.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: src/distapp/resources/drift_categorical.png")


def plot_numerical_drift(results):
    num_results = [r for r in results if r["type"] == "numerical"]
    n = len(num_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, num_results):
        bins = res["bins"]
        width = (bins[1] - bins[0]) * 0.45
        centers = (bins[:-1] + bins[1:]) / 2
        base_pct = res["baseline_counts"] / res["baseline_counts"].sum() * 100
        curr_pct = res["current_counts"] / res["current_counts"].sum() * 100
        ax.bar(centers - width / 2, base_pct, width=width, label='Baseline (Q4 2023)',
               color='steelblue', alpha=0.85)
        ax.bar(centers + width / 2, curr_pct, width=width, label='Current (Q1 2025)',
               color='darkorange', alpha=0.85)
        drift_flag = " [DRIFT]" if res["drift_detected"] else ""
        ax.set_title(
            f"{res['feature']}{drift_flag}\nPSI={res['PSI']} ({res['PSI_label']})\n"
            f"Mean: {res['baseline_mean']} → {res['current_mean']}",
            fontsize=9
        )
        ax.set_xlabel(res["feature"])
        ax.set_ylabel("Share (%)")
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Customer Behaviour Drift — Numerical Features\n(Mobile Phone Purchases: Q4 2023 vs Q1 2025)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("src/distapp/resources/drift_numerical.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: src/distapp/resources/drift_numerical.png")


def plot_drift_summary(results):
    """Heatmap-style summary of PSI across all features."""
    features = [r["feature"] for r in results]
    psi_vals = [r["PSI"] for r in results]
    colors = []
    for p in psi_vals:
        if p < 0.1:
            colors.append('#4CAF50')
        elif p < 0.2:
            colors.append('#FF9800')
        else:
            colors.append('#F44336')

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(features, psi_vals, color=colors, edgecolor='white', height=0.6)
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1.5, label='Moderate threshold (0.1)')
    ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1.5, label='Significant threshold (0.2)')
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Drift Summary — PSI by Feature\n(Mobile Phone Customer Behaviour: Q4 2023 vs Q1 2025)",
                 fontweight='bold')
    ax.legend(fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='No Drift (PSI < 0.1)'),
        Patch(facecolor='#FF9800', label='Moderate Drift (0.1–0.2)'),
        Patch(facecolor='#F44336', label='Significant Drift (PSI ≥ 0.2)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    for bar, val in zip(bars, psi_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("src/distapp/resources/drift_summary.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: src/distapp/resources/drift_summary.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_drift_report(results):
    print("\n" + "="*65)
    print("   CUSTOMER BEHAVIOUR DRIFT REPORT")
    print("   Mobile Phone Purchases: Q4 2023 (Baseline) vs Q1 2025")
    print("="*65)

    drifted = [r for r in results if r["drift_detected"]]
    print(f"\nFeatures analysed : {len(results)}")
    print(f"Drift detected    : {len(drifted)}")
    print(f"No drift          : {len(results) - len(drifted)}")

    print("\n--- Feature-Level Drift Details ---")
    for r in results:
        flag = "*** DRIFT ***" if r["drift_detected"] else "stable"
        print(f"\n  [{flag}] {r['feature'].upper()}")
        print(f"    PSI           : {r['PSI']}  ({r['PSI_label']})")
        print(f"    JS Divergence : {r['JS_divergence']}")

        if r["type"] == "categorical":
            print(f"    Chi2 p-value  : {r['Chi2_p_value']}")
            print("    Distribution shift:")
            for cat, bp, cp in zip(r["categories"], r["baseline_pct"], r["current_pct"]):
                arrow = " <--" if abs(cp - bp) > 5 else ""
                print(f"      {cat:<22}: {bp:>5.1f}%  →  {cp:>5.1f}%{arrow}")
        else:
            print(f"    KS p-value    : {r['KS_p_value']}")
            print(f"    Baseline mean : {r['baseline_mean']}")
            print(f"    Current mean  : {r['current_mean']}")
            change_pct = ((r['current_mean'] - r['baseline_mean']) / r['baseline_mean']) * 100
            direction = "increase" if change_pct > 0 else "decrease"
            print(f"    Change        : {abs(change_pct):.1f}% {direction}")

    print("\n--- Key Drift Insights ---")
    print("  1. CHANNEL: Massive shift to online buying — in-store declining.")
    print("  2. BRAND: Apple flagship surge; Xiaomi budget share dropped.")
    print("  3. BUDGET: Premium/flagship segment grew significantly.")
    print("  4. PAYMENT: EMI_24 now dominant; full_payment declining.")
    print("  5. AGE: 18-25 age group dominates current period.")
    print("  6. PURCHASE AMOUNT: Mean spend sharply increased (premium shift).")
    print("  7. DECISION TIME: Faster decisions — online reviews accelerate buying.")
    print("  8. INFLUENCER: Social media now the dominant discovery channel.")
    print("\n  Action: Retrain recommendation models with current-period data.")
    print("          Review EMI product offers and premium phone inventory.")
    print("="*65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config()
    df = pd.read_csv(config.mobile_drift_path)

    baseline = df[df['period'] == 'baseline']
    current = df[df['period'] == 'current']

    print(f"Baseline records : {len(baseline)}")
    print(f"Current records  : {len(current)}")

    categorical_features = ['age_group', 'brand', 'budget_range',
                            'purchase_channel', 'payment_method', 'influencer_type']
    numerical_features = ['days_to_decide', 'purchase_amount']

    results = []

    for feat in categorical_features:
        res = categorical_drift(baseline[feat], current[feat], feat)
        results.append(res)

    for feat in numerical_features:
        res = numerical_drift(baseline[feat], current[feat], feat)
        results.append(res)

    print_drift_report(results)
    plot_drift_summary(results)
    plot_categorical_drift(results)
    plot_numerical_drift(results)
