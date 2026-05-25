#create power analysis using medicine_effect and fraud_transactions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from distapp.configurations.config import Config


def power_analysis(effect_size, alpha, power=0.95):
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=1,
        alternative="two-sided"
    )
    return sample_size


def calculate_medicine_effect(medicine_effect, control_effect):
    # Cohen's d for two proportions
    effect_size = (medicine_effect - control_effect) / np.sqrt((medicine_effect + control_effect) / 2)
    return effect_size


def calculate_fraud_effect(baseline_fraud_rate, target_fraud_rate):
    # Cohen's h — correct effect size for comparing two proportions
    return abs(proportion_effectsize(baseline_fraud_rate, target_fraud_rate))


def plot_fraud_power_curve(fraud_effect_size, alpha=0.05):
    """Power curve: required sample size vs statistical power."""
    power_levels = np.linspace(0.5, 0.99, 100)
    sample_sizes = [power_analysis(fraud_effect_size, alpha, p) for p in power_levels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Fraud Transaction Power Analysis", fontsize=15, fontweight='bold')

    # --- Plot 1: Power Curve ---
    axes[0].plot(power_levels, sample_sizes, color='steelblue', linewidth=2)
    axes[0].axvline(x=0.80, color='red', linestyle='--', label='80% Power')
    axes[0].axvline(x=0.95, color='green', linestyle='--', label='95% Power')
    axes[0].set_xlabel("Statistical Power")
    axes[0].set_ylabel("Required Sample Size per Group")
    axes[0].set_title("Sample Size vs Statistical Power")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Effect Size Sensitivity ---
    effect_sizes = np.linspace(0.05, 0.8, 100)
    sizes_at_80 = [power_analysis(e, alpha, 0.80) for e in effect_sizes]
    sizes_at_95 = [power_analysis(e, alpha, 0.95) for e in effect_sizes]
    axes[1].plot(effect_sizes, sizes_at_80, color='steelblue', linewidth=2, label='80% Power')
    axes[1].plot(effect_sizes, sizes_at_95, color='darkorange', linewidth=2, label='95% Power')
    axes[1].axvline(x=fraud_effect_size, color='red', linestyle='--', label=f'Current ES={fraud_effect_size:.3f}')
    axes[1].set_xlabel("Effect Size (Cohen's h)")
    axes[1].set_ylabel("Required Sample Size per Group")
    axes[1].set_title("Sample Size vs Effect Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 5000)

    # --- Plot 3: Alpha Sensitivity ---
    alphas = [0.01, 0.05, 0.10]
    colors = ['purple', 'steelblue', 'green']
    for a, c in zip(alphas, colors):
        s = [power_analysis(fraud_effect_size, a, p) for p in power_levels]
        axes[2].plot(power_levels, s, color=c, linewidth=2, label=f'alpha={a}')
    axes[2].set_xlabel("Statistical Power")
    axes[2].set_ylabel("Required Sample Size per Group")
    axes[2].set_title("Sample Size vs Alpha Level")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("src/distapp/resources/fraud_power_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Power curve saved: src/distapp/resources/fraud_power_analysis.png")


def plot_fraud_breakdown(fraud_df):
    """Bar charts for fraud rate by merchant category and transaction channel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fraud Rate Breakdown", fontsize=14, fontweight='bold')

    category_stats = fraud_df.groupby('merchant_category')['is_fraud'].agg(['sum', 'count'])
    category_stats.columns = ['fraud_count', 'total']
    category_stats['fraud_rate'] = category_stats['fraud_count'] / category_stats['total']
    category_stats = category_stats.sort_values('fraud_rate', ascending=False)

    axes[0].bar(category_stats.index, category_stats['fraud_rate'] * 100, color='steelblue', edgecolor='white')
    axes[0].set_title("Fraud Rate by Merchant Category")
    axes[0].set_xlabel("Merchant Category")
    axes[0].set_ylabel("Fraud Rate (%)")
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].grid(axis='y', alpha=0.3)
    for i, (idx, row) in enumerate(category_stats.iterrows()):
        axes[0].text(i, row['fraud_rate'] * 100 + 0.3, f"{row['fraud_rate']*100:.1f}%", ha='center', fontsize=9)

    channel_stats = fraud_df.groupby('transaction_channel')['is_fraud'].agg(['sum', 'count'])
    channel_stats.columns = ['fraud_count', 'total']
    channel_stats['fraud_rate'] = channel_stats['fraud_count'] / channel_stats['total']
    channel_stats = channel_stats.sort_values('fraud_rate', ascending=False)

    axes[1].bar(channel_stats.index, channel_stats['fraud_rate'] * 100, color='darkorange', edgecolor='white')
    axes[1].set_title("Fraud Rate by Transaction Channel")
    axes[1].set_xlabel("Transaction Channel")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)
    for i, (idx, row) in enumerate(channel_stats.iterrows()):
        axes[1].text(i, row['fraud_rate'] * 100 + 0.3, f"{row['fraud_rate']*100:.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("src/distapp/resources/fraud_breakdown.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Breakdown chart saved: src/distapp/resources/fraud_breakdown.png")


if __name__ == "__main__":
    config = Config()

    # --- Medicine Effect Power Analysis ---
    df = pd.read_csv(config.effectiveness_path)
    medicine_effect_count = (df['medicine_name'] == 'Metformin').sum()
    print(medicine_effect_count)
    medicine_effect_mean = medicine_effect_count / len(df)
    control_effect_count = (df['medicine_name'] == 'Glimepiride').sum()
    print(control_effect_count)
    control_effect_mean = control_effect_count / len(df)
    effect_size = calculate_medicine_effect(medicine_effect_mean, control_effect_mean)
    new_medicine_effect_mean = medicine_effect_mean * 1.2
    alpha = new_medicine_effect_mean - medicine_effect_mean

    print(f"Medicine Effect Mean: {medicine_effect_mean}")
    print(f"Control Effect Mean: {control_effect_mean}")
    print(f"Effect Size (Cohen's d): {effect_size}")
    print(f"New Medicine Effect Mean (20% increase): {new_medicine_effect_mean}")
    print(f"Alpha (Difference between new and old medicine effect mean): {alpha}")

    required_sample_size = power_analysis(medicine_effect_mean, alpha)
    print(f"Required Sample Size for 80% Power and Alpha of {alpha}: {required_sample_size}")

    # --- Fraud Transaction Power Analysis ---
    print("\n" + "="*55)
    print("   FRAUD TRANSACTION POWER ANALYSIS")
    print("="*55)

    fraud_df = pd.read_csv(config.fraud_path)

    total_transactions = len(fraud_df)
    fraud_count = int(fraud_df['is_fraud'].sum())
    fraud_rate = fraud_count / total_transactions

    target_fraud_rate = fraud_rate * 0.80
    alpha_fraud = 0.05

    print(f"\nTotal Transactions        : {total_transactions}")
    print(f"Fraud Transactions        : {fraud_count}")
    print(f"Baseline Fraud Rate       : {fraud_rate:.4f} ({fraud_rate * 100:.2f}%)")
    print(f"Target Fraud Rate (-20%)  : {target_fraud_rate:.4f} ({target_fraud_rate * 100:.2f}%)")

    fraud_effect_size = calculate_fraud_effect(fraud_rate, target_fraud_rate)
    print(f"Effect Size (Cohen's h)   : {fraud_effect_size:.4f}")

    required_per_group = power_analysis(fraud_effect_size, alpha_fraud, power=0.80)
    print(f"\nRequired Sample Size/Group (80% power, alpha=0.05): {required_per_group:.0f}")
    print(f"Total Required Sample Size (both groups)          : {required_per_group * 2:.0f}")

    required_95 = power_analysis(fraud_effect_size, alpha_fraud, power=0.95)
    print(f"\nRequired Sample Size/Group (95% power, alpha=0.05): {required_95:.0f}")
    print(f"Total Required Sample Size (both groups)          : {required_95 * 2:.0f}")

    # Adequacy check
    actual_per_group = total_transactions / 2
    is_adequate_80 = actual_per_group >= required_per_group
    is_adequate_95 = actual_per_group >= required_95
    print(f"\nCurrent sample per group  : {actual_per_group:.0f}")
    print(f"Adequate for 80% power?   : {'YES' if is_adequate_80 else 'NO — need more data'}")
    print(f"Adequate for 95% power?   : {'YES' if is_adequate_95 else 'NO — need more data'}")

    print("\nFraud Rate by Merchant Category:")
    category_stats = fraud_df.groupby('merchant_category')['is_fraud'].agg(['sum', 'count'])
    category_stats.columns = ['fraud_count', 'total']
    category_stats['fraud_rate'] = category_stats['fraud_count'] / category_stats['total']
    print(category_stats.to_string())

    print("\nFraud Rate by Transaction Channel:")
    channel_stats = fraud_df.groupby('transaction_channel')['is_fraud'].agg(['sum', 'count'])
    channel_stats.columns = ['fraud_count', 'total']
    channel_stats['fraud_rate'] = channel_stats['fraud_count'] / channel_stats['total']
    print(channel_stats.to_string())

    # High-risk category power analysis
    print("\n--- High-Risk Category Deep Dive (online_shopping) ---")
    online_fraud = fraud_df[fraud_df['merchant_category'] == 'online_shopping']
    if len(online_fraud) > 0:
        online_fraud_rate = online_fraud['is_fraud'].mean()
        online_target = online_fraud_rate * 0.80
        online_es = calculate_fraud_effect(online_fraud_rate, online_target)
        online_n = power_analysis(online_es, 0.05, power=0.80)
        print(f"online_shopping fraud rate : {online_fraud_rate:.4f} ({online_fraud_rate*100:.2f}%)")
        print(f"Effect size (Cohen's h)    : {online_es:.4f}")
        print(f"Sample needed per group    : {online_n:.0f}")

    # Visualisations
    plot_fraud_power_curve(fraud_effect_size, alpha=alpha_fraud)
    plot_fraud_breakdown(fraud_df)
