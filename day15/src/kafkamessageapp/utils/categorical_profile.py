import pandas as pd
import matplotlib.pyplot as plt
from kafkamessageapp.configurations.conf import KafkaConfig

df = pd.read_csv(KafkaConfig.pizza_cat_path)
cat_cols = df.select_dtypes(include="object").columns.tolist()

# ── Print summary ─────────────────────────────────────────────────────────────
for col in cat_cols:
    s = df[col]
    vc = s.value_counts()
    print(f"\n── {col} ──")
    print(f"Unique values : {s.nunique()}")
    print(f"Missing       : {s.isna().sum()}")
    print(f"Mode          : {vc.index[0]} ({vc.iloc[0]}x)")
    print(pd.DataFrame({
        "Count": vc,
        "Percent": (vc / len(s) * 100).round(1).astype(str) + "%"
    }).to_string())

# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(cat_cols)
fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
fig.suptitle("Categorical Distribution Profile", fontsize=14, fontweight="bold")

colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

for i, col in enumerate(cat_cols):
    vc = df[col].value_counts()
    color = colors[i % len(colors)]

    # Bar chart
    axes[i, 0].bar(vc.index, vc.values, color=color, edgecolor="white")
    axes[i, 0].set_title(f"{col} — frequency")
    axes[i, 0].set_ylabel("Count")
    axes[i, 0].tick_params(axis="x", rotation=15)

    # Pie chart
    axes[i, 1].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                   startangle=90, colors=[color] + ["#B5D4F4", "#9FE1CB",
                   "#F5C4B3", "#CECBF6"][:len(vc)-1])
    axes[i, 1].set_title(f"{col} — proportion")

plt.tight_layout()
plt.savefig("categorical_profile.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → categorical_profile.png")
