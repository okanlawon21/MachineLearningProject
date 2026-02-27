"""
==========================================================
  Iris Dataset Exploration
==========================================================
This script loads the classic Iris dataset from scikit-learn,
explores its structure and statistics, and produces visualisations
to help understand the data before building any ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ----------------------------------------------------------
# 1. Load the Iris dataset
# ----------------------------------------------------------
print("=" * 60)
print("Loading the Iris dataset from scikit-learn...")
print("=" * 60)

iris = load_iris()

# Build a DataFrame with feature columns plus a target column
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# ----------------------------------------------------------
# 2. First 5 rows
# ----------------------------------------------------------
print("\n### First 5 rows of the dataset ###")
print(df.head())

# ----------------------------------------------------------
# 3. Shape of the dataset
# ----------------------------------------------------------
print("\n### Dataset shape (rows, columns) ###")
print(df.shape)

# ----------------------------------------------------------
# 4. Summary statistics
# ----------------------------------------------------------
print("\n### Summary statistics (.describe()) ###")
print(df.describe())

# ----------------------------------------------------------
# 5. Transposed summary statistics
# ----------------------------------------------------------
print("\n### Transposed summary statistics (.describe().T) ###")
print(df.describe().T)

# ----------------------------------------------------------
# 6. Count of each target class
# ----------------------------------------------------------
print("\n### Count of each target class ###")
print(df["species"].value_counts())

# ----------------------------------------------------------
# 7. Charts
# ----------------------------------------------------------

feature_cols = iris.feature_names

# --- Chart 1: Distribution of each feature per species ---
print("\n### Generating feature distribution histograms... ###")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    "Feature Distributions by Species\n"
    "(Overlapping histograms reveal how well each feature separates the classes)",
    fontsize=13,
)
colors = {"setosa": "#4e79a7", "versicolor": "#f28e2b", "virginica": "#e15759"}
for ax, col in zip(axes.flat, feature_cols):
    for species, grp in df.groupby("species"):
        ax.hist(grp[col], bins=15, alpha=0.6, label=species, color=colors[species])
    ax.set_title(col)
    ax.set_xlabel("Value (cm)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=120)
plt.show()
print("Saved → feature_distributions.png")

# --- Chart 2: Box plots – feature spread and outliers ---
print("\n### Generating box plots... ###")
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle(
    "Box Plots of Features by Species\n"
    "(Box plots highlight the median, IQR, and potential outliers for each class)",
    fontsize=13,
)
for ax, col in zip(axes, feature_cols):
    data_by_species = [df[df["species"] == s][col].values for s in iris.target_names]
    bp = ax.boxplot(data_by_species, patch_artist=True, tick_labels=iris.target_names)
    for patch, color in zip(bp["boxes"], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col, fontsize=9)
    ax.set_ylabel("Value (cm)")
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig("box_plots.png", dpi=120)
plt.show()
print("Saved → box_plots.png")

# --- Chart 3: Scatter matrix (petal length vs petal width) ---
print("\n### Generating scatter plot (petal length vs petal width)... ###")
fig, ax = plt.subplots(figsize=(7, 6))
for species, grp in df.groupby("species"):
    ax.scatter(
        grp["petal length (cm)"],
        grp["petal width (cm)"],
        label=species,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4,
        s=60,
        color=colors[species],
    )
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title(
    "Petal Length vs Petal Width\n"
    "(Petal dimensions are the most discriminative features for separating species)"
)
ax.legend()
plt.tight_layout()
plt.savefig("petal_scatter.png", dpi=120)
plt.show()
print("Saved → petal_scatter.png")

# --- Chart 4: Correlation heatmap ---
print("\n### Generating correlation heatmap... ###")
corr = df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(feature_cols, fontsize=9)
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title(
    "Feature Correlation Heatmap\n"
    "(Values close to ±1 indicate strong linear relationships between features)"
)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.show()
print("Saved → correlation_heatmap.png")

print("\n" + "=" * 60)
print("Exploration complete.  Four charts have been saved:")
print("  • feature_distributions.png")
print("  • box_plots.png")
print("  • petal_scatter.png")
print("  • correlation_heatmap.png")
print("=" * 60)
