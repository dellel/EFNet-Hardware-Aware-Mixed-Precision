from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/litemstemporal_video64"
INPUT = RESULTS / "test_predictions.csv"
OUT = RESULTS / "clinical_robustness"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = {"FileName", "EF_true", "EF_pred"}
if not required.issubset(df.columns):
    raise ValueError(f"Colonnes manquantes : {required - set(df.columns)}")

df = df.dropna(subset=["EF_true", "EF_pred"]).copy()

y = df["EF_true"].to_numpy(dtype=float)
p = df["EF_pred"].to_numpy(dtype=float)
err = p - y
abs_err = np.abs(err)
mean_pair = (p + y) / 2

bias = float(np.mean(err))
sd = float(np.std(err, ddof=1))
loa_low = bias - 1.96 * sd
loa_high = bias + 1.96 * sd

median_ae = float(np.median(abs_err))
p90 = float(np.percentile(abs_err, 90))
p95 = float(np.percentile(abs_err, 95))

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 130,
})

# Figure 1 — Bland–Altman
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mean_pair, err, s=16, alpha=0.45)
ax.axhline(bias, linestyle="-", label=f"Biais = {bias:.2f}")
ax.axhline(loa_low, linestyle="--",
           label=f"Limite basse 95 % = {loa_low:.2f}")
ax.axhline(loa_high, linestyle="--",
           label=f"Limite haute 95 % = {loa_high:.2f}")
ax.axhline(0, linestyle=":", linewidth=1)
ax.set_xlabel("Moyenne FEVG prédite et réelle (%)")
ax.set_ylabel("FEVG prédite − FEVG réelle (points)")
ax.set_title("Bland–Altman — ensemble TEST")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "01_bland_altman.png", bbox_inches="tight")
plt.close(fig)

# Figure 2 — Valeurs prédites contre valeurs réelles
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y, p, s=16, alpha=0.4)
lo = min(float(y.min()), float(p.min()))
hi = max(float(y.max()), float(p.max()))
ax.plot([lo, hi], [lo, hi], linestyle="--", label="Prédiction idéale")
ax.set_xlabel("FEVG réelle (%)")
ax.set_ylabel("FEVG prédite (%)")
ax.set_title("FEVG prédite vs réelle — ensemble TEST")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "02_predictions_vs_reference.png", bbox_inches="tight")
plt.close(fig)

# Figure 3 — Distribution des erreurs absolues
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.hist(abs_err, bins=35, edgecolor="black", alpha=0.75)
ax.axvline(median_ae, linestyle="-",
           label=f"Médiane = {median_ae:.2f}")
ax.axvline(p90, linestyle="--", label=f"P90 = {p90:.2f}")
ax.axvline(p95, linestyle=":", label=f"P95 = {p95:.2f}")
ax.set_xlabel("Erreur absolue (points de FEVG)")
ax.set_ylabel("Nombre de vidéos")
ax.set_title("Distribution des erreurs absolues — TEST")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "03_absolute_error_distribution.png", bbox_inches="tight")
plt.close(fig)

# Figure 4 — MAE par catégorie de FEVG réelle
bins = [-np.inf, 40, 50, np.inf]
labels = ["FEVG basse (<40 %)", "Intermédiaire (40–49 %)",
          "Préservée (≥50 %)"]
df["EF_category"] = pd.cut(
    df["EF_true"], bins=bins, labels=labels,
    right=False, include_lowest=True
)
df["absolute_error"] = abs_err

summary = (
    df.groupby("EF_category", observed=False)
      .agg(
          N=("absolute_error", "size"),
          MAE=("absolute_error", "mean"),
          Median_AE=("absolute_error", "median"),
          RMSE=("absolute_error",
                lambda x: float(np.sqrt(np.mean(x.to_numpy() ** 2)))),
      )
      .reset_index()
)
summary.to_csv(OUT / "mae_by_ef_category.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5.5))
valid = summary[summary["N"] > 0]
bars = ax.bar(valid["EF_category"].astype(str), valid["MAE"])
ax.set_ylabel("MAE (points de FEVG)")
ax.set_title("Erreur absolue moyenne par catégorie de FEVG réelle")
ax.tick_params(axis="x", rotation=12)
for bar, n in zip(bars, valid["N"]):
    ax.annotate(
        f"N={int(n)}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom"
    )
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "04_mae_by_ef_category.png", bbox_inches="tight")
plt.close(fig)

# Figure 5 — Les 20 plus grandes erreurs
worst = df.copy()
worst["signed_error"] = worst["EF_pred"] - worst["EF_true"]
worst = worst.sort_values("absolute_error", ascending=False).head(20)
worst.to_csv(OUT / "worst_20_predictions.csv", index=False)

worst = worst.sort_values("absolute_error", ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(worst["FileName"].astype(str), worst["absolute_error"])
ax.set_xlabel("Erreur absolue (points de FEVG)")
ax.set_ylabel("Vidéo")
ax.set_title("20 plus grandes erreurs de prédiction — TEST")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "05_worst_20_errors.png", bbox_inches="tight")
plt.close(fig)

# Tableau de synthèse
metrics = {
    "N": len(df),
    "MAE": float(np.mean(abs_err)),
    "RMSE": float(np.sqrt(np.mean(err ** 2))),
    "Bias": bias,
    "Error_SD": sd,
    "Bland_Altman_Lower_95": loa_low,
    "Bland_Altman_Upper_95": loa_high,
    "Median_Absolute_Error": median_ae,
    "Absolute_Error_P90": p90,
    "Absolute_Error_P95": p95,
}
pd.DataFrame([metrics]).to_csv(OUT / "test_error_summary.csv", index=False)

print("\n=== SYNTHÈSE DES ERREURS TEST ===")
for key, value in metrics.items():
    print(f"{key:28s}: {value:.4f}" if isinstance(value, float)
          else f"{key:28s}: {value}")
print(f"\nFigures et tableaux enregistrés dans : {OUT}")
print("Aucun réentraînement ni ajustement du modèle n'a été effectué.")
