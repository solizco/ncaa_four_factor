# 2026 NCAA Tournament — model picks

Model: logistic regression on Four Factor differentials, trained on 100,782 games (hold-out accuracy 94.7%, log loss 0.125, ROC AUC 0.990).

Standardized coefficients (log-odds per 1 SD): DIFF_EFG +7.09, DIFF_TOV -3.88, DIFF_ORB +3.22, DIFF_FTR +1.83

**Most likely champion:** Illinois (X03) — wins the title in 25.4% of simulations.

**Bracket champion (slot-by-slot most likely winners):** Purdue

**Cinderella:** Akron (seed 12) — reaches the Sweet 16 in 57.6% of simulations.

**Fade:** Kansas (seed 4) — wins its first game only 59.7% of the time, the lowest of any 1–4 seed.

**Contrarian:** Wisconsin (seed 5) vs High Point (seed 12). Historically, 5-seeds win this game 61% of the time; the model gives Wisconsin 16%.
