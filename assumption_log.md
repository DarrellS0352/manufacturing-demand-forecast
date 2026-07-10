# Manufacturing Demand Forecast — Assumption Log

Decisions and assumptions made during this project, sourced from EDA and modeling notebooks.
Feeds the README limitations section. Interview Risk column flags where you need a prepared answer.

---

## Data Quality Decisions

| # | Decision | Rationale | Interview Risk |
|---|---|---|---|
| 1 | Dropped 6,669 null `Date` rows from `Whse_A` | Classified as Missing at Random (MAR) — missingness is systematic to Whse_A but not to Product_Code or Product_Category. 4.5% of Whse_A rows — small enough to drop cleanly without imputation. | Low — well-reasoned, documented |
| 2 | Dropped parenthesized `Order_Demand` values e.g. `(100)` | Represent order cancellations, not demand. Including them would negatively bias demand estimates. | Low — standard for this dataset |
| 3 | Filtered to `Order_Demand > 0` after monthly aggregation | Zero-demand months are uninformative for forecasting and distort the ADI calculation in Syntetos-Boylan classification. | Low — defensible |
| 4 | Aggregated raw daily transactions to monthly periods | Matches the 28-day forecast horizon and Syntetos-Boylan classification granularity. Daily data is too sparse for reliable ADI calculation at the SKU × warehouse level. | Low — deliberate, documented |

---

## Syntetos-Boylan Classification Assumptions

| # | Assumption | Rationale | Interview Risk |
|---|---|---|---|
| 5 | ADI denominator uses total dataset span (not per-SKU active period) | Consistent denominator across all SKUs. Using per-SKU active months would understate ADI for recently introduced products, potentially misclassifying them as smooth. | Medium — interviewer may probe this choice |
| 6 | CV² std set to 0 for single-observation SKUs (NaN std replaced with 0) | A single observation implies no observed variability. Setting to 0 classifies these as low-CV² (smooth or intermittent), which is conservative. | Low — explicitly documented in code |
| 7 | Standard Syntetos-Boylan thresholds used: ADI=1.32, CV²=0.49 | Published thresholds from Syntetos & Boylan (2005). No dataset-specific tuning applied — tuning thresholds would require held-out service level data to validate. | Low — cite the paper if asked |

---

## Modeling Scope Decisions

| # | Decision | Rationale | Interview Risk |
|---|---|---|---|
| 8 | Category_019 excluded from LightGBM pipeline entirely | Demand volumes 100x greater than remainder of portfolio (median 18,000, max 10.45M, p99/median ratio 123x, skewness 9.20). Extreme internal heterogeneity persists even within the category — a dedicated model would be required in production. | Medium — must explain why exclusion is legitimate, not avoidance |
| 9 | Target-encoded `product_code_warehouse_encoded` feature removed | Dominated feature importance by ~20x over next feature, causing model to memorize historical means rather than learn demand patterns from lag and rolling features. Removing it improved generalization and interpretability. | Medium — good story, shows diagnostic awareness |
| 10 | Rows missing core lag and rolling features dropped from LightGBM training | 3,936 rows (6.3% of smooth + erratic training data) dropped. Rows missing `demand_lag1–3` and `demand_rolling_mean/std_3m` have insufficient history to contribute signal and add noise. LightGBM handles NaNs natively but removing them improved stability. | Low — small %, well-reasoned |
| 11 | Fold 1 validation set used as early stopping monitor | Avoids carving off recent training data (November–December 2015, the highest-demand months) as a monitor set. Preserves full historical signal in training. | Medium — interviewer may ask why not a separate held-out monitor set |
| 12 | Fixed-origin CV: training data fixed at pre-2016 across all three folds | Dataset size and the importance of full historical signal for lag features makes fixed-origin more appropriate than shrinking the training window per fold. | Medium — must distinguish fixed-origin from rolling-window clearly |
| 13 | Single LightGBM model trained on smooth + erratic segments combined | Demand type one-hot encoding allows model to learn segment-specific patterns without training separate models. Simpler and more defensible. | Low — straightforward rationale |
| 14 | Intermittent SKUs handled with Croston's method implemented from scratch | Croston's method specifically designed for intermittent demand — separates demand frequency estimation from demand size estimation. Implemented from scratch for interview transparency and to demonstrate algorithmic understanding. | Low — strong choice, easy to defend |
| 15 | Lumpy SKUs handled with conservative demand estimate; safety stock buffer absorbs uncertainty | Point forecast unreliable for lumpy demand (high ADI, high CV²). In practice, lumpy SKUs are often handled via manual review or wide safety stock buffers rather than ML forecasting. | Medium — must explain why forecasting lumpy demand directly is not attempted |

---

## Inventory Policy Assumptions

| # | Assumption | Rationale | Interview Risk |
|---|---|---|---|
| 16 | Forecast horizon = 28 days (1 month) | Kaggle dataset description explicitly mentions ocean shipping lead times exceeding one month, making monthly the natural replenishment cycle. No actual lead time data provided in the dataset. | High — must acknowledge this is an assumption, not observed data |
| 17 | Service level target: 95% | Standard manufacturing/distribution target. No client-specified service level available in the dataset. | Low — state it's an assumption |
| 18 | Z = 1.65 | Derived from `scipy.stats.norm.ppf(0.95)`. Assumes normally distributed forecast errors. | Medium — forecast errors may not be normally distributed for erratic or lumpy SKUs |
| 19 | Safety stock formula: `SS = Z × σ × √L` | Standard formula assuming independent, normally distributed demand errors over the lead time. Independence assumption may not hold for seasonal demand patterns. | Medium — should acknowledge the normality and independence assumptions |
| 20 | σ derived from per-SKU × warehouse forecast residuals on validation set | Most direct available estimate of per-SKU forecast uncertainty. In production, would be updated continuously as new actuals arrive. | Low — reasonable approximation |

---

## Known Limitations (for README)

1. **Category_019 not modeled** — a significant product category (33.9% of smooth + erratic training rows, 444 SKU × warehouse combinations) is excluded from the LightGBM pipeline. A production system would require dedicated forecasting treatment.
2. **Lead time is assumed, not observed** — actual replenishment lead times are not in the dataset. The 28-day assumption may not reflect real operational lead times across all SKU × warehouse combinations.
3. **Normal distribution assumed for safety stock** — the standard SS formula assumes normally distributed errors. Erratic and lumpy segments likely violate this assumption, particularly for high-CV² SKUs.
4. **No retraining cadence defined** — the model is trained once on historical data. A production system would need a defined retraining trigger (e.g., monthly, or drift-triggered).
5. **No cost data available** — holding cost and stockout cost are not in the dataset. Dollar-denominated inventory impact calculations would require external cost assumptions.
6. **Single model across all warehouses** — warehouse effects are captured via one-hot encoding. A production system might warrant warehouse-specific models given Whse_J's volume dominance.
7. **Syntetos-Boylan thresholds not tuned** — standard published thresholds used. Dataset-specific threshold optimization would require service-level validation data not available here.
