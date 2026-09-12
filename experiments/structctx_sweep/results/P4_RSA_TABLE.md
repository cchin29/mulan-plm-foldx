# P4 — RSA MaxASA-table sensitivity (TEM-1 P62593 vs 1BTL)

| table | SS3 | RSA Pearson r | RSA MAE | n | >0.85 gate |
|---|---|---|---|---|---|
| tien2013_theoretical (default) | 0.9962 | 0.9841 | 0.0266 | 263 | ✅ |
| tien2013_empirical | 0.9962 | 0.9842 | 0.0279 | 263 | ✅ |
| sander_rost1994 | 0.9962 | 0.9841 | 0.0315 | 263 | ✅ |

**Verdict:** all 3/3 tables clear the >0.85 gate — SS3 is table-invariant and RSA r barely moves (rescale, not reorder), confirming the choice is a reporting recalibration. Highest r: **tien2013_empirical** (0.9842). Keep the Tien-2013-theoretical default (D3).
