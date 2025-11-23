# Model Comparison: OLS Naive vs Two-Way Fixed Effects

This page compares the performance of the OLS Naive model and the Two-Way Fixed Effects (TWFE) model. The goal is to show how model specification affects accuracy, error, and coefficient recovery when the true data generating process is known.

The two models compared here are:

1. OLS Naive Model  
   - No fixed effects  
   - Ignores competitor information  
   - Ignores lagged advertising  
   - Ignores unobserved store-level effects  

2. TWFE Champion Model  
   - Includes store and week fixed effects  
   - Includes competitor price and competitor count  
   - Includes lagged advertising  
   - Includes manager experience and vacancy status  
   - Recovers the true parameters more accurately  

## Summary Table

After running both models and computing metrics, fill in the table below.

| Metric | OLS Naive | TWFE Champion |
|--------|-----------|----------------|
| RMSE | | |
| R2 | | |
| Price absolute error | | |
| Relative price absolute error | | |
| Advertising absolute error | | |
| Advertising lag absolute error | | |
| Competitor count absolute error | | |
| Store size absolute error | | |
| Area income absolute error | | |
| Manager experience absolute error | | |
| Manager vacancy absolute error | | |

This comparison directly evaluates how each model performs relative to the true DGP.

## Visual Comparisons

Once the model scripts generate figures, add them here.

### Predicted vs True: OLS Naive

```markdown
![OLS Predicted vs True](../assets/figures/ols_pred_vs_true.png)
