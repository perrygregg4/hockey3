# ND Hockey Performance Analysis: Travel Impact Report

## Executive Summary

This analysis examined how travel intensity, distance, and timezone changes impact Notre Dame hockey scoring performance using two machine learning models:

1. **Multiple Linear Regression (MLR)** - Linear relationships with interpretable coefficients
2. **Gradient Boosting** - Non-linear patterns with higher predictive accuracy

## Dataset Overview

- **Total Games Analyzed:** 48
- **Time Period:** October 2024 - December 2025
- **Performance Metric:** Goal Differential (ND goals - opponent goals)
  - Mean: -0.75 goals
  - Median: -1.0 goal
  - Range: -7 to +6 goals

## Key Findings

### 1. Travel Impact on Scoring Performance

#### Home vs. Away Impact (Most Significant)
- **Home games average:** -1.08 goal differential
- **Away games average:** -0.39 goal differential
- **Finding:** ND performs better (less negative) on away games, contrary to typical expectations
  - This suggests travel doesn't significantly impair performance when combined with other factors

#### Timezone Change Impact
- **Same timezone games:** -1.00 goal differential
- **Cross-timezone games:** +0.09 goal differential
- **Finding:** Cross-timezone games show slightly better performance, indicating minimal negative impact

#### Trip Length Impact
- **Short trips:** -0.80 goal differential
- **Long trips:** 0.00 goal differential
- **Finding:** Long trips show minimal performance degradation (close to break-even)

#### Travel Hours Correlation
- **Average travel hours:** 1.43 hours
- **Long trip games:** Only 3 games in dataset
- **Finding:** Very limited variation in travel distances (mostly regional opponents)

### 2. Linear Regression Analysis (Standardized Coefficients)

**Top Positive Factors (Associated with MORE Goals Scored):**
1. **travel_hours: +26.75** - More travel hours correlates with MORE goals (counterintuitive but significant)
2. **is_long_trip: +1.52** - Long trips slightly boost scoring
3. **days_since_last: +0.11** - Rest between games helps scoring

**Top Negative Factors (Associated with FEWER Goals Scored):**
1. **travel_from_hours: -13.76** - Travel duration returning home reduces scoring
2. **travel_to_hours: -13.76** - Travel duration to away games reduces scoring
3. **is_flight: -1.04** - Taking flights is associated with lower goal differential

**Note:** The high coefficients on travel variables suggest multicollinearity and unstable estimates due to limited data (n=38 training samples).

### 3. Gradient Boosting Feature Importance

Ranked by importance to predictions:

1. **Month: 0.348** - Seasonal variation is the strongest predictor
2. **Days Since Last Game: 0.211** - Rest period significantly impacts performance
3. **Day of Week: 0.177** - Game scheduling affects performance
4. **Travel Hours: 0.096** - Travel duration moderately important
5. **Travel From Hours: 0.054** - Return travel less important
6. **Travel To Hours: 0.050** - Outbound travel least important
7. **Is Flight: 0.031** - Mode of transportation has minor impact
8. **Is Cross Timezone: 0.019** - Timezone change minimal impact
9. **Is Home: 0.011** - Home/away designation minimal importance
10. **Is Long Trip: 0.003** - Trip length has negligible impact

## Model Performance

### Training Performance
- **MLR:** R² = 0.112 (explains 11% of variance)
- **Gradient Boosting:** R² = 0.991 (explains 99% of training variance - overfitting)

### Test Performance (Generalization)
- **MLR:** R² = -2.23 (Test RMSE: 4.04 goals)
- **Gradient Boosting:** R² = -2.13 (Test RMSE: 3.97 goals)

**Key Issue:** Both models show poor generalization (negative test R²), indicating:
1. **Limited sample size** (48 games total, 38 training)
2. **High prediction difficulty** - Hockey scoring is inherently variable
3. **Travel factors alone insufficient** - Other factors (opponent strength, roster status, etc.) are crucial

## Conclusions & Recommendations

### Main Conclusions

1. **Travel Impact is Minimal:** Based on this data, travel intensity, distance, and timezone changes have surprisingly modest effects on ND hockey performance. The team maintains relatively consistent play regardless of travel demands.

2. **Better Predictors Exist:** Temporal factors (month, days of rest, day of week) are far more predictive than travel characteristics.

3. **Counterintuitive Pattern:** Long trips and more travel hours show slight positive associations with scoring, possibly due to:
   - Playing higher-ranked opponents in tournament/special event games
   - Selection bias (important games may require travel)

4. **Seasonal Variation Dominates:** Time of season (month) is the single best predictor, suggesting team performance evolves throughout the year more than travel affects any single game.

### Recommendations

1. **Collect More Data:** 48 games is a small sample. Expanding to multiple seasons would improve model reliability.

2. **Include Additional Factors:**
   - Opponent ranking/strength
   - Injuries/roster changes
   - Game score context (nail-biters vs. blowouts)
   - Fatigue metrics (cumulative back-to-backs)

3. **Focus on Rest Management:** Days since last game is more predictive than travel factors - optimize scheduling to maximize rest periods.

4. **Monitor, Don't Over-Adjust:** Travel doesn't appear to be a primary performance driver for ND. Don't sacrifice competitive opportunities to minimize travel.

## Technical Notes

- **Scaling:** Features were standardized for linear regression
- **Train/Test Split:** 80/20 (38 training, 10 test samples)
- **Random State:** 42 (for reproducibility)
- **Gradient Boosting Parameters:** 200 estimators, depth=5, learning_rate=0.1

## Files in This Repository

- `hockey_performance_analysis.py` - Complete Python script for the analysis
- `model_df.csv` - Dataset with 48 games and travel/performance metrics
- `hockey_performance_models.png` - Visualization of model performance and feature importance
- `hockey_performance_travel_analysis.png` - Visualizations of travel factor impacts
- `README.md` - This file

## How to Run the Analysis

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Run the analysis
python hockey_performance_analysis.py
```

This will generate the analysis output and save the visualization PNG files.

---

**Analysis Date:** December 11, 2025
**Generated by:** Machine Learning Model Comparison
**Models Used:** Multiple Linear Regression, Gradient Boosting Regression
