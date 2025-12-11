import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats

# Load data
df = pd.read_csv('/Users/perrygregg/Downloads/model_df.csv')

print("=" * 80)
print("ND HOCKEY PERFORMANCE ANALYSIS: TRAVEL IMPACT")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Data overview
print("\n" + "=" * 80)
print("DATA OVERVIEW")
print("=" * 80)
print(df.head(10))
print("\nColumn info:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Define features and target
# Features: Travel factors and contextual info
features = ['travel_hours', 'travel_to_hours', 'travel_from_hours', 
            'is_long_trip', 'is_flight', 'is_cross_timezone', 
            'days_since_last', 'is_home', 'month', 'day_of_week']

# Target: goal_diff (ND goals scored - opponent goals)
# We'll analyze goal_diff which represents scoring differential
target = 'goal_diff'

# Create a clean dataset (remove rows with missing target or critical features)
df_clean = df[features + [target]].dropna(subset=[target])

print(f"\nClean dataset shape: {df_clean.shape}")
print(f"Target variable (goal_diff) statistics:")
print(df_clean[target].describe())

# Prepare data for modeling
X = df_clean[features].copy()
y = df_clean[target].copy()

# Check for any remaining NaN in features
print(f"\nMissing values in features: {X.isnull().sum().sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# MULTIPLE LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("MULTIPLE LINEAR REGRESSION ANALYSIS")
print("=" * 80)

# Standardize features for MLR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit MLR model
mlr_model = LinearRegression()
mlr_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_mlr = mlr_model.predict(X_train_scaled)
y_test_pred_mlr = mlr_model.predict(X_test_scaled)

# Metrics
mlr_train_mse = mean_squared_error(y_train, y_train_pred_mlr)
mlr_test_mse = mean_squared_error(y_test, y_test_pred_mlr)
mlr_train_rmse = np.sqrt(mlr_train_mse)
mlr_test_rmse = np.sqrt(mlr_test_mse)
mlr_train_r2 = r2_score(y_train, y_train_pred_mlr)
mlr_test_r2 = r2_score(y_test, y_test_pred_mlr)
mlr_train_mae = mean_absolute_error(y_train, y_train_pred_mlr)
mlr_test_mae = mean_absolute_error(y_test, y_test_pred_mlr)

print("\nMultiple Linear Regression Results:")
print(f"  Train RMSE: {mlr_train_rmse:.4f}")
print(f"  Test RMSE:  {mlr_test_rmse:.4f}")
print(f"  Train MAE:  {mlr_train_mae:.4f}")
print(f"  Test MAE:   {mlr_test_mae:.4f}")
print(f"  Train R²:   {mlr_train_r2:.4f}")
print(f"  Test R²:    {mlr_test_r2:.4f}")

print("\nRegression Coefficients (standardized):")
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': mlr_model.coef_,
    'Abs_Coefficient': np.abs(mlr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)
print(coef_df)
print(f"\nIntercept: {mlr_model.intercept_:.4f}")

# Statistical significance interpretation
print("\n** Note: With limited data (n=38), p-values may not be reliable **")

# ============================================================================
# GRADIENT BOOSTING MODEL (Similar to XGBoost)
# ============================================================================
print("\n" + "=" * 80)
print("GRADIENT BOOSTING REGRESSION MODEL ANALYSIS")
print("=" * 80)

# Gradient Boosting with similar hyperparameters to XGBoost
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    subsample=0.8
)

gb_model.fit(X_train, y_train)

# Predictions
y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)

# Metrics
gb_train_mse = mean_squared_error(y_train, y_train_pred_gb)
gb_test_mse = mean_squared_error(y_test, y_test_pred_gb)
gb_train_rmse = np.sqrt(gb_train_mse)
gb_test_rmse = np.sqrt(gb_test_mse)
gb_train_r2 = r2_score(y_train, y_train_pred_gb)
gb_test_r2 = r2_score(y_test, y_test_pred_gb)
gb_train_mae = mean_absolute_error(y_train, y_train_pred_gb)
gb_test_mae = mean_absolute_error(y_test, y_test_pred_gb)

print("\nGradient Boosting Results:")
print(f"  Train RMSE: {gb_train_rmse:.4f}")
print(f"  Test RMSE:  {gb_test_rmse:.4f}")
print(f"  Train MAE:  {gb_train_mae:.4f}")
print(f"  Test MAE:   {gb_test_mae:.4f}")
print(f"  Train R²:   {gb_train_r2:.4f}")
print(f"  Test R²:    {gb_test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nGradient Boosting Feature Importance:")
print(feature_importance)

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Metric': ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'Train R²', 'Test R²'],
    'Multiple Linear Regression': [mlr_train_rmse, mlr_test_rmse, mlr_train_mae, mlr_test_mae, mlr_train_r2, mlr_test_r2],
    'Gradient Boosting': [gb_train_rmse, gb_test_rmse, gb_train_mae, gb_test_mae, gb_train_r2, gb_test_r2]
})

print(comparison_df)

# ============================================================================
# KEY INSIGHTS - TRAVEL IMPACT
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: TRAVEL IMPACT ON ND HOCKEY PERFORMANCE")
print("=" * 80)

print("\n1. TRAVEL FACTORS ANALYSIS:")
print(f"   - Average travel hours: {df_clean['travel_hours'].mean():.2f}")
print(f"   - Long trip games: {df_clean['is_long_trip'].sum()} games")
print(f"   - Cross-timezone games: {df_clean['is_cross_timezone'].sum()} games")
print(f"   - Home games: {df_clean['is_home'].sum()} games")

# Average performance by travel factors
print("\n2. AVERAGE GOAL DIFFERENTIAL BY TRAVEL FACTOR:")
print(f"   By Home/Away:")
print(f"     Home games:  {df_clean[df_clean['is_home']==1]['goal_diff'].mean():.2f}")
print(f"     Away games:  {df_clean[df_clean['is_home']==0]['goal_diff'].mean():.2f}")

print(f"\n   By Trip Length:")
print(f"     Short trip:  {df_clean[df_clean['is_long_trip']==0]['goal_diff'].mean():.2f}")
print(f"     Long trip:   {df_clean[df_clean['is_long_trip']==1]['goal_diff'].mean():.2f}")

print(f"\n   By Timezone Change:")
print(f"     Same timezone: {df_clean[df_clean['is_cross_timezone']==0]['goal_diff'].mean():.2f}")
print(f"     Cross timezone: {df_clean[df_clean['is_cross_timezone']==1]['goal_diff'].mean():.2f}")

# MLR interpretation
print("\n3. MULTIPLE LINEAR REGRESSION INTERPRETATION (Standardized Coefficients):")
print("   Positive coefficient = factor associated with MORE goals scored")
print("   Negative coefficient = factor associated with FEWER goals scored")
print("\n   Top positive factors:")
top_pos = coef_df[coef_df['Coefficient'] > 0].head(3)
for idx, row in top_pos.iterrows():
    print(f"     {row['Feature']}: {row['Coefficient']:.4f}")
print("\n   Top negative factors:")
top_neg = coef_df[coef_df['Coefficient'] < 0].head(3)
for idx, row in top_neg.iterrows():
    print(f"     {row['Feature']}: {row['Coefficient']:.4f}")

print("\n4. XGBOOST FEATURE IMPORTANCE:")
print("   Top 5 most important features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"     {row['Feature']}: {row['Importance']:.4f}")

print("\n5. MODEL PERFORMANCE SUMMARY:")
print(f"   MLR Test R²: {mlr_test_r2:.4f} (explains {mlr_test_r2*100:.1f}% of variance)")
print(f"   GB Test R²: {gb_test_r2:.4f} (explains {gb_test_r2*100:.1f}% of variance)")
better_model = "Gradient Boosting" if gb_test_r2 > mlr_test_r2 else "Multiple Linear Regression"
print(f"   Better performing model: {better_model}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Feature Importance Comparison
ax1 = axes[0, 0]
importance_plot_df = feature_importance.head(8)
ax1.barh(importance_plot_df['Feature'], importance_plot_df['Importance'], color='steelblue')
ax1.set_xlabel('Gradient Boosting Feature Importance')
ax1.set_title('Top 8 Most Important Features (Gradient Boosting)')
ax1.invert_yaxis()

# 2. MLR Coefficients
ax2 = axes[0, 1]
coef_plot_df = coef_df.head(8)
colors = ['green' if x > 0 else 'red' for x in coef_plot_df['Coefficient']]
ax2.barh(coef_plot_df['Feature'], coef_plot_df['Coefficient'], color=colors)
ax2.set_xlabel('Standardized Coefficient')
ax2.set_title('Linear Regression Coefficients (Top 8)')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.invert_yaxis()

# 3. Actual vs Predicted - MLR
ax3 = axes[1, 0]
ax3.scatter(y_test, y_test_pred_mlr, alpha=0.6, color='steelblue', label='MLR')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
ax3.set_xlabel('Actual Goal Differential')
ax3.set_ylabel('Predicted Goal Differential')
ax3.set_title(f'Linear Regression: Actual vs Predicted (R² = {mlr_test_r2:.3f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Actual vs Predicted - XGBoost
ax4 = axes[1, 1]
ax4.scatter(y_test, y_test_pred_gb, alpha=0.6, color='darkgreen', label='Gradient Boosting')
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
ax4.set_xlabel('Actual Goal Differential')
ax4.set_ylabel('Predicted Goal Differential')
ax4.set_title(f'Gradient Boosting: Actual vs Predicted (R² = {gb_test_r2:.3f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/perrygregg/Downloads/hockey_performance_models.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: hockey_performance_models.png")

# Additional analysis visualization
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# 1. Goal differential by home/away
ax1 = axes2[0, 0]
home_away_data = [df_clean[df_clean['is_home']==1]['goal_diff'], 
                   df_clean[df_clean['is_home']==0]['goal_diff']]
ax1.boxplot(home_away_data, labels=['Home', 'Away'])
ax1.set_ylabel('Goal Differential')
ax1.set_title('Goal Differential: Home vs Away')
ax1.grid(True, alpha=0.3)

# 2. Goal differential by trip length
ax2 = axes2[0, 1]
trip_data = [df_clean[df_clean['is_long_trip']==0]['goal_diff'], 
             df_clean[df_clean['is_long_trip']==1]['goal_diff']]
ax2.boxplot(trip_data, labels=['Short Trip', 'Long Trip'])
ax2.set_ylabel('Goal Differential')
ax2.set_title('Goal Differential: Trip Length Impact')
ax2.grid(True, alpha=0.3)

# 3. Goal differential by timezone
ax3 = axes2[1, 0]
tz_data = [df_clean[df_clean['is_cross_timezone']==0]['goal_diff'], 
           df_clean[df_clean['is_cross_timezone']==1]['goal_diff']]
ax3.boxplot(tz_data, labels=['Same Timezone', 'Cross Timezone'])
ax3.set_ylabel('Goal Differential')
ax3.set_title('Goal Differential: Timezone Change Impact')
ax3.grid(True, alpha=0.3)

# 4. Travel hours vs goal differential
ax4 = axes2[1, 1]
non_zero_travel = df_clean[df_clean['travel_hours'] > 0]
ax4.scatter(non_zero_travel['travel_hours'], non_zero_travel['goal_diff'], alpha=0.6)
z = np.polyfit(non_zero_travel['travel_hours'], non_zero_travel['goal_diff'], 1)
p = np.poly1d(z)
x_line = np.linspace(non_zero_travel['travel_hours'].min(), non_zero_travel['travel_hours'].max(), 100)
ax4.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend line')
ax4.set_xlabel('Travel Hours')
ax4.set_ylabel('Goal Differential')
ax4.set_title('Travel Hours Impact on Performance')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/perrygregg/Downloads/hockey_performance_travel_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: hockey_performance_travel_analysis.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
