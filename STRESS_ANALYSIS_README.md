# ND Hockey Stress Analysis: Game Performance Under Pressure

## Executive Summary

This analysis examines how multiple stress factors affect Notre Dame hockey game performance, including travel requirements, circadian disruption (jet lag), scheduling density, and logistics complexity. The study reveals counter-intuitive patterns: moderate stress levels actually correlate with better performance, and away games show better outcomes than home games.

## Analysis Overview

- **Total Games Analyzed:** 48
- **Time Period:** October 2024 - March 2025 (across 2025 and 2026 seasons)
- **Performance Metric:** Win/Loss outcome and goal differential
- **Stress Factors Studied:** Travel, circadian disruption, scheduling density, logistics complexity

## Stress Scoring System

Games are classified into stress bands based on a composite score:

- **Minimal (0–3):** Low stress - routine home games or nearby opponents
- **Low (4–7):** Moderate stress - some travel or challenging scheduling
- **Moderate (8–12):** High stress - long trips, cross-timezone games, or dense scheduling
- **High (13–16):** Very high stress - extensive travel combined with scheduling pressure
- **Very High (17+):** Extreme stress - multiple stress factors compounding

### Stress Components

1. **Circadian Load:** Jet lag and sleep disruption from travel (0-3 scale)
   - Cross-timezone flights increase significantly
   - Measures sleep phase shift and recovery needs

2. **Travel Load:** Physical and logistical burden of transportation (0-3 scale)
   - Flight vs. ground travel
   - Distance traveled
   - Trip duration complexity

3. **Logistics Load:** Off-ice operational challenges (0-3 scale)
   - Game scheduling requirements
   - Hotel/venue transitions
   - Administrative complexity

4. **Density Load:** Scheduling intensity (0-3 scale)
   - Back-to-back games
   - Short rest periods between games
   - Tournament/multi-game events

## Key Findings

### 1. Win Rate by Stress Level (Counter-Intuitive Result)

| Stress Band | Win Rate | Games | Avg Stress | Performance |
|---|---|---|---|---|
| Minimal (0–3) | 29.4% | 34 | 1.85 | Below average |
| Low (4–7) | 41.7% | 12 | 5.08 | **Optimal** ✓ |
| Moderate (8–12) | 50.0% | 2 | 9.00 | High (limited data) |

**Key Insight:** The team performs BEST at moderate stress levels (4-7), not at minimal stress. This suggests:
- A "Yerkes-Dodson Effect" - optimal performance at moderate arousal
- Minimal stress may reduce focus/intensity
- Extreme stress (8+) may impair performance

### 2. Travel Impact (Surprising Finding)

| Travel Type | Win Rate | Avg Stress | Games |
|---|---|---|---|
| **Flight Required** | **44.4%** | 4.94 | 18 |
| **No Flight** | **26.7%** | 1.77 | 30 |

**Key Insight:** Games requiring air travel show HIGHER win rates. Possible explanations:
- Selection bias: Flights are to more important/marquee matchups
- Preparation effect: Teams prepare more intensively for distant opponents
- Quality of opposition: Distant opponents may be stronger, leading to tighter games

### 3. Home vs. Away (Unexpected Pattern)

| Location | Win Rate | Avg Stress | Games | Goal Diff |
|---|---|---|---|---|
| **Home** | **28.0%** | 1.76 | 25 | -1.08 |
| **Away** | **39.1%** | 4.26 | 23 | -0.39 |

**Key Insight:** Away games show a **+11.1 percentage point advantage**. This is counter to typical home-ice advantage patterns and suggests:
- The team performs better when "challenged" by travel/opponent strength
- Low stress at home may reduce competitive intensity
- Road games may involve higher-quality opponents, driving better performance

### 4. Stress Component Correlations

Ranked by strength of correlation with overall stress score:

1. **Logistics Load: 0.769** ← Strongest predictor
2. **Circadian Load: 0.630**
3. **Travel Load: 0.603**
4. **Density Load: 0.569**

**Finding:** Scheduling complexity and logistical challenges matter more than pure travel distance.

### 5. Stress Factors and Winning Correlation

| Factor | Correlation with Wins |
|---|---|
| Goal Differential | **0.859** ← Strongest |
| Circadian Load | 0.245 |
| Shot Differential | 0.210 |
| Logistics Load | 0.165 |
| Stress Score | 0.075 |
| Travel Load | 0.048 |
| Density Load | **-0.120** ← Negative |

**Key Finding:** Goal differential is the dominant factor (teams that score more win), while stress itself is a weak predictor. Scheduling density actually shows a slight negative correlation.

### 6. Stress by Opponent

Top stress-generating opponents:

| Opponent | Avg Stress | Win Rate | Games |
|---|---|---|---|
| Harvard/Boston University | 9.0 | 50.0% | 2 |
| Arizona State/Quinnipiac | 5.0 | 0% | 1 |
| **Minnesota** | 4.3 | 42.9% | 7 |
| St Lawrence/Clarkson | 4.0 | **100%** | 2 |
| Merrimack/Boston College | 3.5 | 50.0% | 2 |

**Finding:** High-stress opponents are often important matchups. St. Lawrence shows perfect record despite moderate stress.

## Statistical Testing

**T-test Analysis (Wins vs. Losses):**
- Win games mean stress: 3.19
- Loss games mean stress: 2.84
- Statistical significance: p = 0.612 (NOT statistically significant)

**Conclusion:** Stress level alone is not a significant predictor of game outcomes. Other factors (opponent quality, team composition, execution) matter much more.

## Visualizations Included

The `stress_analysis.ipynb` notebook contains:

1. **Win Rate by Stress Level** - Bar chart showing performance across stress bands
2. **Travel Impact Analysis** - Comparison of flights vs. no flights on performance
3. **Home vs. Away Performance** - Location impact on win rates
4. **Correlation Heatmap** - Relationships between all stress factors and performance metrics
5. **Stress Distribution by Outcome** - Violin plots comparing win vs. loss games
6. **Temporal Trend Analysis** - Stress scores over the season (with game outcomes marked)
7. **Opponent Difficulty Analysis** - Stress vs. win rate scatter plot for each opponent

## Strategic Recommendations

### 1. **Reframe "Stress" as "Challenge"**
- Moderate stress/challenge correlates with better performance
- Don't over-protect the team from difficult matchups
- Embrace "road warrior" mentality for away games

### 2. **Optimize Home Game Preparation**
- 11% win rate disadvantage at home suggests intensity issues
- Implement focused preparation/intensity protocols for home games
- Use home games strategically (e.g., against lower-ranked teams)

### 3. **Leverage Travel Psychology**
- Higher win rates with air travel suggest preparation benefits
- Use travel as motivator for higher-stakes matchups
- Build "road game competence" as competitive advantage

### 4. **Manage Scheduling Density**
- Slight negative correlation with density load (-0.120)
- Prioritize rest quality over rest quantity
- Consider cumulative back-to-backs over multi-week periods

### 5. **Focus on Opponent Preparation**
- Certain opponents (Minnesota, Penn State) drive stress
- Develop opponent-specific preparation protocols
- Ensure adequate recovery after high-stress matchups

### 6. **Monitor, Don't Over-Manage**
- Stress level itself is weakly correlated with outcomes
- Don't use stress as excuse for poor performance
- Focus on controllables: execution, physical conditioning, mental preparation

## Data Insights

### Season Distribution
- **2025 Season:** 38 games analyzed
- **2026 Season:** 10 games analyzed
- Coverage spans regular season and some tournament play

### Stress Band Distribution
- 34 games in Minimal stress (70.8%)
- 12 games in Low stress (25.0%)
- 2 games in Moderate stress (4.2%)
- No games in High or Very High stress categories

**Note:** Most ND games are low-stress due to the team's consistent home-heavy schedule with regional opponents.

### Performance Summary
- **Overall Win Rate:** 33.3% (16 wins, 32 losses)
- **Average Stress Score:** 2.96 (reflecting mostly minimal-stress games)
- **Stress Range:** 0-10 (out of potential 0-16)
- **Best Performance Month:** Varies by stress level

## Technical Notes

### Data Preparation
- Source: `stress_df.csv` with 48 games
- Features: 28 variables including stress metrics, travel indicators, performance outcomes
- Stress scores calculated using weighted component model

### Analysis Tools
- **Pandas/NumPy:** Data manipulation and calculations
- **Plotly:** Interactive visualizations
- **SciPy:** Statistical testing (t-tests, correlations)
- **Seaborn/Matplotlib:** Statistical visualization support

### Methodology
- Correlation analysis for factor relationships
- Categorical grouping for stress band analysis
- T-tests for outcome comparison
- Time series analysis for seasonal trends

## Files in Repository

- **`stress_analysis.ipynb`** - Complete interactive Jupyter notebook with all visualizations
- **`stress_df.csv`** - Raw data with all stress metrics (48 games)
- **`hockey_performance_analysis.py`** - Original travel impact analysis
- **`STRESS_ANALYSIS_README.md`** - This file
- **`README.md`** - Original performance analysis documentation

## How to Use the Notebook

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn jupyter
```

### Running the Analysis
```bash
# Navigate to repository
cd hockey3

# Open the notebook
jupyter notebook stress_analysis.ipynb
```

### Notebook Structure
1. **Libraries & Setup** - Configuration for analysis
2. **Data Loading** - Load and explore stress_df.csv
3. **Statistical Summary** - Win rates, stress metrics, performance by category
4. **Stress Components** - Analysis of individual stress factors
5. **Visualizations** - 7 detailed charts with insights
6. **Key Insights Summary** - Executive findings and recommendations

## Conclusions

### Main Takeaways

1. **Moderate stress ≠ Performance loss** - The team actually performs better with moderate challenge levels
2. **Travel selection bias** - Air travel correlates with better performance (likely due to matchup importance)
3. **Away games are strength** - Contrary to typical home advantage, ND performs better away
4. **Stress is not destiny** - Stress level itself poorly predicts outcomes; execution matters more
5. **Scheduling density has minimal impact** - Even back-to-backs don't strongly affect performance

### Areas for Further Investigation

1. Opponent strength ratings (strength-of-schedule adjustment)
2. Roster composition and injuries during stress periods
3. Goaltender performance under different stress levels
4. Psychological/fatigue data collection
5. Multi-season longitudinal analysis

## Questions & Analysis Requests

For questions about this analysis or to request additional investigations:
- Contact: [Analysis team]
- Repository: https://github.com/perrygregg4/hockey3
- Date: December 11, 2025

---

**Analysis Date:** December 11, 2025  
**Dataset:** ND Hockey 2024-2025 & 2025-2026 Seasons  
**Methodology:** Correlational & Categorical Analysis  
**Key Metric:** Game Outcome (Win/Loss) vs. Stress Factors
