import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_students = 1500 # Increased sample size

print("Generating 'Dynamic' Naija Data ")

# 1. Generate only RELEVANT features
data = {
    # Academic History (The biggest predictors)
    'G1': np.random.uniform(1.0, 5.0, n_students).round(2), # Year 1 GPA
    'G2': np.random.uniform(1.0, 5.0, n_students).round(2), # Year 2 GPA
    'failures': np.random.choice([0, 1, 2, 3], n_students, p=[0.85, 0.10, 0.04, 0.01]), # Carryovers
    'absences': np.random.randint(0, 20, n_students), # Classes missed
    
    # Effort & Lifestyle
    'studytime': np.random.randint(1, 5, n_students), # 1=Low, 4=High
    'health': np.random.randint(1, 6, n_students),    # 1=Poor, 5=Excellent
    'goout': np.random.randint(1, 6, n_students),     # Social life frequency
    'higher': np.random.choice([0, 1], n_students),   # 1=Yes (Masters plan)
    'activities': np.random.choice([0, 1], n_students) # 1=Yes (Sports/SUG)
}

df = pd.DataFrame(data)

# 2. DYNAMIC LOGIC: Calculate Final CGPA (G3) based on these habits
# We create a formula that mimics real life:
# Final = (Past Performance) + (Effort) - (Distractions)

# A. Start with average of previous GPAs (Weighted heavily on most recent)
base_gpa = (df['G1'] * 0.4) + (df['G2'] * 0.6)

# B. Apply "The Hustle" modifiers
# - Study Time: +0.15 GPA per level (Max +0.6)
# - Health: +0.05 GPA per level (Energy to study)
# - Activities: +0.1 GPA (Discipline bonus)
# - Higher Education Plan: +0.2 GPA (Motivation bonus)
bonus = (df['studytime'] * 0.15) + (df['health'] * 0.05) + (df['activities'] * 0.1) + (df['higher'] * 0.2)

# C. Apply "The Wahala" (Penalty) modifiers
# - Failures: -0.4 GPA per carryover (Hard to recover)
# - Absences: -0.02 GPA per class missed
# - Go Out: -0.08 GPA per level (Too much partying hurts)
penalty = (df['failures'] * 0.4) + (df['absences'] * 0.02) + (df['goout'] * 0.08)

# D. Calculate G3
df['G3'] = base_gpa + bonus - penalty

# E. Add a little random "Life Happens" noise (+/- 0.15)
noise = np.random.normal(0, 0.15, n_students)
df['G3'] = df['G3'] + noise

# 3. Clean Up
# Clip scores to strict 0.0 - 5.0 range
df['G3'] = df['G3'].clip(0.0, 5.0).round(2)

# Save
df.to_csv('nigerian_students_dynamic.csv', index=False)
print(" Generated 'nigerian_students_dynamic.csv'.")
print("    Optimized for simplicity and accuracy.")