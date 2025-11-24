import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

print("--- 🚀 TRAINING SIMPLIFIED MODEL ---")

# 1. Load
if not os.path.exists('nigerian_students_dynamic.csv'):
    print("❌ Error: Run 'generate_data.py' first!")
    exit()

df = pd.read_csv('nigerian_students_dynamic.csv')

# 2. Prepare
target = 'G3'
X = df.drop([target], axis=1)
y = df[target]

# 3. Train
print("🧠 Learning relationships...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. Save
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump({'model': model, 'features': X.columns.tolist()}, 'model/model.pkl')
print(f"🎉 SUCCESS! Accuracy: {model.score(X_test, y_test):.2f}")