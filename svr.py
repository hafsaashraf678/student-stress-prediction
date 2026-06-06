import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import numpy as np

# Load dataset
df = pd.read_csv("university_student_stress_dataset.csv")

# Remove leakage columns
df = df.drop(columns=["Anxiety_Level", "Age", "Stress_Level"])

# Text columns encode karo
le = LabelEncoder()
df["Gender"]            = le.fit_transform(df["Gender"])
df["Physical_Exercise"] = le.fit_transform(df["Physical_Exercise"])
df["Tuition"]           = le.fit_transform(df["Tuition"])

df["Family_Income_Level"] = df["Family_Income_Level"].map({"Low": 0, "Medium": 1, "High": 2})
df = pd.get_dummies(df, columns=["University_Type"])

# Features and target
X = df.drop(columns=["Stress_Score"])
y = df["Stress_Score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize (SVR ke liye zaroori hai)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train
model = SVR(kernel='rbf', C=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R² Score: {:.2f}".format(r2_score(y_test, y_pred)))
print("MAE     : {:.2f}".format(mean_absolute_error(y_test, y_pred)))
print("RMSE    : {:.2f}".format(np.sqrt(((y_test - y_pred)**2).mean())))

def get_stress_level(score):
	if score <= 10:
		return "Low"
	elif score <= 20:
		return "Medium"
	else:
		return "High"

y_test_level = y_test.apply(get_stress_level)
y_pred_level = [get_stress_level(s) for s in y_pred]
print("Cat Acc  : {:.2f}%".format(accuracy_score(y_test_level, y_pred_level) * 100))