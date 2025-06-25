import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Excel file
file_path = 'train_data.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Extract inputs and outputs
X = df.iloc[:, [0, 1]]
y = df.iloc[:, 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, 'linear_model.joblib')
print("Model saved as 'linear_model.joblib'")
