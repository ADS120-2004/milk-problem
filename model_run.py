import joblib
import pandas as pd
model = joblib.load('linear_model.joblib')
print("Enter the fat and SNF: ", end="")
lst = [input() for i in range(2)]
new_data = pd.DataFrame([lst])
predictions = model.predict(new_data)

print("Predictions:", predictions)
