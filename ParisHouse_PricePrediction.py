# Importing necessary libraries
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Read Dataset
Dataset = pd.read_csv("/home/samreen-software/Desktop/ML/ParisHouse_pricePrediction/ParisHousing.csv")

# Show Dataset 
print(Dataset.head(5))

# Number of rows and columns in dataset
print("No.of Rows: ", Dataset.shape[0])
print("No.of Columns: ", Dataset.shape[1])

# Pre-processing
for row in Dataset:
        if row == None:
            print("Null value detected in row:", row)

print("No null value")


# Split data into training and testing sets
X = Dataset.drop('price', axis=1)
y = Dataset['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create Random Forest model with regularization hyperparameters
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42)

# Train model on training set
rf.fit(X_train, y_train)

# Evaluate model on testing set
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

