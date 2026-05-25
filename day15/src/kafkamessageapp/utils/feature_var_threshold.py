import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from kafkamessageapp.configurations.conf import KafkaConfig

# Load dataset
df = pd.read_csv(KafkaConfig.customer_data_path)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

target = "churned"

# Convert target boolean → integer
df[target] = df[target].astype(int)

print("Columns:")
print(df.columns.tolist())

# Separate X and y
X = df.drop(columns=[target])
y = df[target]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Missing value handling
X = X.fillna(X.median(numeric_only=True))

print("\nColumns before feature selection:")
print(X.columns.tolist())

# Apply Variance Threshold
selector = VarianceThreshold(threshold=1)

print("\nApplying Variance Threshold...")
print("Variance Threshold:", selector.threshold)
X_selected = selector.fit_transform(X)

print("\nVariance Threshold applied. Number of features selected:", X_selected.shape[1])
selected_columns = X.columns[
    selector.get_support()
]

removed_columns = X.columns[
    ~selector.get_support()
]

print("\nSelected Features:")
print(selected_columns.tolist())

print("\nRemoved Low Variance Features:")
print(removed_columns.tolist())

print("\nOriginal Shape:", X.shape)
print("Selected Shape:", X_selected.shape)