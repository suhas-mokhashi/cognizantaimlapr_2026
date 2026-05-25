import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from kafkamessageapp.configurations.conf import KafkaConfig

# Load dataset
df = pd.read_csv(KafkaConfig.customer_data_path)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

target = "churned"

# Convert boolean target → integer
df[target] = df[target].astype(int)

# Split features and target
X = df.drop(columns=[target])
y = df[target]

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
X = X.fillna(X.median(numeric_only=True))

print("Columns before feature selection:")
print(X.columns.tolist())

# ==================================================
# SelectKBest
# ==================================================
selector = SelectKBest(
    score_func=f_classif,
    k=5
)

X_selected = selector.fit_transform(X, y)

# Selected feature names
selected_features = X.columns[
    selector.get_support()
]

# Feature scores
scores_df = pd.DataFrame({
    "feature": X.columns,
    "score": selector.scores_
})

scores_df = scores_df.sort_values(
    by="score",
    ascending=False
)

print("\nFeature Scores:")
print(scores_df)

print("\nSelected Features:")
print(selected_features.tolist())

print("\nOriginal Shape:", X.shape)
print("Selected Shape:", X_selected.shape)