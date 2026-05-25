import pandas as pd
from kafkamessageapp.configurations.conf import KafkaConfig

# Load data
df = pd.read_csv(KafkaConfig.customer_data_path)

# clean column names
df.columns = df.columns.str.strip().str.lower()

print("Columns:")
print(df.columns.tolist())

target = "churned"

# convert boolean target to integer
df[target] = df[target].astype(int)

print("\nTarget unique values:")
print(df[target].unique())

# numeric columns
numeric_df = df.select_dtypes(include=["number"])

print("\nNumeric Columns:")
print(numeric_df.columns.tolist())

# correlation with target
corr = numeric_df.corr()[target]

print("\nCorrelation with target:")
print(corr.sort_values(ascending=False))

# feature selection
selected_features = (
    corr[corr > 0.05]
    .index
    .tolist()
)

# remove target itself
if target in selected_features:
    selected_features.remove(target)

print("\nSelected Features:")
print(selected_features)