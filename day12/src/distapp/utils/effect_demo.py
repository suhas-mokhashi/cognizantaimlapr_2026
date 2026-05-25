#calculate effect using cohen's d for medicine effectiveness using medicine_effect.csv
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from distapp.configurations.config import Config

def calculate_effectiveness():
    config = Config()
    df = pd.read_csv(config.effectiveness_path)
    encoder = LabelEncoder()
    df['recovery_speed'] = encoder.fit_transform(df['recovery_speed'])
    group1 = df[df['medicine_name'] == 'Metformin']['recovery_speed']
    group2 = df[df['medicine_name'] == 'Glimepiride']['recovery_speed']    
    print(group1.to_list())
    print(group2.to_list())
    #apply cohen's d formula
    mean1 = group1.mean()
    mean2 = group2.mean()
    std1 = group1.std()
    std2 = group2.std()
    n1 = len(group1)
    n2 = len(group2)
    pooled_std = (((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)) ** 0.5
    cohen_d = -(mean1 - mean2) / pooled_std
    print("Cohen's d:", cohen_d)

if __name__ == "__main__":
    calculate_effectiveness()