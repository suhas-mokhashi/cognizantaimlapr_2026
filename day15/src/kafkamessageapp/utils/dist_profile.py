import pandas as pd
from matplotlib import pyplot as plt
from kafkamessageapp.configurations.conf import KafkaConfig
if __name__ == "__main__":
    df = pd.read_csv(KafkaConfig.pizza_path)
    print(df.describe())
    df['pizza_rate'].hist()
    plt.show()

