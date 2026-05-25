#dtale interactive data viewer
import dtale
import pandas as pd
from kafkamessageapp.configurations.conf import KafkaConfig

def dtale_demo(pizza_path):
    df = pd.read_csv(pizza_path)
    d=dtale.show(df,open_browser=True)
    print("D-Tale URL:")
    print(d._main_url)

    input("Press Enter to stop D-Tale...")

if __name__ == "__main__":
    pizza_path=KafkaConfig.pizza_path
    dtale_demo(pizza_path)

