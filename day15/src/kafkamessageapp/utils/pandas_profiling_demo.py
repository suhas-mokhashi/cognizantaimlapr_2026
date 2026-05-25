import pandas as pd
from pandas_profiling import ProfileReport

from kafkamessageapp.configurations.conf import KafkaConfig


df = pd.read_csv(KafkaConfig.pizza_path)

profile = ProfileReport(
    df,
    title="Pizza Profiling Report",
    explorative=True
)

profile.to_file("pizza_report.html")

print("Report Generated")