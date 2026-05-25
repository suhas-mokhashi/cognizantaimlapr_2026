#create hypothesis test to prove ai assisted tools better than human using aivshuman.csv file
import pandas as pd
from scipy import stats
from distapp.configurations.config import Config
def hypothesis_test():
    config = Config()
    df = pd.read_csv(config.hypothesis_path)
    ai_assisted = df[df['developer_type'] == 'AI Assisted']['logic_score']
    human = df[df['developer_type'] == 'Human Only']['logic_score']
    #hypothesis test
    #calculate hypothesis using sample mean, population mean, standard deviation and sample size
    sample_mean_ai = ai_assisted.mean()
    sample_mean_human = human.mean()
    std_ai = ai_assisted.std()
    std_human = human.std()
    n_ai = len(ai_assisted) 
    n_human = len(human)
    z_score = (sample_mean_ai - sample_mean_human) / ((std_ai**2 / n_ai) + (std_human**2 / n_human))**0.5
    p_value = 1 - stats.norm.cdf(z_score)
    return ai_assisted.to_list(), human.to_list(), z_score, p_value

if __name__ == "__main__":
    ai_assisted, human, z_score, p_value = hypothesis_test()
    print("AI Assisted Logic Scores:", ai_assisted)
    print("Human Logic Scores:", human)
    print(f"Z-score: {z_score}, P-value: {p_value}")
    if p_value < 0.05:
        print("Reject the null hypothesis: AI assisted tools are better than human.")   
    else:
        print("Fail to reject the null hypothesis: No significant difference between AI assisted tools and human.")
