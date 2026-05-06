#create bernoulli distribution using email spam dataset from resources folder
import pandas as pd
from scipy.stats import bernoulli
from sklearn.preprocessing import LabelEncoder  
from distapp.configurations.config import Config    
from matplotlib import pyplot as plt

def create_bernoulli_dist():
    config = Config()
    df = pd.read_csv(config.bernoulli_path)
    #create bernoulli distribution using email spam dataset from resources folder
    #calculate the probability of spam emails in the dataset using label column
    # label column has text either spam or not spam, so we can calculate the probability of spam emails by counting the number of spam emails and dividing by the total number of emails
    #p = df['label'].value_counts()['spam'] / len(df)   
    #label encoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    #encoded labels will be 0 for not spam and 1 for spam, so we can calculate the probability of spam emails by taking the mean of the label column
    print(f"Encoded labels: {df['label']}")
    #spam probability is the mean of the label column since 1 represents spam and 0 represents not spam
    p = df['label'].mean()
    print(f"Probability of spam emails: {p}")
    print(f"Probability of not spam emails: {1 - p}")
    #create a bernoulli distribution object using the calculated probability
    bernoulli_dist = bernoulli(p)
    return bernoulli_dist

def plot_bernoulli_dist(bernoulli_dist):
    #plot the bernoulli distribution using matplotlib
    x = [0, 1]
    y = bernoulli_dist.pmf(x)
    plt.bar(x, y)
    plt.xticks(x, ['Not Spam', 'Spam'])
    plt.xlabel('Email Type')
    plt.ylabel('Probability')
    plt.title('Bernoulli Distribution of Email Spam')
    plt.show()

if __name__ == "__main__":
    bernoulli_dist = create_bernoulli_dist()   
    #generate 25 random samples from the bernoulli distribution
    samples = bernoulli_dist.rvs(size=25)
    print(f"Random samples from the bernoulli distribution: {samples}")
    plot_bernoulli_dist(bernoulli_dist)
