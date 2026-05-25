#create binomial distribution for ecommerce product search and buy data using conversion rate as probability of success and number of searches as number of trials
import pandas as pd
from scipy.stats import binom
from distapp.configurations.config import Config
from matplotlib import pyplot as plt

def create_binomial_distribution():
    config = Config()
    #read ecommerce product search and buy data
    df = pd.read_csv(config.binomial_path)
    #get conversion rate as probability of success
    conversion_rate = df['people_bought'].mean() / df['people_searched'].mean()
    #get number of searches as number of trials
    num_searches = df['people_searched'].mean()
    #create binomial distribution
    binomial_dist = binom(n=int(num_searches), p=conversion_rate)
    return binomial_dist,conversion_rate,num_searches

def draw_binomial_distribution(binomial_dist, conversion_rate, num_searches):
    #draw binomial distribution
    x = range(0, int(num_searches)+1)
    y = binomial_dist.pmf(x)
    plt.bar(x, y)
    plt.xlabel('Number of people bought the product')
    plt.ylabel('Probability')
    plt.title('Binomial Distribution of people bought the product')
    plt.show()
    

if __name__ == "__main__":
    binomial_dist, conversion_rate, num_searches = create_binomial_distribution()
    #how many people bought the product in 10 searches
    sample = binomial_dist.rvs(size=50)
    print(sample)
    draw_binomial_distribution(binomial_dist, conversion_rate, num_searches)
    