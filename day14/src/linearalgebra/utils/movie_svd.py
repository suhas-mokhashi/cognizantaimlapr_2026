#create svd for movie recommendation system
from matplotlib.pylab import svd
import numpy as np
import pandas as pd
from linearalgebra.configurations.conf import Config

def svd_movie_recommendation(movie_path):
   df=pd.read_csv(movie_path)
   df=df.drop('user_id',axis=1)
   print(df.head())
   u,s,v= svd(df)
   return u,s,v

if __name__ == "__main__":
    movie_path = Config.movie_path
    svd_movie_recommendation(movie_path)
    u,s,v = svd_movie_recommendation(movie_path)
    print("U matrix: ", u)
    print("S matrix: ", s)
    print("V matrix: ", v)