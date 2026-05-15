#pull the posts from the url and sort them by the number of likes using bigosort algorithm
import requests
from linearalgebra.configurations.conf import Config
def bigosort(url):
   #get the api response from the url
   response = requests.get(url)
    #check if the response is successful
   if response.status_code == 200:
        posts = response.json()
        #sort the posts by the number of likes using bigosort algorithm
        return posts
   else:
        print("Failed to fetch posts")
        return []

if __name__ == "__main__":
    url = Config.api_url
    posts= bigosort(url)
    #print the sorted posts
    print(posts)