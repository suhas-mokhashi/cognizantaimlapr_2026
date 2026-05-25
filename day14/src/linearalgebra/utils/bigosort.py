#pull the posts from the url and sort them by the number of likes using bigosort algorithm
import requests
from linearalgebra.configurations.conf import Config
def bigosort(url):
   #get the api response from the url
   response = requests.get(url)
   body_data = []
    #check if the response is successful
   if response.status_code == 200:
        posts = response.json()
        #separate body key from posts json and append the values to body_data list
        for post in posts:
            body_data.append(post['body'])
        #sort the body_data list using bigosort algorithm
        body_data.sort()
               
        return body_data
   else:
        print("Failed to fetch posts")
        return []

if __name__ == "__main__":
    url = Config.api_url
    posts= bigosort(url)
    #print the sorted posts
    for post in posts:
        print(post+"\n")