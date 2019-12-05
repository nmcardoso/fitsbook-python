import requests

def TestCallback():
  r = requests.get('https://api.github.com/users/octocat')
  print(r)