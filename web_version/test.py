import requests
url = "http://127.0.0.1:8000/bridge"
files = {'file':open('8.jpg','rb')}
r = requests.post(url,files = files)
print(r.json())
print(r.text)
print(r.status_code)
print(r.headers)