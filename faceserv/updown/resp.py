import requests

API_URL = 'http://127.0.0.1:5000'
API_KEY = 'i0cgsdYL3hpeOGkoGmA2TxzJ8LbbU1HpbkZo8B3kFG2bRKjx3V'
TARGET_URL='http://httpbin.org/post'
headers = {'UserAPI-Key': API_KEY,'target_URL': TARGET_URL}

#загрузка видео на сервер
filename='shortvideo.mp4'
with open(filename,'rb') as fp:
    content = fp.read()
path='{}/files/'+filename
path=path.format(API_URL)
print(path)
target_URL="target URL"
array=[]
array.append(content)
array.append(target_URL)
response = requests.post(
    path, headers=headers, data=content,
)
