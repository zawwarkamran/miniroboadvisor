#TestDataHub
#URL https://xkcd.com/353/

import requests
from requests.auth import HTTPBasicAuth
import time
import json
#import Data.py

#Data File

teamid ='bathdrones'
teamkey = 'd22ec71d-af83-4cd6-847c-ea5031870d9b'
Authentication = ('bathdrones','d22ec71d-af83-4cd6-847c-ea5031870d9b')

url = "https://api.mksmart.org/sciroc-competition/master/sciroc-episode12-patient"
#"https://api.mksmart.org/sciroc-competition/master/sciroc-episode12-menu"


Status_Info = {
  "@id": "P001",
  "@type": "RobotStatus",
  "message": "ready to launch",
  "episode": "EPISODE12",
  "team": "bathdrones",
  "timestamp": "2019-09-15T15:03:56.981Z",
  "x": 4,
  "y": 5,
  "z": 0
}


Robot_Location = {
  "@id": "bathdrones",
  "@type": ["RobotLocation"],
  "episode": "EPISODE12",
  "team": "string",
  "timestamp": "2019-09-14T16:24:11.152Z",
  "x": 0,
  "y": 0,
  "z": 0
}

Inventory_Order = {
  "@id": "bathdrones",
  "@type": "Patient",
  "x": 0,
  "y": 0,
  "z": 0
}

Image_Report = {
  "@id": "bathdrones",
  "@type": "ImageReport",
  "team": "string",
  "timestamp": "2019-09-14T16:24:33.073Z",
  "x": 0,
  "y": 0,
  "z": 0,
  "base64": "string",
  "format": "image/jpeg"}

print('start get\n\n')
#
response = requests.request("GET", url, auth=HTTPBasicAuth(teamkey, ''))
print(type(response))
print(response.status_code)
# print(response.text)

data = response.json()

# print(data)

# print(Status_Info)
payload = json.dumps(Status_Info)


url = "https://api.mksmart.org/sciroc-competition/"+teamid+"/sciroc-robot-status/P001"
response = requests.post(url, data=payload, auth=HTTPBasicAuth(teamkey, ''))
print(type(response))
print(response.status_code)
print('hello')
# print(response.text)

# r = requests.get('https://api.pp.mksmart.org/sciroc-competition/', auth= Authentication)#data= payload)
# print(r.status_code)
# print(r.headers)
# print('\n\nend get  \n\nstart \n\n')
# #payload = {'@id':teamid,'@type':requesttype,'label':label,'description':description,'status':status}
# r = requests.post('https://api.pp.mksmart.org/sciroc-competition/', auth= Authentication, data=json.dumps(Status_Info))#data= payload)
# print(r.status_code)
# print(r.headers)
# print('end post\n\n')

# #create an episode


