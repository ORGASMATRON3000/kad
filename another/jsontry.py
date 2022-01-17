import json
tmp=[]
for i in range(10):
    tmp.append(i)
xex=[]
for i in range(10):
    xex.append(i)

kek=[]
kek.append(tmp)
kek.append(xex)


a={"CIA":kek}
y = json.dumps(a)
print(y)
