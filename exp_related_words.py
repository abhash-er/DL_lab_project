import json

with open('relatedwords.json') as f:
    d = json.load(f)
l = []
count = 0
for item in d:
    if item["score"] >=0:
        count+=1
        # item = json.dumps(item)
        # item.replace("'",'"')
        l.append(item)

x = {}
for item in l:
    x[item["word"]] = item["score"]

print(x)