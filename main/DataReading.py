import json

def readFromData():
    with open('C:\Faculty\Master2\SPALN\FiiCros\\training\multilingual\\training.en-en.data', 'r',  encoding="utf8") as myfile:
        data=myfile.read()

    obj = json.loads(data)
    for o in obj:
        print(o)

readFromData()