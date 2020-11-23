import json
# from lesk.lesk import lesk_algorithm

def readFromData():
    with open('C:\\Users\\User\\Desktop\\Master\\SPLN\\FiiCros\\training\multilingual\\training.en-en.data', 'r',  encoding="utf8") as myfile:
        data = myfile.read()

    obj = json.loads(data)
    return obj

def apply_lesk():
    obj = readFromData()
    for context in obj:
        sentence1 = context["sentence1"]
        print(sentence1)


apply_lesk()