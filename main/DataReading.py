import json
from main.lesk.lesk import lesk_algorithm

file_train_gold = '../training/new_train_gold.txt'
file_data_context = '../training/train_data_context.txt'


def readFromData():
    with open(
            'C:\Faculty\Master2\SPALN\FiiCros\\training\multilingual\\training.en-en.data',
            'r', encoding="utf8") as myfile:
        data = myfile.read()

    obj = json.loads(data)
    return obj


def createDataContext(tag, sentence1, sentence2, answer1, definition1, answer2, definition2, file_context):
    f = open(file_context, 'a', encoding="utf-8")
    f.writelines("\nTag: " + tag + "\n")
    f.writelines("Sentence 1: " + sentence1 + "\n")
    f.writelines("-Sense:" + str(answer1) + "\n")
    f.writelines("-Definition:" + definition1 + "\n")
    f.writelines("Sentence 2: " + sentence2 + "\n")
    f.writelines("-Sense:" + str(answer2) + "\n")
    f.writelines("-Definition:" + definition2 + "\n")
    f.writelines('\n')
    f.close()


def apply_lesk(train_gold):
    f = open(train_gold, 'a', encoding="utf-8")
    obj = readFromData()
    mapOfContext = {}
    mapOfContext['context'] = []
    for context in obj:
        sentence1 = context["sentence1"]
        sentence2 = context["sentence2"]
        lemma = context["lemma"]
        tag, answer1, definition1, answer2, definition2 = lesk_algorithm(sentence1, sentence2, lemma)
        createDataContext(tag, sentence1, sentence2, answer1, definition1, answer2, definition2, file_data_context)
        #print(context["id"] + '=>' + tag + '\n')

        mapOfContext['context'].append({
            'id': context["id"],
            'tag' : tag
        })
       # f.writelines(context["id"] + '=>' + tag + '\n')

    with open('../training/example.json', 'w') as jsonFile:
        json.dump(mapOfContext, jsonFile, indent= 3)
    f.close()


apply_lesk(file_train_gold)
