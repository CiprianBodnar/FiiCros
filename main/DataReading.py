import json
from main.lesk.lesk import lesk_algorithm

file_train_gold = '../training/new_train_gold.txt'
file_data_context = '../training/train_data_context.txt'


def readFromData():
    with open(
            'C:\\Users\\User\\Desktop\\Master\\SPLN\\FiiCros\\training\\multilingual\\training.en-en.data',
            'r', encoding="utf8") as myfile:
        data = myfile.read()

    obj = json.loads(data)
    return obj

def readFromGoldData():
    with open(
            'C:\\Users\\User\\Desktop\\Master\\SPLN\\FiiCros\\training\\multilingual\\training.en-en.gold',
            'r', encoding="utf8") as myfile:
        data = myfile.read()

    gold_obj = json.loads(data)
    return gold_obj

def readFromTrainData():
    with open(
            'C:\\Users\\User\\Desktop\\Master\\SPLN\\FiiCros\\training\\example.json',
            'r', encoding="utf8") as myfile:
        data = myfile.read()

    train_obj = json.loads(data)
    return train_obj

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


def result_accuracy():
    gold = readFromGoldData()
    train = readFromTrainData()

    good = 0
    for i in range(0, len(gold)):
        if(gold[i]['tag'] == train[i]['tag']):
            good = good + 1
    result = good/len(gold) * 100
    print("Good answer(8000 initial): ", good, '\n')
    print("Percentage of good answer: ", result, '%')

result_accuracy()


# apply_lesk(file_train_gold)
