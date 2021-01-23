import json
from main.lesk.lesk import lesk_algorithm

file_train_gold = '../training/new_train_gold.txt'
file_data_context = '../training/train_data_context.txt'
file_dataTrain = '../test/multilingual/test.fr-fr.data-translated'
file_goldTrain = '../training/multilingual/training.en-en.gold'

def readFromData(file_train_data):
    with open(
            file_train_data ,'r', encoding="utf-8") as myfile:
        data = myfile.read()

    obj = json.loads(data)
    return obj

def readFromGoldData(file_train_gold):
    with open(
            file_train_gold,'r', encoding="utf-8") as myfile:
        data = myfile.read()

    gold_obj = json.loads(data)
    return gold_obj

def readFromTrainData():
    with open(
            'C:\\Users\\Andrada\\OneDrive\\Desktop\\Master\\SPLN\\FiiCros\\results\\training.en-en.data',
            'r', encoding="utf-8") as myfile:
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
    obj = readFromData(file_dataTrain)
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

    with open('../results/test.fr-fr', 'w') as jsonFile:
        json.dump(mapOfContext, jsonFile, indent= 3)
    f.close()


def result_accuracy():
    gold = readFromGoldData(file_goldTrain)
    train = readFromTrainData()

    good = 0
    for i in range(0, len(gold)):
        if gold[i]['tag'] == train[i]['tag']:
            good = good + 1
    result = good/len(gold) * 100
    print("Good answer(8000 initial): ", good, '\n')
    print("Percentage of good answer: ", result, '%')

def translate(source):
    from deep_translator import GoogleTranslator
    obj = readFromData(file_dataTrain)
    mapOfContext = {}
    mapOfContext['context'] = []
    for context in obj:
        try:
            sentence1 = GoogleTranslator(source, target='en').translate(context["sentence1"])
        except:
            sentence1 = context["sentence1"]

        print("sentence1", sentence1)
        try:
            sentence2 = GoogleTranslator(source, target='en').translate(context["sentence2"])
        except:
            sentence2 = context["sentence2"]
        print("sentence2", sentence2)

        try:
            lemma = GoogleTranslator(source, target='en').translate(context["lemma"])
        except:
            lemma = context["lemma"]
        print("lemma", lemma)

        mapOfContext['context'].append({
            'id': context["id"],
            'lemma': lemma,
            'pos': context["pos"],
            'sentence1': sentence1,
            'sentence2': sentence2,

            'start1': context["start1"],
            'end1': context["end1"],
            'start2': context["start2"],
            'end2': context["end2"]

            # 'ranges1': context["ranges1"],
            # 'ranges2': context["ranges2"]
        })

    with open('../test/multilingual/test.ar-ar'
              '.data-translated', 'w') as jsonFile:
        json.dump(mapOfContext, jsonFile, indent=3)
# result_accuracy()

apply_lesk(file_train_gold)