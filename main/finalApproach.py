import json
from main.DataReading import readFromData, readFromGoldData
from main.lesk.lesk import lesk_algorithm
from main.word2vecTraining.transformData import transformSentence, sent2vecOnSentence, get_cosine_vectorial

file_train_gold = '../training/new_train_gold.txt'
file_goldTrain = '../training/multilingual/training.en-en.gold'
file_dataTrain = '../test/multilingual/test.en-en.data'
final_data = '../results/final/test.en-en'


def LeskWithSent2Vec(final_data):
    # gold = readFromGoldData(file_goldTrain)
    train = readFromData(file_dataTrain)

    # total_inputs = len(gold)
    correct_outputs = 0

    mapOfScore = {}
    mapOfScore['context'] = []

    for index in range(0, len(train)):
        sentence1 = '"' + transformSentence(train[index]['sentence1'], train[index]['lemma'], 0) + '"'
        sentence2 = '"' + transformSentence(train[index]['sentence2'], train[index]['lemma'], 0) + '"'
        lemma = train[index]['lemma']
        # tag = gold[index]['tag']
        id = train[index]['id']
        score_sent2vec = sent2vecOnSentence(sentence1, sentence2)
        lesk_tag = lesk_algorithm(sentence1, sentence2, lemma)

        if lesk_tag == "T":
            lesk_score = 0.9
        else:
            lesk_score = 0.3

        cosine_vectorial = get_cosine_vectorial(sentence1, sentence2)
        final_score2 = 0.3 * score_sent2vec * 10 + 0.2 * lesk_score + 0.5 * cosine_vectorial * 10

        # final_score1 = 0.3 * score_sent2vec*10 + 0.3 * lesk_score + 0.4 * cosine_vectorial*10
        # final_score3 = 0.3 * score_sent2vec*10 + 0.4 * lesk_score + 0.3 * cosine_vectorial*10
        # final_score4 = 0.4 * score_sent2vec*10 + 0.2 * lesk_score + 0.4 * cosine_vectorial*10
        if final_score2 > 1.035:
            final_tag = "T"
        else:
            final_tag = "F"

        # if final_tag == tag:
        #     correct_outputs = correct_outputs + 1

        mapOfScore['context'].append({
            "id": id,
            "tag": final_tag,
        })
        # print("sent2vec:", score_sent2vec, ", lesk:", lesk_score, ", cosine:", cosine_vectorial, ", true tag:", tag)
        # print("true tag:", tag)
        # print("Scor1 - 30sent2vec*10, 30lesk, 40cosine*10:", final_score1)
        # print("Scor2 - 30sent2vec*10, 20lesk, 50cosine*10:", final_score2)
        # print("Tag scor2:", final_tag)
        # print("Scor3 - 30sent2vec*10, 40lesk, 30cosine*10:", final_score3)
        # print("Scor4 - 40sent2vec*10, 20lesk, 40cosine*10:", final_score4)
        # print('----------', )

    with open(final_data, 'w') as jsonFile:
        json.dump(mapOfScore, jsonFile, indent=3)

    # print("Raspunsuri corecte: ", correct_outputs, " din ", total_inputs)
    # print("Acuratetea este de:", correct_outputs / total_inputs * 100, "%")


LeskWithSent2Vec(final_data)
