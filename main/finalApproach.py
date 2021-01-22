import json
from main.DataReading import readFromData, readFromGoldData
from main.lesk.lesk import lesk_algorithm
from main.word2vecTraining.transformData import transformSentence, sent2vecOnSentence, get_cosine_vectorial

# file_train_gold = '../training/new_train_gold.txt'
# file_goldTrain = '../training/multilingual/training.en-en.gold'
file_dataTest = '../test/multilingual/test.ar-ar.data-translated'
final_data = '../results/final/test.ar-ar'


def LeskWithSent2Vec(final_data):
    # gold = readFromGoldData(file_goldTrain)
    # total_inputs = len(test)
    # correct_outputs = 0

    test = readFromData(file_dataTest)
    mapOfScore = {}
    mapOfScore['context'] = []

    for index in range(0, len(test)):
        sentence1 = '"' + transformSentence(test[index]['sentence1'], test[index]['lemma'], 0) + '"'
        sentence2 = '"' + transformSentence(test[index]['sentence2'], test[index]['lemma'], 0) + '"'
        lemma = test[index]['lemma']
        # tag = gold[index]['tag']
        id = test[index]['id']
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
