import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from gensim.parsing.preprocessing import STOPWORDS
import string
from main.DataReading import readFromData, readFromGoldData
import numpy as np
import collections
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from scipy import spatial
from sent2vec.vectorizer import Vectorizer


def tokenizationSentence(sentence):
    nltk_tokens = nltk.word_tokenize(sentence)
    return nltk_tokens


def getLemma(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)


def getStemmer(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


def getLowercase(word):
    result = word[0].lower() + word[1:]
    return str(result)


def removeStopElements(sentence):
    my_stop_words = STOPWORDS.union({'mystopword1', 'mystopword2'})
    tokenization_list = tokenizationSentence(sentence)
    punctuation_list = string.punctuation

    clean_sentence = ""
    for word in tokenization_list:
        if getLowercase(word) not in my_stop_words and word not in my_stop_words and word not in punctuation_list:
            clean_sentence = clean_sentence + ' ' + word
    return clean_sentence


def transformSentence(sentence, word_searched, distance):
    position = 0
    clean_sentence_tokenizated = tokenizationSentence(removeStopElements(sentence))
    index = 0
    lemmatization_sentence = []
    for word in clean_sentence_tokenizated:
        if getLemma(word) == word_searched or getStemmer(word) == word_searched:
            position = index
        index = index + 1
        lemmatization_sentence.append(getLemma(word))

    new_sentence = ''
    index_new = 0
    for word in lemmatization_sentence:
        if index_new == 0:
            new_sentence = word + ' '
        else:
            if index_new == len(lemmatization_sentence):
                new_sentence = new_sentence + word
            else:
                new_sentence = new_sentence + word + ' '
        index_new = index_new + 1

    if distance == 0:
        return new_sentence

    if distance <= position + 1:
        left_limit = position - distance
    else:
        left_limit = 0
    if (position + distance) <= len(new_sentence):
        right_limit = position + distance
    else:
        right_limit = len(new_sentence)

    final_sentence_list = tokenizationSentence(new_sentence)[left_limit:right_limit + 1]

    final_sentence = ''
    for word in final_sentence_list:
        if word == final_sentence_list[0]:
            final_sentence = word + ' '
        else:
            if word == final_sentence_list[-1]:
                final_sentence = final_sentence + word
            else:
                final_sentence = final_sentence + word + ' '
    return final_sentence


def createNewTrainData():
    file_train_data = '../../training/multilingual/training.en-en.data'
    file_train_gold = '../../training/multilingual/training.en-en.gold'
    final_train_data = '../../training/train_data_word2vec.json'
    data_train = readFromData(file_train_data)
    gold_train = readFromGoldData(file_train_gold)

    index = 0
    mapOfContext = {}
    mapOfContext['context'] = []
    for context in data_train:
        mapOfContext['context'].append({
            "sentence1": transformSentence(context["sentence1"], context["lemma"], 0),
            "sentence2": transformSentence(context["sentence2"], context["lemma"], 0),
            "lemma": context["lemma"],
            "tag": gold_train[index]['tag']
        })
        index = index + 1
    with open(final_train_data, 'w') as jsonFile:
        json.dump(mapOfContext, jsonFile, indent=3)


def getVocabulary():
    all_words = []
    final_train_data = '../../training/train_data_word2vec.json'
    with open(final_train_data, "r") as f:
        data = json.load(f)

    data_matrix = []

    for input in data:
        temp = []

        for word in input['sentence1'].split():
            all_words.append(word.lower())
        for word in input['sentence2'].split():
            all_words.append(word.lower())

        for j in word_tokenize(input['sentence1']):
            temp.append(j.lower())
        for j in word_tokenize(input['sentence2']):
            temp.append(j.lower())

        data_matrix.append(temp)

    ctr = collections.Counter(all_words)

    final_vocabulary = []
    for i in ctr:
        if (ctr[i] > 1):
            final_vocabulary.append(i)
    count = len(final_vocabulary)

    return final_vocabulary, count, data_matrix


final_vocabulary, count, data_matrix = getVocabulary()


# print(final_vocabulary)
# print(count)

def cosine_similarity(vocabulary, word1, word2):
    model1 = gensim.models.Word2Vec(vocabulary, min_count=1, size=100)
    return model1.similarity(word1, word2)


# cosine_similarity(data_matrix, 'context', 'coordination')


def addToMatrix():
    vocabulary, count, data_matrix = getVocabulary()
    final_train_data = '../../training/train_data_word2vec.json'
    with open(final_train_data, "r") as f:
        data = json.load(f)

    word2vec_matrix = [vocabulary]
    score_list = []
    # MyFile = open('output2.txt', 'w')

    MyFile = open('sum_score.txt', 'w')
    for input in data:
        row = []
        words_list = []
        for word in vocabulary:
            if word in input['sentence1'].lower().split():
                row.append('1')
                words_list.append(word)
            else:
                row.append('0')
            if word in input['sentence2'].lower().split():
                row.append('1')
                words_list.append(word)
            else:
                row.append('0')

        sum_score = 0
        for word1 in words_list:
            for word2 in words_list:
                if word1 != word2:
                    # print(cosine_similarity(data_matrix, word1, word2))
                    sum_score = sum_score + cosine_similarity(data_matrix, word1, word2)
        sum_score = sum_score / len(words_list)

        MyFile.writelines(sum_score)
        MyFile.write('\n')

        score_list.append(sum_score)
        word2vec_matrix.append(row)
    #     MyFile.writelines(row)
    #     MyFile.write('\n')
    MyFile.close()

    return word2vec_matrix, score_list


# addToMatrix()

def sent2vecOnSentence(sentence1, sentence2):
    sentences = [sentence1, sentence2]
    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    vectors_bert = vectorizer.vectors

    dist = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])
    return dist


def getCosineSimilarity():
    final_train_data = '../../training/train_data_word2vec.json'
    score_data = '../../training/score_data_sent2vec.json'
    with open(final_train_data, "r") as f:
        data = json.load(f)

    index = 0
    mapOfScore = {}
    mapOfScore['context'] = []
    for input in data:
        sentence1_cosine = '"' + input['sentence1'] + '"'
        sentence2_cosine = '"' + input['sentence2'] + '"'
        mapOfScore['context'].append({
            "sentence1": input["sentence1"],
            "sentence2": input["sentence2"],
            "tag": input['tag'],
            "score": sent2vecOnSentence(sentence1_cosine, sentence2_cosine)
        })
        index = index + 1
        print(index)
        print(sent2vecOnSentence(sentence1_cosine, sentence2_cosine))

    with open(score_data, 'w') as jsonFile:
        json.dump(mapOfScore, jsonFile, indent=3)

getCosineSimilarity()
# print(sent2vecOnSentence("context coordination integration Bolivia hold key play process infrastructure development ","school water needed girl sent fetch taking time away study play "))
# sent2vecOnSentence("context coordination integration Bolivia hold key play process infrastructure development ","school water needed girl sent fetch taking time away study play ")
