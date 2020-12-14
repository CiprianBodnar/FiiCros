import json

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from gensim.parsing.preprocessing import STOPWORDS
import string
from main.DataReading import readFromData, readFromGoldData


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

def countVocabulary():
    final_train_data = '../../training/train_data_word2vec.json'
    with open(final_train_data, "r") as f:
        words = set(f.readlines())
        count = len(words)
    print(count)

# print(transformSentence("The Sahul Shelf is sometimes taken to also include the Rowley Shelf to the southwest,
# girding the north coast of Western Australia as far as North West Cape.", 'gird', 5)) createNewTrainData()
countVocabulary()