from pywsd.lesk import simple_lesk

def lesk_algorithm(sentence1, sentence2, lemma):
    answer1 = simple_lesk(sentence1, lemma)
    answer2 = simple_lesk(sentence2, lemma)

    print("Sentence 1: ", sentence1)
    print("-Sense:", answer1)
    print("-Definition:", answer1.definition())

    print("Sentence 2: ", sentence2)
    print("-Sense:", answer2)
    print("-Definition:", answer2.definition())

    if(answer1 == answer2):
        print("TRUE")
    else:
        print("FALSE")

lesk_algorithm('I went to the bank to deposit my money', 'I go to the bank to deposit my money', 'bank')