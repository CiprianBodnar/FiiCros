from pywsd.lesk import simple_lesk

def lesk_algorithm(sentence1, sentence2, lemma):
    answer1 = simple_lesk(sentence1, lemma)
    answer2 = simple_lesk(sentence2, lemma)

    definition1 = ""
    if hasattr(answer1, 'definition'):
        definition1 = answer1.definition()
    definition2 = ""
    if hasattr(answer2, 'definition'):
        definition2 = answer2.definition()
    if answer1 == answer2:
        tag = "T"
    else:
        tag = "F"
    return tag, answer1, definition1, answer2, definition2

