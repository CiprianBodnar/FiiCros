from nltk.corpus import wordnet
from nltk.wsd import lesk
# from ..lesk.lesk_utils import get_sense_key, evaluate_accuracy


SENSE_KEY = 0
def get_sense_key(synset):
    """returns the sense key as a string for the given synset"""
    sense_keys = [sense.key() for sense in synset.lemmas()]
    return sense_keys[SENSE_KEY]


def get_synsets(lemma):
    """
    return the list of synsets corresponding to a lemma
    """
    return wordnet.synsets(lemma)


def get_synset_definition(lemma):
    synsets = wordnet.synsets(lemma)
    return [synset.definition() for synset in synsets]


def check_wordnet_version(wordnet):
    if not '3.0' == wordnet.get_version():
        raise ValueError("Wordnet version is {}. Must be 3.0".format(wordnet.get_version()))
    pass


def evaluate_accuracy(predictions, targets):
    """Evaluate accuracy

    Args
        predictions: (list of string)
        targets: (list of list os string)
    """
    correct = 0
    total = len(targets)

    for prediction, target in zip(predictions, targets):
        if prediction in target:
            correct += 1
    accuracy = round((correct / total) * 100, 4)
    return accuracy

def apply_lesk(context, lemma):
    """returns word sense for synset found using lesk's algorithm"""
    synset = lesk(context, lemma)
    if synset is not None:
        return get_sense_key(synset)
    else:
        print('synset empty for {}'.format(lemma))
        return None


print(apply_lesk('The indeterminate sentence must be reviewed by the court when the nominal sentence has expired and every three years afterward.','indeterminate'))