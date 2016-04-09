import nltk

reflections = {
    "am": "are",
    "was": "were",
    "i": "you",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "are": "am",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "I",
    "me": "you"
}


def reflection_check(sentence):
    """Takes a sentence and finds the first key in reflection dict, and
       returns value of that key with the first VB (verb base) found
       in the sentence."""
    reflect_vals = []
    final = []

    sent_toks = sentence.lower().split()

    for idx, token in enumerate(sent_toks):
        if token in reflections:
            reflect = reflections[token]
            reflect_vals.append(reflect)

    final.append(reflect_vals[0])
    final.append(pos_check(sentence))

    return final


def pos_check(sentence):
    """Returns first VB (verb base) in sentence"""
    pos_sent = nltk.pos_tag(nltk.word_tokenize(sentence))
    output = []
    for item in pos_sent:
        if item[1] == 'VB':
            output.append(item)
    return output[0]



