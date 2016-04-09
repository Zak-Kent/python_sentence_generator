import nltk
import random
from nltk.util import bigrams
import logging
logging.basicConfig(format='%(levelname)s %(funcName)s %(lineno)d:%(message)s', level=logging.DEBUG)


class SentenceGen:
    """ A class that generates a sentence based on the patterns in a provided corpus

    >>> gen = SentenceGen(nltk.corpus.inaugural)
    >>> sent = gen.sent_gen(('want', 'VB'), .5)
    >>> isinstance(sent, list)
    True
    >>> 1000 > len(sent) > 1
    True
    """
    def __init__(self, corpra, *args, **kwargs):

        if isinstance(corpra, basestring):
            self.words = nltk.word_tokenize(corpra)
        else:
            self.words = corpra.words(*args, **kwargs)

        self.pos_words = nltk.pos_tag(self.words)
        self.bgrams = bigrams(self.pos_words)
        self.freqdist = nltk.ConditionalFreqDist(self.bgrams)

        self.word_dict = {}
        self.pos_dict = {}
        self.punct = [',', '.', '!', ':', ';', '?', '--', '-', '"', "'", 's', '$', ',"']
        self.word_dict_builder()

    def trip_seqs(self):
        """generates sequences of trigrams used to build dictionary in self.word_dict_builder"""
        if len(self.pos_words) < 3:
            raise StopIteration

        for item in range(len(self.pos_words) - 2):
            yield (self.pos_words[item], self.pos_words[item + 1], self.pos_words[item + 2])

    def word_dict_builder(self):
        """builds a dictionary with bigram combos as a key with list of possible following words as its value"""
        for word1, word2, word3 in self.trip_seqs():
            # make keys only words no POS tuples, makes matching easier
            word_tuple = (word1[0], word2[0])
            if word_tuple in self.word_dict:
                # don't add punctuation to the dict
                if word3[0] not in self.punct:
                    self.word_dict[word_tuple].append(word3)
            else:
                if word3[0] not in self.punct:
                    self.word_dict[word_tuple] = [word3]

            # build another dict using only the part of speech of each word/POS tuple
            pos_tuple = (word1[1], word2[1])
            if pos_tuple in self.pos_dict:
                self.pos_dict[pos_tuple].append(word3[1])
            else:
                self.pos_dict[pos_tuple] = [word3[1]]

    def pos_structure_builder(self, seed1, seed2, sent_length=15):
        """Builds a sentence structure for a response. seed1 & seed2 need to be a POS tag that is in pos_dict."""
        sentence_terms = ['.', '!', '?']
        output_pos = []

        while len(output_pos) < sent_length:
                output_pos.append(seed1)

                if seed1 in sentence_terms:
                    logging.debug("break inside pos_struct builder")
                    break

                new_pos = random.choice(self.pos_dict[seed1, seed2])
                seed1, seed2 = seed2, new_pos

        logging.debug((str(output_pos)))
        return output_pos

    def seed_maker(self, seed_word):
        """generates a second seed word using bigrams and NLTKs conditional frequency distribution"""
        word_bank = []
        for item in self.freqdist[seed_word]:
            if item[0] not in self.punct:
                word_bank.append(item)

        if len(word_bank) > 0:
            seed2 = random.choice(word_bank)
            return seed2
        else:
            raise Exception('seed_word not found in dict')

    def sent_gen(self, seed_word=None, threshold=.4):
        """Generates sentence using provided seed_word and a generated sentence structure built with NLTK pos_tags.
           This method needs a seed_word to start the sentence and a threshold value that is used to measure how
           accurate the POS choices were when the sentence is created.
        """
        percent_correct_pos = None

        if seed_word is None:
            seed_word = ('want', 'VB')

        # run sentence building loop until the % correct POS chosen is above threshold
        while percent_correct_pos < threshold:
            percent_correct_pos = 0
            sent_output = []
            
            # set seed1 = seed_word so original isn't lost on multiple tries to get sentence over threshold
            seed1 = seed_word
            seed2 = self.seed_maker(seed_word)
            logging.debug(str(seed1 + seed2))
            
            # build a target sentence structure using the POS of seed1 and seed2
            pos_structure = self.pos_structure_builder(seed1[1], seed2[1])

            for pos in pos_structure:
                sent_output.append(seed1[0])

                pos_correct = []
                pos_random = []

                # use 0th element of seed1 and seed2, keys in dict don't have POS tags
                follow_words = self.word_dict.get((seed1[0], seed2[0]))
                logging.debug(('words found in word dict ' + str(follow_words)))

                if follow_words is None:
                    logging.debug(('Break inside for loop, no word in word_dict ' + '*' * 50))
                    break

                # if possible select the correct POS out of the follow_words list, if POS not there select
                # a random word from the list and continue
                for tup in follow_words:
                    if tup[1] == pos:
                        pos_correct.append(tup)
                    else:
                        pos_random.append(tup)

                if len(pos_correct) != 0:
                    new_word = random.choice(pos_correct)
                    # keep track of how many times the correct POS is chosen, use to calculate % correct choices
                    percent_correct_pos += 1
                else:
                    new_word = random.choice(pos_random)
                
                # change the seed words and move the process one word along in the sentence
                seed1, seed2 = seed2, new_word

                logging.debug('-' * 35)
                logging.debug("word choice: {}".format(new_word))
                logging.debug('-' * 35)

            # calculate the percentage of correct POS chosen and use to evaluate how good generated
            # sentence is, if it is below the threshold continue while loop and generate another sentence.
            percent_correct_pos = float(percent_correct_pos) / len(pos_structure)
            logging.debug(("correct POS %: ", percent_correct_pos))
            logging.debug(sent_output)

        sent_output.append(seed2[0])
        return sent_output, percent_correct_pos

def test():
    import doctest
    logging.basicConfig(format='%(levelname)s %(funcName)s %(lineno)d:%(message)s', level=logging.DEBUG)
    doctest.testmod()

if __name__ == '__main__':
    test()

