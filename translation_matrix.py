#!/usr/bin/env python
# encoding: utf-8

from gensim import utils
import numpy as np
import gensim
import gensim.models.fasttext as fasttext

"""
Produce translation matrix to translate the word from one language to another language, using either
standard nearest neighbour method or globally corrected neighbour retrieval method [1].
This method can be used to augment the existing phrase tables with more candidate translations, or
filter out errors from the translation tables and known dictionaries [2]. What's more, It also work
for any two sets of named-vectors where there are some paired-guideposts to learn the transformation.
"""


def normalize(mat):
    """ Normalize the word vector's matrix """
    return mat / np.sqrt(np.sum(np.multiply(mat, mat), axis=1, keepdims=True))


class TranslationMatrix(object):
    def __init__(self, src_model, tgt_model, word_pairs=None, random_state=None):
        """
        Initialize the model from a list pair of `word_pair`. Each word_pair is tupe
         with source language word and target language word.
        Examples: [("one", "uno"), ("two", "due")]
        Args:
            `word_pair` (list): a list pair of words
            `src_model` (Word2Vec): a word2vec model of source language
            `tgt_model` (Word2Vec): a word2vec model of target language
        """

        self.source_word = None
        self.target_word = None

        self.src_model = src_model
        self.tgt_model = tgt_model

        self.src_model.init_sims()
        self.tgt_model.init_sims()
        # self.src_mat = normalize(src_model.wv.vectors)
        # self.tgt_mat = normalize(tgt_model.wv.vectors)

        self.random_state = utils.get_random_state(random_state)
        self.translation_matrix = None

        if word_pairs is not None:
            if len(word_pairs[0]) != 2:
                raise ValueError("Each training data item must contain two different language words.")
            self.train(word_pairs)


    def train(self, word_pair):
        """
        Build the translation matrix that mapping from source space to target space.
        Args:
            `word_pairs` (list): a list pair of words
        Returns:
            `translation matrix` that mapping from the source language to target language
        """
        self.src_word, self.tgt_word = zip(*word_pair)

        m1 = [self.src_model.wv.word_vec(item, True) for item in self.src_word]
        m2 = [self.tgt_model.wv.word_vec(item, True) for item in self.tgt_word]

        self.translation_matrix = np.linalg.lstsq(m1, m2, -1)[0]

    def apply_transmat(self):
        """
        Map the target word model to the source word model using translation matrix
        Returns:
            A `Word2Vec` object with the translated matrix
        """
        new_mat = np.dot(self.tgt_model.wv.vectors_norm, self.translation_matrix)
        self.tgt_model.wv.vectors_norm = new_mat
        if isinstance(tgt_model, FastText):
            new_mat_ngram = np.dot(self.tgt_model.wv.vectors_ngrams_norm, self.translation_matrix)
            self.tgt_model.wv.vectors_ngrams_norm = new_mat_ngram


if __name__ == '__main__':
    en_path = '/home/valnyz/PhD/en_cbow_model_4_6'
    fr_path = '/home/valnyz/PhD/fr_cbow_model_4_6'
    pair_file = '/home/valnyz/python/MUSE/data/crosslingual/dictionaries/fr-en.txt'

    # en_model = fasttext.load_facebook_vectors(en_path + '.bin')
    en_model = fasttext.load_facebook_model(en_path + '.bin')
    en_model.vectors = en_model.wv

    # fr_model = fasttext.load_facebook_vectors(fr_path + '.bin')
    fr_model = fasttext.load_facebook_model(fr_path + '.bin')
    fr_model.vectors = fr_model.wv

    word_pair = []
    with open(pair_file, 'r') as f:
        for line in f:
            tup = tuple(line.strip().split())
            # if tup[0] in fr_model and tup[1] in en_model:
            word_pair.append(tup)

    transmat = TranslationMatrix(en_model, fr_model)
    transmat.train(word_pair)
    print('the shape of translation matrix is:',
          transmat.translation_matrix.shape)

    # translation the word
    words = [('one', 'un'), ('two', 'deux'), ('three', 'trois'), ('four',
                                                                  'quatre'), ('five', 'cinq')]
    source_word, target_word = zip(*words)
    translated_word = transmat.translate(source_word, 5)
    print(translated_word)



