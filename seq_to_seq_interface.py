from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import random
import numpy as np

from lstm_teach import max_encoder_seq_length, num_encoder_tokens, decode_sequence
from seq_param import input_token_index


def encode_word(input_word, seq_length, dict_length):
    encoder_word = np.zeros((1, seq_length, dict_length), dtype='float32')
    for t, char in enumerate(input_word):
        encoder_word[0, t, input_token_index[char]] = 1.
    return encoder_word


model = load_model('s2s.h5')

while True:
    test_seq = input("Input word to rhyme: ")
    if test_seq == "":
        break
    encoded_word = encode_word(test_seq, max_encoder_seq_length, num_encoder_tokens)
    print("\noutput :", decode_sequence(encoded_word))






