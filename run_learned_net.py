from keras.models import load_model
from seq_param import max_seq_length, input_token_index, target_token_index
from keras.utils import plot_model
from seq_param import *
import numpy as np
import pydot_ng
from nltk.corpus import wordnet


def generate_sequence(input_word, seq_length, dict_length):
    encoder_word = np.zeros((1, seq_length, dict_length), dtype='float32')
    for t, char in enumerate(input_word):
        encoder_word[0, t, input_token_index[char]] = 1.
    return encoder_word


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# MAIN
checkIfExist = False

filename = 'modelB_20e_REV_lat1000'
is_reversed = True

print("Loading model")
model = load_model(filename + '_model.h5')
# plot_model(model, to_file=filename + '_model.png', show_shapes=True)

encoder_model = load_model(filename + '_encoder.h5')
# plot_model(encoder_model, to_file=filename + '_encoder.png', show_shapes=True)
decoder_model = load_model(filename + '_decoder.h5')
# plot_model(decoder_model, to_file=filename + '_decoder.png', show_shapes=True)
print("model loaded")

num_samples = 100
data_path = 'test_data_shuffle.txt'
input_texts = []
target_texts = []
lines = open(data_path).read().split('\n')


for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split(' ')
    if is_reversed:
        input_text = input_text[::-1]    # reversed
        target_text = target_text[::-1]  # reversed
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)

num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)
max_encoder_seq_length = max_seq_length
max_decoder_seq_length = max_seq_length

# liczba slow, max dlugosc pojedynczej sekwencji, dlugosc slownika
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

print(input_texts)
print('#')
print(target_text)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


file = open(filename + '_test.txt', 'w')
for seq_index in range(100):

    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq).rstrip()  # remove white-chars
    input_sentence = input_texts[seq_index]
    if is_reversed:
        input_sentence = input_sentence[::-1]
        decoded_sentence = decoded_sentence[::-1]
    # print('-')
    # print(input_sentence)
    # print(decoded_sentence)
    exist_point = ""
    if checkIfExist:
        if not wordnet.synsets(decoded_sentence):
            exist_point = "0;"
        else:
            exist_point = "1;"
    file.write(input_sentence + ";" + decoded_sentence + exist_point + "\n")
file.close()


# while True:
#     test_seq = input("Input word to rhyme: ")
#     if test_seq == "" or test_seq == 'q':
#         break
#     if is_reversed:
#         test_seq = test_seq[::-1]
#     test_np_array = generate_sequence(test_seq, max_encoder_seq_length, num_encoder_tokens)
#     output_text = decode_sequence(test_np_array)
#     if is_reversed:
#         output_text = output_text[::-1]
#     print("\noutput :", output_text)




