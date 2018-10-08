from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from seq_param import max_seq_length, input_token_index, target_token_index
from keras.callbacks import History
import random
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    for iteration in range(0, 1):
        if iteration == 0:
            filename = 'modelB_20e_REV_lat1000'
            is_reversed = True
            latent_dim = 1000

        batch_size = 128
        epochs = 20
        num_samples = 20000
        checkpoint_name = filename + '_ep-{epoch:02d}vlos-{val_loss:.2f}.h5'
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=100);
        history = History()
        #callbacks = [checkpoint, history]
        callbacks = [history]
        data_path = 'best_data_shuffle.txt'

        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        lines = open(data_path).read().split('\n')
        #print("Shuffling data.")
        #random.shuffle(lines)
        for line in lines[: min(num_samples, len(lines) - 1)]:
            input_text, target_text = line.split(' ')
            if is_reversed:
                input_text = input_text[::-1]  #reversed
                target_text = target_text[::-1]  #reversed
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)


        # input_characters = sorted(list(input_characters))
        # target_characters = sorted(list(target_characters))
        # num_encoder_tokens = len(input_characters)  # liczba wszystkich znaków możliwych #we
        # num_decoder_tokens = len(target_characters)
        # max_encoder_seq_length = max([len(txt) for txt in input_texts])  # max_seq_length  # max([len(txt) for txt in input_texts])
        # max_decoder_seq_length = max([len(txt) for txt in target_texts])  #max_seq_length  # max([len(txt) for txt in target_texts])

        num_encoder_tokens = len(input_token_index)
        num_decoder_tokens = len(target_token_index)
        max_encoder_seq_length = max_seq_length
        max_decoder_seq_length = max_seq_length

        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

        # input_token_index = dict(
        #     [(char, i) for i, char in enumerate(input_characters)])
        # target_token_index = dict(
        #     [(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.

        encoder_inputs = Input(shape=(None, num_encoder_tokens), name="InputWords")
        encoder = LSTM(latent_dim, return_state=True, name='LSTM1')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, num_decoder_tokens), name='TargetWords')
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='LSTM2')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='RhymingLSTM')

        # Run training
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2,
                  callbacks=callbacks)
        model.save(filename + '_model.h5')

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)
        encoder_model.save(filename + '_encoder.h5')

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        decoder_model.save(filename + '_decoder.h5')


        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())




        # for seq_index in range(100):
        #     # Take one sequence (part of the training test)
        #     # for trying out decoding.
        #     input_seq = encoder_input_data[seq_index: seq_index + 1]
        #     decoded_sentence = decode_sequence(input_seq)
        #     print('-')
        #     print('Input sentence:', input_texts[seq_index])
        #     print('Decoded sentence:', decoded_sentence)

        print(history.history['val_loss'])
        file = open(filename + '_val_loss.txt', 'w')
        for i in history.history['val_loss']:
            file.write("%s\n" % i)
        file.close()

        file = open(filename + '_loss.txt', 'w')
        for i in history.history['loss']:
            file.write("%s\n" % i)
        file.close()

        figure = plt.figure(1)

        #plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        #plt.show()
        #figure.savefig('xd.png')

        # # console input
        # while True:
        #         test_seq = input("Input word to rhyme: ")
        #         if test_seq == "" or test_seq == 'q':
        #             break
        #         test_np_array = generate_sequence(test_seq, max_encoder_seq_length, num_encoder_tokens)
        #         print("\noutput :", decode_sequence(test_np_array))

    # end of iteration
