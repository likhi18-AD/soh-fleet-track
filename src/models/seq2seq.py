import tensorflow as tf
from tensorflow.keras import layers, models

def build_seq2seq(input_steps=12, feature_dim=8, output_steps=6, hidden_units=64):
    # Encoder
    encoder_inputs = layers.Input(shape=(input_steps, feature_dim))
    encoder_lstm = layers.LSTM(hidden_units, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = layers.Input(shape=(output_steps, feature_dim))
    decoder_lstm = layers.LSTM(hidden_units, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.TimeDistributed(layers.Dense(1))
    outputs = decoder_dense(decoder_outputs)

    model = models.Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
