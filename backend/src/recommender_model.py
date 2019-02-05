from keras.layers import Dense, Embedding, GRU, Input, TimeDistributed
from keras.models import Model

# define dimensions
d_embedding = 8
d_hidden = 8


def rnn_recommender(N_players):
    x = Input(shape=(None,)) # We use non so we can deal with variable length sequences
    embedding = Embedding(N_players + 1, output_dim=d_embedding, mask_zero=True)(x)
    out = GRU(d_hidden, return_sequences=True)(embedding)
    out = TimeDistributed(Dense(units=N_players, activation='softmax'))(out)
    model = Model(inputs=x, outputs=out)
    return model