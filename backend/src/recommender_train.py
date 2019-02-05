import os
import json
import itertools
import numpy as np
import pandas as pd
import requests
import time

from keras.optimizers import Adam
from keras.utils import to_categorical
from recommender_model import rnn_recommender

# --> helper functions
def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def get_splits(y, train_frac=0.6, val_frac=0.2, n_perm=24):
    n_max = int(y.shape[0])

    # Training examples
    n_train = int(np.round(n_max * train_frac))
    n_train = n_train - n_train % n_perm # make sure matches number of permutations (to prevent leakage)

    n_val = int(n_train + np.round(n_max * val_frac))
    n_val = n_val - n_val % n_perm

    print("Training samples:   %i"%(n_train))
    print("Validation samples: %i"%(n_val - n_train))
    print("Test samples:       %i" %(n_max - n_val))
    idx_train = range(0, n_train)
    idx_val = range(n_train, n_val)
    idx_test = range(n_val, n_max)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))

def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

# --> Import training data from server
matches_url = os.environ["KICKERSCORE_MATCHES"]
r_matches = requests.request('GET', url=matches_url)
print()
print(r_matches)

# --> Convert json to pandas
df_matches = pd.io.json.json_normalize(json.loads(r_matches.text), meta=['id'])
df_matches = df_matches[['id',
                         'players.blue.defense.name', 'players.blue.offense.name',
                         'players.red.defense.name', 'players.red.offense.name']]
N_games = len(df_matches) # Number of games played



# --> Create data array
x = df_matches.drop('id', axis=1).values
unique_ids = np.arange(1, len(np.unique(x))+1)
N_players = unique_ids.shape[0]
print("number of unique players: %i"%N_players)

# Convert ids to integers
id2int = dict(np.c_[np.unique(x), unique_ids])
x = df_matches.drop('id', axis=1).replace(id2int).values

# Permute all games (24 per games)
x_shuffled = np.array([list(itertools.permutations(_x)) for _x in x])
x_shuffled = x_shuffled.reshape((-1, 4))

# --> Convert
x_1h = to_categorical(x_shuffled)
y_1h = x_1h[:,1:,1:]
x_1h = x_1h[:,:-1,1:]
x_cat = x_shuffled[:,:-1] # integers from 1 through N_players

y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y_1h, train_frac = 0.7, val_frac=0.2)
# --> split into training/validation/test set


# --> Train model
N_EPOCH = 100
EARLY_STOP_CRITERIUM = 10
best_val_loss = 1e9

model = rnn_recommender(N_players)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

for epoch in range(1, N_EPOCH):

    tic = time.time()
    model.fit(x=x_cat, y=y_1h, batch_size=64, epochs=1, sample_weight=train_mask)
    preds = model.predict(x_cat)
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])

    if epoch % 10 == 0:
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - tic))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= EARLY_STOP_CRITERIUM:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Save model and dictionaries
model.save('recommender_model.h5')
with open('recommender_id2int.json', 'w') as fp:
    json.dump(id2int, fp)
