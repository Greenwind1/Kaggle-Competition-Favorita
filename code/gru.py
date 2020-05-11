import os, gc
import pandas as pd
from datetime import date
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from time import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Lambda,
                                     concatenate, Reshape,
                                     Conv1D, Dense)
from tensorflow.keras.layers import GRU

from util.Utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tf warnings

# -------------------------------------------------------------------
#   Env Setting
# -------------------------------------------------------------------
TIMESTEPS = 365
EPOCH = 15

# -------------------------------------------------------------------
#   Load Dataset
# -------------------------------------------------------------------
df_name = './input/unstack_train.f'
promo_name = './input/unstack_promo.f'
df, promo_df, items, stores = load_unstack(df_name, promo_name)
print('df shape =', df.shape, '\npromo_df shape =', promo_df.shape)
print('data span:', df.columns[0].strftime('%Y-%m-%d'),
      '-', df.columns[-1].strftime('%Y-%m-%d'), '\n')

df_data_range = pd.date_range(date(2017, 1, 1), date(2017, 8, 15))
promo_df = promo_df[df[df_data_range].max(axis=1) > 0]
promo_df = promo_df.astype('int')
df = df[df[df_data_range].max(axis=1) > 0]
print('df shape =', df.shape, '\npromo_df shape =', promo_df.shape)
print('data span:', df.columns[0].strftime('%Y-%m-%d'),
      '-', df.columns[-1].strftime('%Y-%m-%d'), '\n')

df_test = pd.read_csv(
    './input/test.csv',
    usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]
).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))

df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]
print('df shape =', df.shape, '\npromo_df shape =', promo_df.shape)
print('data span:', df.columns[0].strftime('%Y-%m-%d'),
      '-', df.columns[-1].strftime('%Y-%m-%d'))

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test
gc.collect()

# -------------------------------------------------------------------
#   Prepare Dataset
#   train_generator and create_dataset return the following,
#   X0: X
#   X1: is0
#   X2: promo
#   X3: df_quarter_ago
#   X4: weekday
#   X5: dom
#   X6: cat_features
#   X7: item_mean
#   X8: store_mean
# -------------------------------------------------------------------
train_data = train_generator(
    df, promo_df, items, stores,
    timesteps=TIMESTEPS,
    first_pred_start=date(2017, 7, 9),
    n_range=7 * 8,
    day_skip=1,
    batch_size=1000,
    aux_as_tensor=True,
    reshape_output=2
)

Xval, Yval = create_dataset(df, promo_df, items, stores,
                            timesteps=TIMESTEPS,
                            first_pred_start=date(2017, 7, 26),
                            aux_as_tensor=True,  # cat: (160964, 216, 6)
                            reshape_output=2)

Xtest, _ = create_dataset(df, promo_df, items, stores,
                          timesteps=TIMESTEPS,
                          first_pred_start=date(2017, 8, 16),
                          aux_as_tensor=True,
                          is_train=False,
                          reshape_output=2)

# w = (Xval[7][:, 0, 2] * 0.25 + 1) / (Xval[7][:, 0, 2] * 0.25 + 1).mean()
w = (Xval[7][:, 2] * 0.25 + 1) / (Xval[7][:, 2] * 0.25 + 1).mean()

del df, promo_df
gc.collect()

# -------------------------------------------------------------------
#   Seq2Seq GRU Model
# -------------------------------------------------------------------
latent_dim = 100

# Define input
seq_in = Input(shape=(TIMESTEPS, 1))
is0_in = Input(shape=(TIMESTEPS, 1))
promo_in = Input(shape=(TIMESTEPS + 16, 1))
# yearAgo_in = Input(shape=(TIMESTEPS + 16, 1))
quarterAgo_in = Input(shape=(TIMESTEPS + 16, 1))
item_mean_in = Input(shape=(TIMESTEPS, 1))
store_mean_in = Input(shape=(TIMESTEPS, 1))
weekday_in = Input(shape=(TIMESTEPS + 16,), dtype='uint8')
weekday_embed_encode = Embedding(7, 4, input_length=TIMESTEPS + 16)(weekday_in)
dom_in = Input(shape=(TIMESTEPS + 16,), dtype='uint8')
# dom_embed_encode = Embedding(31, 4, input_length=TIMESTEPS + 16)(dom_in)

# aux input
cat_features = Input(shape=(TIMESTEPS + 16, 6))
item_family = Lambda(lambda x: x[:, :, 0])(cat_features)
# item_class = Lambda(lambda x: x[:, :, 1])(cat_features)
item_perish = Lambda(lambda x: x[:, :, 2])(cat_features)
store_nbr = Lambda(lambda x: x[:, :, 3])(cat_features)
store_cluster = Lambda(lambda x: x[:, :, 4])(cat_features)
store_type = Lambda(lambda x: x[:, :, 5])(cat_features)

# store_in = Input(shape=(timesteps+16,), dtype='uint8')
family_embed = Embedding(33, 8, input_length=TIMESTEPS + 16)(item_family)
# class_embed = Embedding(337, 8, input_length=TIMESTEPS + 16)(item_class)
store_embed = Embedding(54, 8, input_length=TIMESTEPS + 16)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=TIMESTEPS + 16)(store_cluster)
type_embed = Embedding(5, 2, input_length=TIMESTEPS + 16)(store_type)

# Encoder
encode_slice = Lambda(lambda x: x[:, :TIMESTEPS, :])
encode_features = concatenate([
    promo_in,
    # yearAgo_in,
    quarterAgo_in,
    weekday_embed_encode,
    family_embed,
    Reshape((TIMESTEPS + 16, 1))(item_perish),
    store_embed,
    cluster_embed,
    type_embed,
], axis=2)
encode_features = encode_slice(encode_features)

conv_in = Conv1D(filters=4, kernel_size=5, padding='same')(seq_in)  # 1ch to 4ch

x_encode = concatenate([seq_in, encode_features, conv_in, item_mean_in], axis=2)
print('Input dim:', x_encode.shape)

# _, h = CuDNNGRU(latent_dim, return_state=True)(x_encode)
_, h = GRU(units=latent_dim, return_state=True)(x_encode)

# Connector
h = Dense(latent_dim, activation='tanh')(h)
# h1 = Dense(latent_dim, activation='tanh')(h1)
# h2 = Dense(latent_dim, activation='tanh')(h2)

# Decoder
previous_x = Lambda(lambda x: x[:, -1, :])(seq_in)

decode_slice = Lambda(lambda x: x[:, TIMESTEPS:, :])
decode_features = concatenate([
    promo_in,
    # yearAgo_in,
    quarterAgo_in,
    weekday_embed_encode,
    family_embed,
    Reshape((TIMESTEPS + 16, 1))(item_perish),
    store_embed,
    cluster_embed,
    type_embed,
], axis=2)
decode_features = decode_slice(decode_features)

# decoder = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
decoder = GRU(latent_dim, return_state=True, return_sequences=False)
decoder_dense2 = Dense(1, activation='relu')
slice_at_t = Lambda(lambda x: tf.slice(x, [0, i, 0], [-1, 1, -1]))

for i in range(16):
    previous_x = Reshape((1, 1))(previous_x)
    features_t = slice_at_t(decode_features)

    decode_input = concatenate([previous_x, features_t], axis=2)
    output_x, h = decoder(decode_input, initial_state=h)
    # aux input
    output_x = decoder_dense2(output_x)

    # gather outputs
    if i == 0:
        decoder_outputs = output_x
    elif i > 0:
        # noinspection PyUnboundLocalVariable
        decoder_outputs = concatenate([decoder_outputs, output_x])

    previous_x = output_x

# noinspection PyUnboundLocalVariable
model = Model([
    seq_in,
    is0_in,
    promo_in,
    # yearAgo_in,
    quarterAgo_in,
    weekday_in,
    dom_in,
    cat_features,
    item_mean_in,
    store_mean_in
], decoder_outputs)

model.compile(optimizer='adam', loss='mean_squared_error')

# -------------------------------------------------------------------
#   Training
# -------------------------------------------------------------------
st = time()
history = model.fit_generator(
    train_data,
    steps_per_epoch=1500,
    # workers=5,
    # use_multiprocessing=True,
    epochs=EPOCH,
    verbose=2,
    validation_data=(Xval, Yval, w[:, 0])
)
print(f'elapsed time: {np.round((time() - st) / 60, 1)} (m)')

# -------------------------------------------------------------------
#   Validation
# -------------------------------------------------------------------
val_pred = model.predict(Xval)
print('-' * 50)
cal_score(Yval, val_pred)

# -------------------------------------------------------------------
#   Inference
# -------------------------------------------------------------------
test_pred = model.predict(Xtest)
make_submission(df_index, test_pred,
                f'./sub/dnn/gru_epoch{EPOCH}.csv')
