from keras.models import Model
from keras.layers import (Input, Lambda, Embedding, concatenate,
                          Conv1D, Dense, Dropout, Flatten)

# -------------------------------------------------------------------
#   Env Setting
# -------------------------------------------------------------------
TIMESTEPS = 200

# -------------------------------------------------------------------
#   CNN Model
# -------------------------------------------------------------------
latent_dim = 32

# Define input
seq_in = Input(shape=(TIMESTEPS, 1))
is0_in = Input(shape=(TIMESTEPS, 1))
promo_in = Input(shape=(TIMESTEPS + 16, 1))
# yearAgo_in = Input(shape=(TIMESTEPS + 16, 1))
quarterAgo_in = Input(shape=(TIMESTEPS + 16, 1))
item_mean_in = Input(shape=(TIMESTEPS, 1))
store_mean_in = Input(shape=(TIMESTEPS, 1))
weekday_in = Input(shape=(TIMESTEPS + 16,), dtype='uint8')
dom_in = Input(shape=(TIMESTEPS + 16,), dtype='uint8')
# dom_embed_encode = Embedding(31, 4, input_length=TIMESTEPS + 16)(dom_in)

# aux input
cat_features = Input(shape=(6,))
item_family = Lambda(lambda x: x[:, 0, None])(cat_features)
# item_class = Lambda(lambda x: x[:, 1, None])(cat_features)
item_perish = Lambda(lambda x: x[:, 2, None])(cat_features)
store_nbr = Lambda(lambda x: x[:, 3, None])(cat_features)
store_cluster = Lambda(lambda x: x[:, 4, None])(cat_features)
store_type = Lambda(lambda x: x[:, 5, None])(cat_features)

family_embed = Embedding(33, 8, input_length=1)(item_family)
store_embed = Embedding(54, 8, input_length=1)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=1)(store_cluster)
type_embed = Embedding(5, 2, input_length=1)(store_type)

# for promo_in, promo_in includes future onpromotion variables
encode_slice = Lambda(lambda x: x[:, :TIMESTEPS, :])

x_in = concatenate([seq_in, encode_slice(promo_in), item_mean_in], axis=2)

# Define network
c1 = Conv1D(latent_dim, 2, dilation_rate=1, padding='causal',
            activation='relu')(x_in)
c2 = Conv1D(latent_dim, 2, dilation_rate=2, padding='causal',
            activation='relu')(c1)
c2 = Conv1D(latent_dim, 2, dilation_rate=4, padding='causal',
            activation='relu')(c2)
c2 = Conv1D(latent_dim, 2, dilation_rate=8, padding='causal',
            activation='relu')(c2)

c3 = concatenate([c1, c2])

conv_out = Conv1D(8, 1, activation='relu')(c3)
conv_out = Dropout(0.25)(conv_out)
conv_out = Flatten()(conv_out)

decode_slice = Lambda(lambda x: x[:, TIMESTEPS:, :])
promo_pred = decode_slice(promo_in)

# Raw sequence in results overfitting!!!
dnn_out = Dense(512, activation='relu')(Flatten()(seq_in))
dnn_out = Dense(256, activation='relu')(dnn_out)
# dnn_out = BatchNormalization()(dnn_out)
dnn_out = Dropout(0.25)(dnn_out)

x = concatenate([conv_out, dnn_out,
                 Flatten()(promo_pred), Flatten()(family_embed),
                 Flatten()(store_embed), Flatten()(cluster_embed),
                 Flatten()(type_embed), item_perish])
# x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
# x = Dense(256, activation='relu')(x)
# x = BatchNormalization()(x)
# x = concatenate([x, seq_in])
output = Dense(16, activation='relu')(x)

model = Model(
    [seq_in, is0_in, promo_in, quarterAgo_in, weekday_in, dom_in,
     cat_features, item_mean_in, store_mean_in],
    output
)
