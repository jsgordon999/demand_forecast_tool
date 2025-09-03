import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks


# ------------ Builder ------------------
def build_tabular_emb_mlp(n_stores, n_skus, cont_dim, d_store=16, d_sku=16, 
    hidden=256, depth=3, lr=1e-3):

    # Inputs
    store_in = layers.Input(shape=(), dtype="int32", name="store_id")
    sku_in   = layers.Input(shape=(), dtype="int32", name="sku_id")
    cont_in  = layers.Input(shape=(cont_dim,), dtype="float32", name="x_cont")

    # Embed categorical input and flatten
    store_emb = layers.Embedding(input_dim=n_stores, output_dim=d_store, name="emb_store")(store_in)
    sku_emb   = layers.Embedding(input_dim=n_skus, output_dim=d_sku, name="emb_sku")(sku_in)
    store_vec = layers.Flatten()(store_emb)
    sku_vec   = layers.Flatten()(sku_emb)

    # Combine all inputs into 1 layer
    x = layers.Concatenate(name="all_inputs")([store_vec, sku_vec, cont_in])

    # MLP trunk
    for i in range(depth):
        x = layers.Dense(hidden, activation="relu", name=f"dense_{i+1}")(x)
        x = layers.Dropout(0.10, name=f"drop_{i+1}")(x)

    # Output: predict log1p(units_sold)
    out = layers.Dense(1, name="out")(x)

    model = keras.Model(inputs=[store_in, sku_in, cont_in], outputs=out, name="NNForecaster")
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse")
    return model

# --------------- LR decay -----------------
class StepDecay(callbacks.Callback):
    def __init__(self, decayevery, decayrate):
        super().__init__()
        self.decayevery = decayevery
        self.decayrate = decayrate
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and self.decayevery > 0 and (epoch % self.decayevery == 0):
            old_lr = float(keras.backend.get_value(self.model.optimizer.lr))
            keras.backend.set_value(self.model.optimizer.lr, old_lr * self.decayrate)


# --------------- Forecaster class -----------
class NNForecaster:
    """
    Tabular demand forecaster with embeddings for store_id and sku_id.
    Predicts log1p(units_sold); use predict_units() to get values in original scale.
    """
    def __init__(self, n_stores, n_skus,
                 hidden=256, depth=3, lr=1e-3,
                 decayevery=0, decayrate=0.5, seed=1234):
        self.n_stores = int(n_stores)
        self.n_skus   = int(n_skus)

        self.hidden = hidden
        self.depth = depth
        self.lr = lr
        self.decayevery = decayevery
        self.decayrate = decayrate

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.model = None
        self.history_ = None

    def fit(self,
            store_train, sku_train, Xcont_train, y_train,
            store_val=None,   sku_val=None,   Xcont_val=None,   y_val=None,
            epochs=100, batch_size=2048, verbose=1, patience=10):
        """
        Inputs:
            store_train, sku_train: int arrays of shape (N,)
            Xcont_train: float array of shape (N, cont_dim)
            y_train: float array of shape (N,) - raw units (not log1p)
        """
        # prepare target: log1p
        y_train_log = np.log1p(y_train.astype(np.float32))
        y_val_log = None
        use_val = (store_val is not None and sku_val is not None and 
                   Xcont_val is not None and y_val is not None)
        if use_val:
            y_val_log = np.log1p(y_val.astype(np.float32))

        cont_dim = Xcont_train.shape[1]
        self.model = build_tabular_emb_mlp(
            n_stores=self.n_stores, n_skus=self.n_skus, cont_dim=cont_dim,
            hidden=self.hidden, depth=self.depth, lr=self.lr
        )

        # Callbacks
        cbs = []
        if self.decayevery and self.decayevery > 0:
            cbs.append(StepDecay(self.decayevery, self.decayrate))
        if use_val:
            cbs.append(callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ))

        hist = self.model.fit(
            [store_train, sku_train, Xcont_train], y_train_log,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=([store_val, sku_val, Xcont_val], y_val_log) if use_val else None,
            callbacks=cbs
        )
        self.history_ = hist.history
        return self

    def predict_units(self, store_ids, sku_ids, Xcont, return_log=False):
        y_log = self.model.predict([store_ids, sku_ids, Xcont], verbose=0).reshape(-1)
        if return_log:
            return y_log
        return np.expm1(y_log)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
