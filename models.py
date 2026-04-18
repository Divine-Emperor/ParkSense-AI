import tensorflow as tf
from config import CFG

# ── BayesianDense (exact notebook port, TF 2.20 keyword-arg fix) ───────────
class BayesianDense(tf.keras.layers.Layer):
    def __init__(self, units, kl_weight=1e-5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.kl_weight  = kl_weight
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        d = int(input_shape[-1])
        # TF 2.20 requires name= and shape= as keyword args
        self.w_mu  = self.add_weight(name="w_mu",  shape=(d, self.units), initializer="glorot_uniform")
        self.w_rho = self.add_weight(name="w_rho", shape=(d, self.units), initializer=tf.constant_initializer(-4.0))
        self.b_mu  = self.add_weight(name="b_mu",  shape=(self.units,),   initializer="zeros")
        self.b_rho = self.add_weight(name="b_rho", shape=(self.units,),   initializer=tf.constant_initializer(-4.0))

    def call(self, inputs, training=False):
        w_sigma = tf.nn.softplus(self.w_rho) + 1e-6
        b_sigma = tf.nn.softplus(self.b_rho) + 1e-6
        if training:
            w = self.w_mu + w_sigma * tf.random.normal(tf.shape(self.w_mu))
            b = self.b_mu + b_sigma * tf.random.normal(tf.shape(self.b_mu))
            kl = 0.5 * tf.reduce_sum(self.w_mu**2 + w_sigma**2 - tf.math.log(w_sigma**2) - 1.0)
            kl += 0.5 * tf.reduce_sum(self.b_mu**2 + b_sigma**2 - tf.math.log(b_sigma**2) - 1.0)
            self.add_loss(self.kl_weight * kl)
        else:
            w, b = self.w_mu, self.b_mu
        out = tf.matmul(inputs, w) + b
        return self.activation(out) if self.activation else out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units, kl_weight=self.kl_weight,
                   activation=tf.keras.activations.serialize(self.activation))
        return cfg

# ── BNN factory (notebook Section 2.2) ────────────────────────────────────
def build_bnn(input_dim, num_classes, hidden_sizes, kl_weight, name="BNN"):
    inp = tf.keras.Input(shape=(input_dim,), name="input")
    x   = inp
    for i, h in enumerate(hidden_sizes):
        x = BayesianDense(h, kl_weight=kl_weight, activation="relu", name=f"bayes_{i}")(x)
    out = BayesianDense(num_classes, kl_weight=kl_weight, name="logits")(x)
    return tf.keras.Model(inp, out, name=name)

# ── Deterministic MLP Builder ──────────────────────────────────────────────
def build_mlp(input_dim, num_classes, hidden_sizes, name="MLP"):
    inp = tf.keras.Input(shape=(input_dim,), name="input")
    x   = inp
    for i, h in enumerate(hidden_sizes):
        x = tf.keras.layers.Dense(h, activation="relu", name=f"dense_{i}")(x)
    out = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return tf.keras.Model(inp, out, name=name)

# ── BNS Model (Improvement 6: Extended TSC in train_step) ─────────────────
class BNS_Model(tf.keras.Model):
    def __init__(self, backbone, mu_hour, mu_week, mu_hol, N_train,
                 lambda_max=0.05, alpha=0.4, beta_w=0.4, gamma=0.2, high_class=2):
        super().__init__()
        self.backbone   = backbone
        self.emp_hour   = tf.constant(mu_hour,          dtype=tf.float32)
        self.emp_week   = tf.constant(mu_week,          dtype=tf.float32)  # shape (168,)
        self.emp_hol    = tf.constant(mu_hol,           dtype=tf.float32)
        self.N_train    = N_train
        self.lambda_max = lambda_max
        self.lambda_t   = 0.0
        self.alpha      = alpha
        self.beta_w     = beta_w
        self.gamma      = gamma
        self.high_class = high_class
        self._loss_tr   = tf.keras.metrics.Mean("loss")
        self._acc_tr    = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        self._tsc_tr    = tf.keras.metrics.Mean("tsc")

    @property
    def metrics(self):
        return [self._loss_tr, self._acc_tr, self._tsc_tr]

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def _ext_tsc(self, logits, hours, dows, hols):
        """Extended TSC: hourly + weekly + holiday components."""
        probs      = tf.nn.softmax(logits)
        high_probs = probs[:, self.high_class]
        h_i32  = tf.cast(hours, tf.int32)
        d_i32  = tf.cast(dows,  tf.int32)
        hf     = tf.cast(hols,  tf.float32)

        # Hourly TSC
        mu_hat_h = []
        for h in range(24):
            mask = tf.cast(tf.equal(h_i32, h), tf.float32)
            mu_hat_h.append(tf.reduce_sum(mask * high_probs) / (tf.reduce_sum(mask) + 1e-6))
        mu_hat_h  = tf.stack(mu_hat_h)
        tsc_hourly = tf.reduce_mean(tf.square(mu_hat_h - self.emp_hour))

        # Weekly TSC (day × hour)
        mu_hat_w = []
        for d in range(7):
            for h in range(24):
                mask = tf.cast(tf.equal(h_i32, h) & tf.equal(d_i32, d), tf.float32)
                mu_hat_w.append(tf.reduce_sum(mask * high_probs) / (tf.reduce_sum(mask) + 1e-6))
        mu_hat_w  = tf.stack(mu_hat_w)
        tsc_weekly = tf.reduce_mean(tf.square(mu_hat_w - self.emp_week))

        # Holiday TSC
        mu_hat_hol = []
        for h in range(24):
            mask = tf.cast(tf.equal(h_i32, h), tf.float32) * hf
            mu_hat_hol.append(tf.reduce_sum(mask * high_probs) / (tf.reduce_sum(mask) + 1e-6))
        mu_hat_hol  = tf.stack(mu_hat_hol)
        tsc_holiday = tf.reduce_mean(tf.square(mu_hat_hol - self.emp_hol))

        return self.alpha * tsc_hourly + self.beta_w * tsc_weekly + self.gamma * tsc_holiday

    def train_step(self, data):
        # x_full = [features(13) | hour | dow | hol_flag]
        x_full, y = data
        x_feat = x_full[:, :13]
        hours  = x_full[:, 13]
        dows   = x_full[:, 14]
        hols   = x_full[:, 15]

        with tf.GradientTape() as tape:
            logits    = self.backbone(x_feat, training=True)
            ce        = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
            kl_losses = self.backbone.losses
            kl_scaled = tf.add_n(kl_losses) / self.N_train if kl_losses else 0.0
            tsc       = self._ext_tsc(logits, hours, dows, hols)
            loss      = ce + kl_scaled + self.lambda_t * tsc

        grads = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_variables))
        self._loss_tr.update_state(loss)
        self._acc_tr.update_state(y, logits)
        self._tsc_tr.update_state(tsc)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_full, y = data
        x_feat = x_full[:, :13]
        hours  = x_full[:, 13]
        dows   = x_full[:, 14]
        hols   = x_full[:, 15]
        logits = self.backbone(x_feat, training=False)
        tsc    = self._ext_tsc(logits, hours, dows, hols)
        ce     = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        self._loss_tr.update_state(ce)
        self._acc_tr.update_state(y, logits)
        self._tsc_tr.update_state(tsc)
        return {m.name: m.result() for m in self.metrics}

class LambdaWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=5):
        super().__init__()
        self.warmup = warmup_epochs
    def on_epoch_begin(self, epoch, logs=None):
        lam = min(epoch / max(self.warmup, 1), 1.0) * self.model.lambda_max
        self.model.lambda_t = lam
