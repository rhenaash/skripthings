import streamlit as st
import os
import random
import numpy as np
import pandas as pd
import gc

# ==========================================================================
# 1. SET SEED GLOBAL & DETERMINISTIC ENVIRONMENT
# ==========================================================================

SEED = 49

# Paksa deterministic
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Paksa CPU only agar sinkron Colab CPU Runtime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Sinkron float precision
tf.keras.backend.set_floatx('float32')

# Batasi thread CPU
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Global seed
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Aktifkan deterministic ops
tf.config.experimental.enable_op_determinism()

# ==========================================================================
# IMPORT LIBRARY
# ==========================================================================

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

from pyswarms.single import GlobalBestPSO

# ==========================================================================
# FUNGSI RESET INTERNAL
# ==========================================================================

def reset_seeds_internal(seed=SEED):

    clear_session()

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    gc.collect()

# ==========================================================================
# STREAMLIT CONFIG
# ==========================================================================

st.set_page_config(
    page_title="Prediksi Emas GRU Hybrid",
    layout="wide"
)

# Sinkronisasi state awal Streamlit
if "initialized" not in st.session_state:
    reset_seeds_internal(SEED)
    st.session_state.initialized = True

st.title("Prediksi Harga Emas dengan Arsitektur GRU")
st.write(
    "Perbandingan model GRU Standar (Adam) "
    "dan GRU-PSO secara deterministik."
)

# ==========================================================================
# UPLOAD FILE
# ==========================================================================

uploaded_file = st.file_uploader(
    "Unggah File Data (.csv / .xlsx)",
    type=["csv", "xlsx"]
)

# ==========================================================================
# MAIN PROCESS
# ==========================================================================

if uploaded_file is not None:

    # ======================================================================
    # LOAD DATA
    # ======================================================================

    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)

    emas = emas[['Tanggal', 'Terakhir']]
    emas.dropna(inplace=True)

    emas['Tanggal'] = pd.to_datetime(
        emas['Tanggal'],
        dayfirst=True
    )

    emas = emas.sort_values(
        by='Tanggal'
    ).reset_index(drop=True)

    st.success("Data berhasil dimuat!")

    with st.expander("Preview Data"):
        st.dataframe(emas.head(10), use_container_width=True)

    # ======================================================================
    # PREPROCESSING
    # ======================================================================

    feature_cols = ['Terakhir']
    target_col = 'Terakhir'

    data_features = emas[feature_cols].values
    data_target = emas[[target_col]].values

    n = len(data_features)
    n_train = int(n * 0.8)

    # Fit scaler HANYA di training
    scaler_X = MinMaxScaler().fit(data_features[:n_train])
    scaler_y = MinMaxScaler().fit(data_target[:n_train])

    Xs = scaler_X.transform(data_features)
    ys = scaler_y.transform(data_target)

    # ======================================================================
    # WINDOWING
    # ======================================================================

    window = 1

    def make_sequences(X_scaled, y_scaled, window):

        X_seq = []
        y_seq = []

        for i in range(window, len(X_scaled)):

            X_seq.append(X_scaled[i-window:i])
            y_seq.append(y_scaled[i])

        return np.array(X_seq), np.array(y_seq)

    X_seq_all, y_seq_all = make_sequences(Xs, ys, window)

    dtrain_end = n_train - window

    X_train = X_seq_all[:dtrain_end]
    y_train = y_seq_all[:dtrain_end]

    X_test = X_seq_all[dtrain_end:]
    y_test = y_seq_all[dtrain_end:]

    X_train = X_train.reshape(
        (X_train.shape[0], X_train.shape[1], 1)
    )

    X_test = X_test.reshape(
        (X_test.shape[0], X_test.shape[1], 1)
    )

    # ======================================================================
    # VALIDATION SPLIT EXPLICIT
    # ======================================================================

    val_ratio = 0.2

    n_tr = X_train.shape[0]
    n_tr_split = int(n_tr * (1 - val_ratio))

    X_tr = X_train[:n_tr_split]
    y_tr = y_train[:n_tr_split]

    X_val = X_train[n_tr_split:]
    y_val = y_train[n_tr_split:]

    # ======================================================================
    # MODEL 1 : GRU ADAM
    # ======================================================================

    def jalankan_training_adam():

        reset_seeds_internal(SEED)

        GS_epoch = 50
        GS_batch = 32
        GS_units = 16
        GS_dropout = 0.0
        GS_LR = 0.001

        clear_session()

        model = Sequential([
            Input(shape=(window, 1)),

            GRU(
                units=GS_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                reset_after=True
            ),

            Dropout(GS_dropout),

            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=GS_LR),
            loss='mse'
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )

        model.fit(
            X_tr,
            y_tr,

            validation_data=(X_val, y_val),

            epochs=GS_epoch,
            batch_size=GS_batch,

            shuffle=False,

            callbacks=[early_stop],

            verbose=0
        )

        y_pred_scaled = model.predict(
            X_test,
            verbose=0
        )

        y_pred_inv = scaler_y.inverse_transform(
            y_pred_scaled
        ).flatten()

        y_test_inv = scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        rmse = np.sqrt(
            mean_squared_error(y_test_inv, y_pred_inv)
        )

        mae = mean_absolute_error(
            y_test_inv,
            y_pred_inv
        )

        mape = (
            mean_absolute_percentage_error(
                y_test_inv,
                y_pred_inv
            ) * 100
        )

        return (
            GS_units,
            GS_LR,
            GS_batch,
            rmse,
            mae,
            mape,
            y_test_inv,
            y_pred_inv
        )

    # ======================================================================
    # MODEL 2 : GRU PSO
    # ======================================================================

    def jalankan_pemodelan_pso_gru():

        # ------------------------------------------------------------------
        # FITNESS FUNCTION
        # ------------------------------------------------------------------

        def make_pso_obj(X_tr, y_tr, X_va, y_va):

            def obj_fn(particles):

                n_particles = particles.shape[0]

                costs = np.zeros(n_particles)

                for i, p in enumerate(particles):

                    units = int(np.round(p[0]))
                    lr = float(p[1])
                    batch = int(np.round(p[2]))
                    dropout = float(p[3])

                    try:

                        reset_seeds_internal(SEED)

                        model = Sequential([

                            Input(
                                shape=(
                                    X_tr.shape[1],
                                    X_tr.shape[2]
                                )
                            ),

                            GRU(
                                units=units,
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                reset_after=True
                            ),

                            Dropout(dropout),

                            Dense(1)

                        ])

                        model.compile(
                            optimizer=Adam(
                                learning_rate=lr
                            ),
                            loss='mse'
                        )

                        model.fit(
                            X_tr,
                            y_tr,

                            epochs=10,

                            batch_size=batch,

                            shuffle=False,

                            verbose=0
                        )

                        yv_pred = model.predict(
                            X_va,
                            verbose=0
                        )

                        yv_pred_orig = scaler_y.inverse_transform(
                            yv_pred
                        ).flatten()

                        yv_true_orig = scaler_y.inverse_transform(
                            y_va.reshape(-1, 1)
                        ).flatten()

                        costs[i] = mean_squared_error(
                            yv_true_orig,
                            yv_pred_orig
                        )

                    except Exception:

                        costs[i] = 1e12

                    clear_session()
                    gc.collect()

                return costs

            return obj_fn

        # ------------------------------------------------------------------
        # PSO CONFIG
        # ------------------------------------------------------------------

        PSOSL_particles = 18
        PSOSL_iters = 5

        PSOSL_options = {
            'c1': 2.0,
            'c2': 2.0,
            'w': 0.7
        }

        PSOSL_bounds = (
            [16, 0.0001, 16, 0.01],
            [128, 0.01, 128, 0.5]
        )

        # ------------------------------------------------------------------
        # RESET SEED BEFORE SWARM
        # ------------------------------------------------------------------

        reset_seeds_internal(SEED)

        np.random.seed(SEED)

        optimizer = GlobalBestPSO(
            n_particles=PSOSL_particles,
            dimensions=4,
            options=PSOSL_options,
            bounds=PSOSL_bounds
        )

        n_particles, dims = optimizer.swarm.position.shape

        optimizer.swarm.pbest_pos_PSOSL = (
            optimizer.swarm.position.copy()
        )

        optimizer.swarm.pbest_cost_PSOSL = np.full(
            n_particles,
            np.inf
        )

        history_gbest_pos_PSOSL = []

        pso_obj = make_pso_obj(
            X_tr,
            y_tr,
            X_val,
            y_val
        )

        # ------------------------------------------------------------------
        # STREAMLIT LOG
        # ------------------------------------------------------------------

        st.write("### Progress Iterasi PSO")

        pso_progress_box = st.empty()

        pso_log_text = ""

        # ------------------------------------------------------------------
        # LOOP PSO
        # ------------------------------------------------------------------

        for it in range(PSOSL_iters):

            costs = pso_obj(
                optimizer.swarm.position
            )

            mask = (
                costs <
                optimizer.swarm.pbest_cost_PSOSL
            )

            optimizer.swarm.pbest_cost_PSOSL[mask] = (
                costs[mask]
            )

            optimizer.swarm.pbest_pos_PSOSL[mask] = (
                optimizer.swarm.position[mask].copy()
            )

            best_idx = np.argmin(
                optimizer.swarm.pbest_cost_PSOSL
            )

            optimizer.swarm.best_cost_PSOSL = (
                optimizer.swarm.pbest_cost_PSOSL[best_idx]
            )

            optimizer.swarm.best_pos_PSOSL = (
                optimizer.swarm.pbest_pos_PSOSL[best_idx].copy()
            )

            history_gbest_pos_PSOSL.append(
                optimizer.swarm.best_pos_PSOSL.copy()
            )

            pso_log_text += (
                f"ITERATION {it+1}\n"
                f"Global Best Loss : "
                f"{optimizer.swarm.best_cost_PSOSL:.6f}\n"
                f"Best Loss Iterasi : "
                f"{np.min(costs):.6f}\n"
                f"{'='*50}\n"
            )

            pso_progress_box.code(
                pso_log_text,
                language="text"
            )

            # --------------------------------------------------------------
            # DETERMINISTIC R1 R2
            # --------------------------------------------------------------

            np.random.seed(SEED + it)

            r1 = np.random.rand(
                *optimizer.swarm.position.shape
            )

            r2 = np.random.rand(
                *optimizer.swarm.position.shape
            )

            optimizer.swarm.velocity = (

                PSOSL_options['w']
                * optimizer.swarm.velocity

                + PSOSL_options['c1']
                * r1
                * (
                    optimizer.swarm.pbest_pos_PSOSL
                    - optimizer.swarm.position
                )

                + PSOSL_options['c2']
                * r2
                * (
                    optimizer.swarm.best_pos_PSOSL
                    - optimizer.swarm.position
                )
            )

            optimizer.swarm.position += (
                optimizer.swarm.velocity
            )

            lb = np.array(PSOSL_bounds[0])
            ub = np.array(PSOSL_bounds[1])

            optimizer.swarm.position = np.clip(
                optimizer.swarm.position,
                lb,
                ub
            )

        # ------------------------------------------------------------------
        # BEST PARAMETER
        # ------------------------------------------------------------------

        best_pos = history_gbest_pos_PSOSL[-1]

        best_units = int(np.round(best_pos[0]))
        best_lr = float(best_pos[1])
        best_batch = int(np.round(best_pos[2]))
        best_dropout = float(best_pos[3])

        # ------------------------------------------------------------------
        # FINAL RETRAIN
        # ------------------------------------------------------------------

        reset_seeds_internal(SEED)

        clear_session()

        model_final = Sequential([

            Input(
                shape=(
                    X_train.shape[1],
                    X_train.shape[2]
                )
            ),

            GRU(
                units=best_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                reset_after=True
            ),

            Dropout(best_dropout),

            Dense(1)

        ])

        model_final.compile(
            optimizer=Adam(
                learning_rate=best_lr
            ),
            loss='mse'
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )

        model_final.fit(

            X_tr,
            y_tr,

            validation_data=(X_val, y_val),

            epochs=50,

            batch_size=best_batch,

            shuffle=False,

            callbacks=[early_stop],

            verbose=0
        )

        # ------------------------------------------------------------------
        # TEST PREDICTION
        # ------------------------------------------------------------------

        y_pred = model_final.predict(
            X_test,
            verbose=0
        )

        y_pred_inv = scaler_y.inverse_transform(
            y_pred
        ).flatten()

        y_test_inv = scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        rmse = np.sqrt(
            mean_squared_error(y_test_inv, y_pred_inv)
        )

        mae = mean_absolute_error(
            y_test_inv,
            y_pred_inv
        )

        mape = (
            mean_absolute_percentage_error(
                y_test_inv,
                y_pred_inv
            ) * 100
        )

        return (
            best_units,
            best_lr,
            best_batch,
            best_dropout,
            rmse,
            mae,
            mape,
            y_test_inv,
            y_pred_inv
        )
