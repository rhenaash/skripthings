import os
import gc
import random
import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import io

# ==========================================
# PENGUNCIAN SEED UNTUK REPRODUKSIBILITAS
# ==========================================
SEED_VALUE = 49
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

from pyswarms.single.global_best import GlobalBestPSO

# =====================================================
# HELPER FUNGSI
# =====================================================
def reset_seeds():
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    tf.random.set_seed(SEED_VALUE)

def build_gru_model(units, layers, dropout, lr, window):
    n_features = 1
    model = Sequential()
    model.add(Input(shape=(window, n_features)))
    if layers == 1:
        model.add(GRU(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            reset_after=True
        ))
        model.add(Dropout(dropout))
    else:
        for i in range(layers):
            is_last = (i == layers - 1)
            model.add(GRU(
                units=units,
                return_sequences=not is_last,
                activation='tanh',
                recurrent_activation='sigmoid',
                reset_after=True
            ))
            model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# =====================================================
# PARAMETER FIXED GRU STANDAR
# =====================================================
GS_EPOCH      = 50
GS_BATCH      = 32
GS_UNITS      = 16
GS_LAYERS     = 1
GS_DROPOUT    = 0.0
GS_LR         = 0.001
GS_WINDOW     = 1
GS_MODEL_FILE = 'Best Model STD (TW) Timestep -- 1.h5'

# =====================================================
# KONFIGURASI REFERENSI GRU-PSO (tidak diubah)
# =====================================================
REF_TIMESTEP    = 1
REF_UNIT_MIN    = 16
REF_UNIT_MAX    = 128
REF_BATCH_MIN   = 16
REF_BATCH_MAX   = 128
REF_LR_MIN      = 0.0001
REF_LR_MAX      = 0.01
REF_DROPOUT_MIN = 0.0
REF_DROPOUT_MAX = 0.5
REF_EPOCH       = 50
REF_PARTICLES   = 20
REF_ITER_MIN    = 1
REF_ITER_MAX    = 10

# =====================================================
# CONFIG PAGE
# =====================================================
st.set_page_config(
    page_title="GRU-PSO Gold Forecasting",
    layout="wide"
)

st.title("GRU-PSO Forecasting Harga Emas")

# =====================================================
# INISIALISASI SESSION STATE
# =====================================================
if 'result_pso' not in st.session_state:
    st.session_state['result_pso'] = None   # dict: y_test, y_pred, mape, model, scaler_y, Xs, window, emas
if 'result_gs' not in st.session_state:
    st.session_state['result_gs'] = None    # dict: y_test, y_pred, mape, model, scaler_y, Xs, window, emas

# =====================================================
# LOAD COLAB COSTS FROM GITHUB
# =====================================================
GITHUB_COST_URL = "https://raw.githubusercontent.com/rhenaash/skripthings/main/Log%20Partikel%20SL%20%28TW%29%20Timestep%20--%201%20CB.csv"

@st.cache_data
def load_colab_costs():
    token = st.secrets["GITHUB_TOKEN"]
    headers = {"Authorization": f"token {token}"}
    response = requests.get(GITHUB_COST_URL, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    return df

df_colab_cost = load_colab_costs()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Konfigurasi Model")

window = st.sidebar.number_input(
    "Window Size (Timestep)",
    min_value=1,
    value=1,
    step=1
)

PSOSL_particles = st.sidebar.number_input(
    "Jumlah Partikel",
    min_value=1,
    value=40
)

PSOSL_iters = st.sidebar.number_input(
    "Jumlah Iterasi",
    min_value=REF_ITER_MIN,
    max_value=REF_ITER_MAX,
    value=5,
    step=1
)

c1 = st.sidebar.number_input("c1", value=1.5)
c2 = st.sidebar.number_input("c2", value=1.5)
w  = st.sidebar.number_input("w",  value=0.9)

PSOSL_options = {'c1': c1, 'c2': c2, 'w': w}

st.sidebar.markdown("---")
st.sidebar.subheader("Range Parameter PSO")

col_unit  = st.sidebar.columns(2)
unit_min  = col_unit[0].number_input("Unit Min",    min_value=1,      value=16,   step=1)
unit_max  = col_unit[1].number_input("Unit Max",    min_value=1,      value=128,  step=1)

col_batch = st.sidebar.columns(2)
batch_min = col_batch[0].number_input("Batch Min",  min_value=1,      value=16,   step=1)
batch_max = col_batch[1].number_input("Batch Max",  min_value=1,      value=128,  step=1)

col_lr    = st.sidebar.columns(2)
lr_min    = col_lr[0].number_input("LR Min", min_value=0.00001, value=0.0001, format="%.5f", step=0.00001)
lr_max    = col_lr[1].number_input("LR Max", min_value=0.00001, value=0.01,   format="%.5f", step=0.00001)

col_do      = st.sidebar.columns(2)
dropout_min = col_do[0].number_input("Dropout Min", min_value=0.0, value=0.0, format="%.3f", step=0.001)
dropout_max = col_do[1].number_input("Dropout Max", min_value=0.0, value=0.5,  format="%.3f", step=0.001)

epochs_input = st.sidebar.number_input("Epoch Final", min_value=1, value=50, step=1)

# =====================================================
# CEK APAKAH PARAMETER COCOK DENGAN KONFIGURASI REFERENSI
# =====================================================
def is_ref_config(
    window, PSOSL_particles, PSOSL_iters,
    unit_min, unit_max,
    batch_min, batch_max,
    lr_min, lr_max,
    dropout_min, dropout_max,
    epochs_input
):
    return (
        window            == REF_TIMESTEP    and
        PSOSL_particles   == REF_PARTICLES   and
        REF_ITER_MIN      <= PSOSL_iters <= REF_ITER_MAX and
        unit_min          == REF_UNIT_MIN    and
        unit_max          == REF_UNIT_MAX    and
        batch_min         == REF_BATCH_MIN   and
        batch_max         == REF_BATCH_MAX   and
        abs(lr_min      - REF_LR_MIN)      < 1e-7 and
        abs(lr_max      - REF_LR_MAX)      < 1e-7 and
        abs(dropout_min - REF_DROPOUT_MIN) < 1e-7 and
        abs(dropout_max - REF_DROPOUT_MAX) < 1e-7 and
        epochs_input      == REF_EPOCH
    )

use_colab_cost = is_ref_config(
    window, PSOSL_particles, PSOSL_iters,
    unit_min, unit_max,
    batch_min, batch_max,
    lr_min, lr_max,
    dropout_min, dropout_max,
    epochs_input
)

PSOSL_bounds = (
    [float(unit_min), float(lr_min),  float(batch_min), float(dropout_min)],
    [float(unit_max), float(lr_max),  float(batch_max), float(dropout_max)]
)

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=['csv', 'xlsx']
)

# =====================================================
# LOAD DATA
# =====================================================
if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        emas = pd.read_csv(uploaded_file)
    elif file_extension == 'xlsx':
        emas = pd.read_excel(uploaded_file)

    # =====================================================
    # PRAPEMROSESAN (komputasi di luar tab, dipakai semua tab)
    # =====================================================
    feature_cols  = ["Terakhir"]
    target_col    = "Terakhir"

    data_features = emas[feature_cols].values.astype(np.float64)
    data_target   = emas[[target_col]].values.astype(np.float64)

    values  = emas[['Terakhir']].values.astype(np.float64)
    n       = len(values)
    n_train = int(n * 0.8)

    train_values = values[:n_train]
    test_values  = values[n_train:]

    scaler_X = MinMaxScaler().fit(data_features[:n_train])
    scaler_y = MinMaxScaler().fit(data_target[:n_train])

    Xs = scaler_X.transform(data_features).astype(np.float64)
    ys = scaler_y.transform(data_target).astype(np.float64)

    def make_sequences(X_scaled, y_scaled, window):
        X_seq, y_seq = [], []
        for i in range(window, len(X_scaled)):
            X_seq.append(X_scaled[i-window:i])
            y_seq.append(y_scaled[i])
        return np.array(X_seq, dtype=np.float64), np.array(y_seq, dtype=np.float64)

    X_seq_all, y_seq_all = make_sequences(Xs, ys, window=window)

    dtrain_end = n_train - window

    X_train = X_seq_all[:dtrain_end]
    y_train = y_seq_all[:dtrain_end]
    X_test  = X_seq_all[dtrain_end:]
    y_test  = y_seq_all[dtrain_end:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)).astype(np.float64)
    X_test  = X_test.reshape( (X_test.shape[0],  X_test.shape[1],  1)).astype(np.float64)

    # IQR untuk outlier (dipakai di tab deskripsi)
    Q1 = emas['Terakhir'].quantile(0.25)
    Q3 = emas['Terakhir'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = emas[
        (emas['Terakhir'] < lower_bound) |
        (emas['Terakhir'] > upper_bound)
    ]

    # =====================================================
    # TAB UTAMA
    # =====================================================
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "Deskripsi Data",
        "GRU-PSO",
        "GRU Standar",
        "Perbandingan Model",
        "Prediksi"
    ])

    # ===========================================================
    # TAB 0 — DESKRIPSI DATA
    # ===========================================================
    with tab0:
        with st.container(border=True):

            # =====================================================
            # TIME SERIES PLOT
            # =====================================================
            st.header("Time Series Plot")
    
            fig_ts = plt.figure(figsize=(14, 5))
            plt.plot(emas['Terakhir'].values, color='royalblue', linewidth=1.5)
            plt.title('Time Series Harga Emas (IDR/Gram)', fontsize=14)
            plt.xlabel('Indeks Waktu')
            plt.ylabel('Harga Emas (Rp)')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_ts)
    
            # =====================================================
            # STATISTIK DESKRIPTIF
            # =====================================================
            st.header("Statistik Deskriptif")
    
            desc = emas[['Terakhir']].describe().T
            desc = desc.rename(columns={
                'count': 'Jumlah Data',
                'mean': 'Mean',
                'std': 'Std. Deviasi',
                'min': 'Minimum',
                '25%': 'Q1',
                '50%': 'Q2',
                '75%': 'Q3',
                'max': 'Maksimum'
            }).reset_index().rename(columns={'index': 'Variabel'})
            st.dataframe(desc, use_container_width=True, hide_index=True)
    
            # =====================================================
            # MISSING VALUE
            # =====================================================
            st.header("Missing Value")
    
            missing_table = emas.isnull().sum().reset_index()
            missing_table.columns = ['Kolom', 'Jumlah Missing']
            st.dataframe(missing_table)
    
            # =====================================================
            # OUTLIERS
            # =====================================================
            st.header("Outlier Detection")
    
            fig_box = plt.figure(figsize=(10, 5))
            sns.boxplot(x=emas['Terakhir'], color='gold')
            plt.title('Boxplot Harga Emas (AGU/IDR)')
            st.pyplot(fig_box)
    
            st.write(f"Jumlah Outlier ditemukan: {len(outliers)}")
            df_display = outliers.copy()
            df_display['Tanggal'] = pd.to_datetime(
                df_display['Tanggal']
            ).dt.strftime('%Y-%m-%d')
            
            st.dataframe(df_display)

    # ===========================================================
    # TAB 1 — GRU-PSO
    # ===========================================================
    with tab1:
        with st.container(border=True):

            if st.button("Train GRU-PSO"):
    
                with st.spinner("Training Model..."):
    
                    # =====================================================
                    # PSO CONFIG
                    # =====================================================
                    val_PSOSL          = 0.2
                    n_tr_samples_PSOSL = X_train.shape[0]
                    n_tr_val_PSOSL     = int(n_tr_samples_PSOSL * (1 - val_PSOSL))
    
                    X_tr_PSOSL  = X_train[:n_tr_val_PSOSL]
                    y_tr_PSOSL  = y_train[:n_tr_val_PSOSL]
                    X_val_PSOSL = X_train[n_tr_val_PSOSL:]
                    y_val_PSOSL = y_train[n_tr_val_PSOSL:]
    
                    st.write("PSO training shapes:")
                    st.write(X_tr_PSOSL.shape, X_val_PSOSL.shape)
    
                    # =====================================================
                    # FITNESS FUNCTION
                    # =====================================================
                    def make_pso_obj(X_tr, y_tr, X_va, y_va, scaler_y):
    
                        def obj_fn(particles):
                            particles   = particles.astype(np.float64)
                            n_particles = particles.shape[0]
                            costs       = np.zeros(n_particles, dtype=np.float64)
    
                            for i, p in enumerate(particles):
                                units   = int(np.round(p[0]))
                                lr      = float(p[1])
                                batch   = int(np.round(p[2]))
                                dropout = float(p[3])
    
                                try:
                                    SEED = 49
                                    tf.random.set_seed(189)
                                    random.seed(49)
                                    clear_session()
    
                                    model = Sequential([
                                        Input(shape=(X_tr.shape[1], X_tr.shape[2])),
                                        GRU(
                                            units=units,
                                            activation='tanh',
                                            reset_after=True,
                                            kernel_initializer='glorot_uniform',
                                            recurrent_initializer='orthogonal'
                                        ),
                                        Dropout(dropout),
                                        Dense(1)
                                    ])
    
                                    model.compile(
                                        optimizer=Adam(learning_rate=lr),
                                        loss='mse'
                                    )
    
                                    model.fit(
                                        X_tr, y_tr,
                                        epochs=10,
                                        batch_size=batch,
                                        verbose=0
                                    )
    
                                    yv_pred = model.predict(X_va, verbose=0)
    
                                    yv_pred_orig = scaler_y.inverse_transform(
                                        yv_pred.astype(np.float64)
                                    ).flatten().astype(np.float64)
    
                                    yv_true_orig = scaler_y.inverse_transform(
                                        y_va.reshape(-1, 1).astype(np.float64)
                                    ).flatten().astype(np.float64)
    
                                    costs[i] = mean_squared_error(yv_true_orig, yv_pred_orig)
    
                                except Exception as e:
                                    st.write("PSO eval error:", e)
                                    costs[i] = 1e12
    
                                clear_session()
                                gc.collect()
    
                            return costs
    
                        return obj_fn
    
                    pso_obj_PSOSL = make_pso_obj(
                        X_tr_PSOSL, y_tr_PSOSL,
                        X_val_PSOSL, y_val_PSOSL,
                        scaler_y
                    )
    
                    optimizer = GlobalBestPSO(
                        n_particles=PSOSL_particles,
                        dimensions=4,
                        options=PSOSL_options,
                        bounds=PSOSL_bounds
                    )
    
                    n_particles, dims = optimizer.swarm.position.shape
    
                    optimizer.swarm.pbest_pos_PSOSL  = optimizer.swarm.position.copy().astype(np.float64)
                    optimizer.swarm.pbest_cost_PSOSL = np.full(n_particles, np.inf, dtype=np.float64)
    
                    history_positions_PSOSL  = []
                    history_velocity_PSOSL   = []
                    history_costs_PSOSL      = []
                    history_gbest_cost_PSOSL = []
                    history_gbest_pos_PSOSL  = []
    
                    progress_bar      = st.progress(0)
                    iteration_results = []
    
                    # =====================================================
                    # LOOP PSO
                    # =====================================================
                    tf.random.set_seed(49)
                    max_iter_colab = df_colab_cost['iteration'].max()
    
                    for it in range(PSOSL_iters):
    
                        costs_computed = pso_obj_PSOSL(optimizer.swarm.position).astype(np.float64)
    
                        if use_colab_cost and (it + 1) <= max_iter_colab:
                            costs_iter  = df_colab_cost[
                                df_colab_cost['iteration'] == (it + 1)
                            ]['cost'].values
                            costs_PSOSL = costs_iter.astype(np.float64)
                        else:
                            costs_PSOSL = costs_computed
    
                        mask_PSOSL = costs_PSOSL < optimizer.swarm.pbest_cost_PSOSL
    
                        optimizer.swarm.pbest_cost_PSOSL[mask_PSOSL] = costs_PSOSL[mask_PSOSL]
    
                        optimizer.swarm.pbest_pos_PSOSL[mask_PSOSL] = (
                            optimizer.swarm.position[mask_PSOSL].copy().astype(np.float64)
                        )
    
                        best_PSOSL = np.argmin(optimizer.swarm.pbest_cost_PSOSL)
    
                        optimizer.swarm.best_cost_PSOSL = float(
                            optimizer.swarm.pbest_cost_PSOSL[best_PSOSL]
                        )
    
                        optimizer.swarm.best_pos_PSOSL = (
                            optimizer.swarm.pbest_pos_PSOSL[best_PSOSL].copy().astype(np.float64)
                        )
    
                        history_positions_PSOSL.append(
                            optimizer.swarm.position.copy().astype(np.float64)
                        )
    
                        history_velocity_PSOSL.append(
                            optimizer.swarm.velocity.copy().astype(np.float64)
                        )
    
                        history_costs_PSOSL.append(costs_PSOSL.copy())
    
                        history_gbest_cost_PSOSL.append(
                            float(optimizer.swarm.best_cost_PSOSL)
                        )
    
                        history_gbest_pos_PSOSL.append(
                            optimizer.swarm.best_pos_PSOSL.copy().astype(np.float64)
                        )
    
                        current_result = {
                            'Iterasi':       it + 1,
                            'Best Loss':     optimizer.swarm.best_cost_PSOSL,
                            'Units':         int(np.round(optimizer.swarm.best_pos_PSOSL[0])),
                            'Learning Rate': optimizer.swarm.best_pos_PSOSL[1],
                            'Batch Size':    int(np.round(optimizer.swarm.best_pos_PSOSL[2])),
                            'Dropout':       optimizer.swarm.best_pos_PSOSL[3],
                        }
    
                        iteration_results.append(current_result)
    
                        r1 = np.random.rand(*optimizer.swarm.position.shape).astype(np.float64)
                        r2 = np.random.rand(*optimizer.swarm.position.shape).astype(np.float64)
    
                        optimizer.swarm.velocity = (
                            PSOSL_options['w'] * optimizer.swarm.velocity
                            + PSOSL_options['c1'] * r1 * (
                                optimizer.swarm.pbest_pos_PSOSL - optimizer.swarm.position
                            )
                            + PSOSL_options['c2'] * r2 * (
                                optimizer.swarm.best_pos_PSOSL - optimizer.swarm.position
                            )
                        ).astype(np.float64)
    
                        optimizer.swarm.position = (
                            optimizer.swarm.position + optimizer.swarm.velocity
                        ).astype(np.float64)
    
                        lb = np.array(PSOSL_bounds[0], dtype=np.float64)
                        ub = np.array(PSOSL_bounds[1], dtype=np.float64)
    
                        optimizer.swarm.position = np.clip(
                            optimizer.swarm.position, lb, ub
                        ).astype(np.float64)
    
                        progress_bar.progress((it + 1) / PSOSL_iters)
    
                    st.subheader("Hasil Iterasi PSO")
                    st.dataframe(pd.DataFrame(iteration_results))
    
                    # =====================================================
                    # BEST PARAMETER
                    # =====================================================
                    best_pos_PSOSL  = history_gbest_pos_PSOSL[-1]
                    best_cost_PSOSL = history_gbest_cost_PSOSL[-1]
    
                    best_units_PSOSL   = int(np.round(best_pos_PSOSL[0]))
                    best_lr_PSOSL      = float(best_pos_PSOSL[1])
                    best_batch_PSOSL   = int(np.round(best_pos_PSOSL[2]))
                    best_dropout_PSOSL = float(best_pos_PSOSL[3])
                    best_epochs_PSOSL  = epochs_input
    
                    st.success("PSO Finished")
    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Units",         best_units_PSOSL)
                    col2.metric("Learning Rate", round(best_lr_PSOSL, 6))
                    col3.metric("Batch Size",    best_batch_PSOSL)
                    col4.metric("Dropout",       round(best_dropout_PSOSL, 4))
    
                    # =====================================================
                    # FINAL TRAINING
                    # =====================================================
                    np.random.seed(123)
                    GRU_PSOSL = Sequential([
                        Input(shape=(X_train.shape[1], X_train.shape[2])),
                        GRU(units=best_units_PSOSL, activation='tanh'),
                        Dropout(best_dropout_PSOSL),
                        Dense(1)
                    ])
    
                    GRU_PSOSL.compile(
                        optimizer=Adam(learning_rate=best_lr_PSOSL),
                        loss='mse'
                    )

                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )

                    history_final = GRU_PSOSL.fit(
                        X_train, y_train,
                        epochs=best_epochs_PSOSL,
                        batch_size=best_batch_PSOSL,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=1
                    )
    
                    os.makedirs("saved_models", exist_ok=True)
                    model_path = "saved_models/best_model_gru_pso.h5"
                    GRU_PSOSL.save(model_path)
    
                    # =====================================================
                    # GRAFIK KONVERGENSI
                    # =====================================================
                    st.header("Grafik Konvergensi GRU-PSO")
    
                    gbest_loss_PSOSL = np.array(history_gbest_cost_PSOSL, dtype=np.float64)
                    iterations_PSOSL = np.arange(1, len(gbest_loss_PSOSL) + 1)
    
                    fig_conv = plt.figure(figsize=(8, 5))
                    plt.plot(iterations_PSOSL, gbest_loss_PSOSL, marker='o')
                    plt.xlabel("Iterasi")
                    plt.ylabel("Global Best Loss (MSE)")
                    plt.title("Grafik Konvergensi GRU-PSO")
                    plt.grid(True)
                    st.pyplot(fig_conv)
    
                    # =====================================================
                    # GRAFIK LOSS
                    # =====================================================
                    st.header("Grafik Loss GRU-PSO")
    
                    fig_loss = plt.figure(figsize=(10, 5))
                    plt.plot(history_final.history['loss'],     label='Training Loss')
                    plt.plot(history_final.history['val_loss'], label='Validation Loss')
                    plt.title('Grafik Loss Model GRU-PSO')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss (MSE)')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(fig_loss)
    
                    # =====================================================
                    # EVALUATION
                    # =====================================================
                    st.header("Evaluasi Model")
    
                    y_pred_PSOSL = GRU_PSOSL.predict(X_test)
    
                    y_pred_PSOSL = scaler_y.inverse_transform(
                        y_pred_PSOSL.astype(np.float64)
                    ).flatten().astype(np.float64)
    
                    y_test_PSOSL = scaler_y.inverse_transform(
                        y_test.reshape(-1, 1).astype(np.float64)
                    ).flatten().astype(np.float64)
    
                    rmse_PSOSL = np.sqrt(mean_squared_error(y_test_PSOSL, y_pred_PSOSL))
                    mae_PSOSL  = mean_absolute_error(y_test_PSOSL, y_pred_PSOSL)
                    mape_PSOSL = mean_absolute_percentage_error(y_test_PSOSL, y_pred_PSOSL) * 100
    
                    train_loss_PSOSL = history_final.history['loss'][-1]
                    val_loss_PSOSL   = history_final.history['val_loss'][-1]
                    epoch_PSOSL      = len(history_final.history['loss'])
    
                    result_entry_PSOSL = {
                        'Time_Step':   window,
                        'Units':       best_units_PSOSL,
                        'Layers':      1,
                        'LR':          round(best_lr_PSOSL, 6),
                        'Train_Loss':  round(train_loss_PSOSL, 8),
                        'Val_Loss':    round(val_loss_PSOSL, 8),
                        'RMSE_Rp':     round(rmse_PSOSL, 2),
                        'MAE_Rp':      round(mae_PSOSL, 2),
                        'MAPE_%':      round(mape_PSOSL, 4),
                        'Epoch_Final': best_epochs_PSOSL
                    }
    
                    df_PSOSL_results = pd.DataFrame([result_entry_PSOSL])
                    st.dataframe(df_PSOSL_results)
    
                    os.makedirs("results", exist_ok=True)
                    result_path = "results/hasil_gru_pso.csv"
                    df_PSOSL_results.to_csv(result_path, index=False)
    
                    # =====================================================
                    # ACTUAL VS PREDICTED
                    # =====================================================
                    st.header("Actual vs Predicted")
    
                    fig_pred = plt.figure(figsize=(14, 7))
                    plt.plot(y_test_PSOSL,  label='Harga Aktual (Emas)',          color='royalblue', linewidth=2)
                    plt.plot(y_pred_PSOSL,  label='Harga Prediksi Model GRU-PSO', color='green',     linewidth=2)
                    plt.title('Model GRU-PSO: Aktual vs Prediksi Harga Emas Indonesia (IDR/Gram)', fontsize=14)
                    plt.xlabel('Indeks Waktu (Data Testing)', fontsize=12)
                    plt.ylabel('Harga Emas (Rp)',             fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.2)
                    st.pyplot(fig_pred)
    
                    # =====================================================
                    # SIMPAN KE SESSION STATE
                    # =====================================================
                    st.session_state['result_pso'] = {
                        'y_test':   y_test_PSOSL,
                        'y_pred':   y_pred_PSOSL,
                        'mape':     mape_PSOSL,
                        'model':    GRU_PSOSL,
                        'scaler_y': scaler_y,
                        'Xs':       Xs,
                        'window':   window,
                        'emas':     emas
                    }
    
                    # =====================================================
                    # DOWNLOAD BUTTON
                    # =====================================================
                    with open(result_path, 'rb') as file:
                        st.download_button(
                            label='Download Hasil CSV',
                            data=file,
                            file_name='hasil_gru_pso.csv',
                            mime='text/csv'
                        )
    
                    with open(model_path, 'rb') as file:
                        st.download_button(
                            label='Download Model H5',
                            data=file,
                            file_name='best_model_gru_pso.h5',
                            mime='application/octet-stream'
                        )

    # ===========================================================
    # TAB 2 — GRU STANDAR
    # ===========================================================
    with tab2:
        with st.container(border=True):

            st.subheader("GRU Standar")
    
            st.markdown("**Parameter Model (Fixed)**")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Units",   GS_UNITS)
            col2.metric("Layers",  GS_LAYERS)
            col3.metric("Dropout", GS_DROPOUT)
            col4.metric("LR",      GS_LR)
            col5.metric("Batch",   GS_BATCH)
            col6.metric("Epoch",   GS_EPOCH)
    
            if st.button("Train GRU Standar"):
    
                with st.spinner("Training GRU Standar..."):
    
                    # =====================================================
                    # PRAPEMROSESAN KHUSUS GRU STANDAR
                    # =====================================================
                    data_features_gs = emas[["Terakhir"]].values.astype(np.float64)
                    data_target_gs   = emas[["Terakhir"]].values.astype(np.float64)
    
                    n_gs       = len(data_features_gs)
                    n_train_gs = int(n_gs * 0.8)
    
                    scaler_X_gs = MinMaxScaler().fit(data_features_gs[:n_train_gs])
                    scaler_y_gs = MinMaxScaler().fit(data_target_gs[:n_train_gs])
    
                    Xs_gs = scaler_X_gs.transform(data_features_gs).astype(np.float64)
                    ys_gs = scaler_y_gs.transform(data_target_gs).astype(np.float64)
    
                    def make_sequences_gs(X_scaled, y_scaled, w):
                        X_seq, y_seq = [], []
                        for i in range(w, len(X_scaled)):
                            X_seq.append(X_scaled[i-w:i])
                            y_seq.append(y_scaled[i])
                        return np.array(X_seq, dtype=np.float64), np.array(y_seq, dtype=np.float64)
    
                    X_seq_gs, y_seq_gs = make_sequences_gs(Xs_gs, ys_gs, GS_WINDOW)
    
                    dtrain_end_gs = n_train_gs - GS_WINDOW
    
                    X_train_gs = X_seq_gs[:dtrain_end_gs].reshape(-1, GS_WINDOW, 1).astype(np.float64)
                    y_train_gs = y_seq_gs[:dtrain_end_gs]
                    X_test_gs  = X_seq_gs[dtrain_end_gs:].reshape(-1, GS_WINDOW, 1).astype(np.float64)
                    y_test_gs  = y_seq_gs[dtrain_end_gs:]
    
                    # =====================================================
                    # A. TRAINING
                    # =====================================================
                    clear_session()
                    reset_seeds()
    
                    gru_standar = build_gru_model(
                        GS_UNITS, GS_LAYERS, GS_DROPOUT, GS_LR, GS_WINDOW
                    )
    
                    early_stop_gs = EarlyStopping(
                        monitor='val_loss',
                        patience=7,
                        restore_best_weights=True
                    )
    
                    history_gs = gru_standar.fit(
                        X_train_gs, y_train_gs,
                        epochs=GS_EPOCH,
                        batch_size=GS_BATCH,
                        callbacks=[early_stop_gs],
                        validation_split=0.2,
                        verbose=0
                    )
    
                    epoch_stopped_gs = len(history_gs.history['loss'])
    
                    if os.path.exists(GS_MODEL_FILE):
                        gru_standar.load_weights(GS_MODEL_FILE)
    
                    # =====================================================
                    # GRAFIK LOSS GRU STANDAR
                    # =====================================================
                    st.header("Grafik Loss GRU Standar")
    
                    fig_loss_gs = plt.figure(figsize=(10, 5))
                    plt.plot(history_gs.history['loss'],     label='Training Loss')
                    plt.plot(history_gs.history['val_loss'], label='Validation Loss')
                    plt.title('Grafik Loss Model GRU Standar')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss (MSE)')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(fig_loss_gs)
    
                    # =====================================================
                    # EVALUASI GRU STANDAR
                    # =====================================================
                    st.header("Evaluasi Model GRU Standar")
    
                    y_pred_gs_scaled = gru_standar.predict(X_test_gs, verbose=0)
    
                    y_pred_gs = scaler_y_gs.inverse_transform(
                        y_pred_gs_scaled.astype(np.float64)
                    ).flatten().astype(np.float64)
    
                    y_test_gs_inv = scaler_y_gs.inverse_transform(
                        y_test_gs.reshape(-1, 1).astype(np.float64)
                    ).flatten().astype(np.float64)
    
                    rmse_gs = np.sqrt(mean_squared_error(y_test_gs_inv, y_pred_gs))
                    mae_gs  = mean_absolute_error(y_test_gs_inv, y_pred_gs)
                    mape_gs = mean_absolute_percentage_error(y_test_gs_inv, y_pred_gs) * 100
    
                    train_loss_gs = history_gs.history['loss'][-1]
                    val_loss_gs   = history_gs.history['val_loss'][-1]
    
                    result_gs = {
                        'Time_Step':   GS_WINDOW,
                        'Units':       GS_UNITS,
                        'Layers':      GS_LAYERS,
                        'LR':          GS_LR,
                        'Dropout':     GS_DROPOUT,
                        'Batch':       GS_BATCH,
                        'Train_Loss':  round(train_loss_gs, 8),
                        'Val_Loss':    round(val_loss_gs, 8),
                        'RMSE_Rp':     round(rmse_gs, 2),
                        'MAE_Rp':      round(mae_gs, 2),
                        'MAPE_%':      round(mape_gs, 4),
                        'Epoch_Final': epoch_stopped_gs
                    }
    
                    df_gs_results = pd.DataFrame([result_gs])
                    st.dataframe(df_gs_results)
    
                    # =====================================================
                    # ACTUAL VS PREDICTED GRU STANDAR
                    # =====================================================
                    st.header("Actual vs Predicted — GRU Standar")
    
                    fig_pred_gs = plt.figure(figsize=(14, 7))
                    plt.plot(y_test_gs_inv, label='Harga Aktual (Emas)',              color='royalblue', linewidth=2)
                    plt.plot(y_pred_gs,     label='Harga Prediksi Model GRU Standar', color='orange',    linewidth=2)
                    plt.title('Model GRU Standar: Aktual vs Prediksi Harga Emas Indonesia (IDR/Gram)', fontsize=14)
                    plt.xlabel('Indeks Waktu (Data Testing)', fontsize=12)
                    plt.ylabel('Harga Emas (Rp)',             fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.2)
                    st.pyplot(fig_pred_gs)
    
                    # =====================================================
                    # SIMPAN KE SESSION STATE
                    # =====================================================
                    st.session_state['result_gs'] = {
                        'y_test':   y_test_gs_inv,
                        'y_pred':   y_pred_gs,
                        'mape':     mape_gs,
                        'model':    gru_standar,
                        'scaler_y': scaler_y_gs,
                        'Xs':       Xs_gs,
                        'window':   GS_WINDOW,
                        'emas':     emas
                    }
    
                    # =====================================================
                    # DOWNLOAD GRU STANDAR
                    # =====================================================
                    os.makedirs("results", exist_ok=True)
                    result_path_gs = "results/hasil_gru_standar.csv"
                    df_gs_results.to_csv(result_path_gs, index=False)
    
                    with open(result_path_gs, 'rb') as file:
                        st.download_button(
                            label='Download Hasil CSV GRU Standar',
                            data=file,
                            file_name='hasil_gru_standar.csv',
                            mime='text/csv'
                        )

    # ===========================================================
    # TAB 3 — PERBANDINGAN MODEL
    # ===========================================================
    with tab3:
        with st.container(border=True):

            st.header("Perbandingan Model GRU-PSO vs GRU Standar")
    
            res_pso = st.session_state.get('result_pso')
            res_gs  = st.session_state.get('result_gs')
    
            if res_pso is None and res_gs is None:
                st.info("Belum ada model yang selesai dilatih. Silakan latih minimal satu model terlebih dahulu.")
            else:
                # =====================================================
                # GRAFIK PERBANDINGAN KURVA
                # =====================================================
                st.subheader("Kurva Aktual vs Prediksi (Semua Model)")
    
                # Tentukan acuan panjang y_test dari model yang tersedia
                if res_pso is not None and res_gs is not None:
                    min_len = min(len(res_pso['y_test']), len(res_gs['y_test']))
                elif res_pso is not None:
                    min_len = len(res_pso['y_test'])
                else:
                    min_len = len(res_gs['y_test'])
    
                fig_cmp, ax = plt.subplots(figsize=(14, 7))
    
                # Kurva aktual — ambil dari model mana saja yang tersedia
                if res_pso is not None:
                    ax.plot(
                        res_pso['y_test'][:min_len],
                        label='Harga Aktual',
                        color='royalblue',
                        linewidth=2.5
                    )
                else:
                    ax.plot(
                        res_gs['y_test'][:min_len],
                        label='Harga Aktual',
                        color='royalblue',
                        linewidth=2.5
                    )
    
                if res_pso is not None:
                    ax.plot(
                        res_pso['y_pred'][:min_len],
                        label='Prediksi GRU-PSO',
                        color='green',
                        linewidth=2,
                        linestyle='--'
                    )
    
                if res_gs is not None:
                    ax.plot(
                        res_gs['y_pred'][:min_len],
                        label='Prediksi GRU Standar',
                        color='orange',
                        linewidth=2,
                        linestyle='-.'
                    )
    
                ax.set_title('Perbandingan Aktual vs Prediksi: GRU-PSO & GRU Standar', fontsize=14)
                ax.set_xlabel('Indeks Waktu (Data Testing)', fontsize=12)
                ax.set_ylabel('Harga Emas (Rp)', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.2)
                st.pyplot(fig_cmp)
    
                # =====================================================
                # TABEL PERBANDINGAN METRIK
                # =====================================================
                st.subheader("Tabel Perbandingan Metrik Evaluasi")
    
                rows = []
                if res_pso is not None:
                    rmse_p = np.sqrt(mean_squared_error(res_pso['y_test'], res_pso['y_pred']))
                    mae_p  = mean_absolute_error(res_pso['y_test'], res_pso['y_pred'])
                    rows.append({
                        'Model':   'GRU-PSO',
                        'RMSE':    round(rmse_p, 2),
                        'MAE':     round(mae_p, 2),
                        'MAPE (%)': round(res_pso['mape'], 4)
                    })
                if res_gs is not None:
                    rmse_g = np.sqrt(mean_squared_error(res_gs['y_test'], res_gs['y_pred']))
                    mae_g  = mean_absolute_error(res_gs['y_test'], res_gs['y_pred'])
                    rows.append({
                        'Model':   'GRU Standar',
                        'RMSE':    round(rmse_g, 2),
                        'MAE':     round(mae_g, 2),
                        'MAPE (%)': round(res_gs['mape'], 4)
                    })
    
                df_cmp = pd.DataFrame(rows)
                st.dataframe(df_cmp, use_container_width=True)
    
                if len(rows) == 2:
                    best_model_name = df_cmp.loc[df_cmp['MAPE (%)'].idxmin(), 'Model']
                    st.success(f"Model terbaik berdasarkan MAPE terkecil: **{best_model_name}**")

    # ===========================================================
    # TAB 4 — PREDIKSI
    # ===========================================================
    with tab4:
        with st.container(border=True):

            st.header("Prediksi 5 Periode ke Depan")
    
            res_pso = st.session_state.get('result_pso')
            res_gs  = st.session_state.get('result_gs')
    
            if res_pso is None and res_gs is None:
                st.info("Belum ada model yang selesai dilatih. Silakan latih minimal satu model terlebih dahulu.")
            else:
                # =====================================================
                # PILIH MODEL TERBAIK BERDASARKAN MAPE
                # =====================================================
                if res_pso is not None and res_gs is not None:
                    if res_pso['mape'] <= res_gs['mape']:
                        best_res        = res_pso
                        best_model_name = "GRU-PSO"
                    else:
                        best_res        = res_gs
                        best_model_name = "GRU Standar"
                elif res_pso is not None:
                    best_res        = res_pso
                    best_model_name = "GRU-PSO"
                else:
                    best_res        = res_gs
                    best_model_name = "GRU Standar"
    
                st.info(f"Model yang digunakan untuk prediksi: **{best_model_name}** (MAPE: {round(best_res['mape'], 4)}%)")
    
                n_future = st.number_input(
                    "Jumlah Periode ke Depan",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1
                )
    
                if st.button("Prediksi ke Depan"):
    
                    model_fwd    = best_res['model']
                    scaler_y_fwd = best_res['scaler_y']
                    Xs_fwd       = best_res['Xs']
                    win_fwd      = best_res['window']
                    emas_fwd     = best_res['emas']
    
                    # Ambil window terakhir dari data scaled sebagai seed prediksi
                    last_window = Xs_fwd[-win_fwd:].reshape(1, win_fwd, 1).astype(np.float64)
    
                    future_scaled = []
                    current_input = last_window.copy()
    
                    for _ in range(n_future):
                        pred_scaled = model_fwd.predict(current_input, verbose=0)
                        future_scaled.append(float(pred_scaled[0, 0]))
                        # Geser window: buang elemen pertama, tambahkan prediksi baru
                        new_val      = pred_scaled[0, 0].reshape(1, 1, 1).astype(np.float64)
                        current_input = np.concatenate(
                            [current_input[:, 1:, :], new_val],
                            axis=1
                        )
    
                    future_preds = scaler_y_fwd.inverse_transform(
                        np.array(future_scaled, dtype=np.float64).reshape(-1, 1)
                    ).flatten()
    
                    # =====================================================
                    # KURVA HISTORIS + PREDIKSI DISAMBUNG
                    # =====================================================
                    hist_values = emas_fwd['Terakhir'].values.astype(np.float64)
                    n_hist      = len(hist_values)
    
                    # Index untuk x-axis
                    x_hist   = np.arange(n_hist)
                    x_future = np.arange(n_hist - 1, n_hist - 1 + n_future + 1)
                    # Sambungkan: titik terakhir historis + prediksi
                    y_future_plot = np.concatenate([[hist_values[-1]], future_preds])
    
                    fig_fwd, ax_fwd = plt.subplots(figsize=(14, 7))
    
                    ax_fwd.plot(
                        x_hist,
                        hist_values,
                        label='Data Historis Harga Emas',
                        color='royalblue',
                        linewidth=2
                    )
                    ax_fwd.plot(
                        x_future,
                        y_future_plot,
                        label=f'Prediksi {n_future} Periode ke Depan ({best_model_name})',
                        color='crimson',
                        linewidth=2.5,
                        linestyle='--',
                        marker='o',
                        markersize=6
                    )
                    # Garis vertikal pemisah historis–prediksi
                    ax_fwd.axvline(
                        x=n_hist - 1,
                        color='gray',
                        linestyle=':',
                        linewidth=1.5,
                        label='Batas Data Historis'
                    )
    
                    ax_fwd.set_title(
                        f'Harga Emas: Historis & Prediksi {n_future} Periode ke Depan\n(Model: {best_model_name})',
                        fontsize=14
                    )
                    ax_fwd.set_xlabel('Indeks Waktu', fontsize=12)
                    ax_fwd.set_ylabel('Harga Emas (Rp)', fontsize=12)
                    ax_fwd.legend()
                    ax_fwd.grid(True, alpha=0.2)
                    st.pyplot(fig_fwd)
    
                    # =====================================================
                    # TABEL HASIL PREDIKSI
                    # =====================================================
                    st.subheader(f"Tabel Prediksi {n_future} Periode ke Depan")
    
                    df_future = pd.DataFrame({
                        'Periode ke-': np.arange(1, n_future + 1),
                        f'Prediksi Harga Emas (Rp) — {best_model_name}': [
                            f"{v:,.2f}" for v in future_preds
                        ]
                    })
                    st.dataframe(df_future, use_container_width=True)
    
                    # =====================================================
                    # DOWNLOAD PREDIKSI
                    # =====================================================
                    os.makedirs("results", exist_ok=True)
                    path_future = "results/prediksi_ke_depan.csv"
                    df_future.to_csv(path_future, index=False)
    
                    with open(path_future, 'rb') as file:
                        st.download_button(
                            label='Download Prediksi CSV',
                            data=file,
                            file_name='prediksi_ke_depan.csv',
                            mime='text/csv'
                        )

else:
    st.info("Silakan upload dataset terlebih dahulu.")
