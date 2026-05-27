import os
import gc
import random
import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================TF
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
# CONFIG PAGE
# =====================================================
st.set_page_config(
    page_title="GRU-PSO Gold Forecasting",
    layout="wide"
)

st.title("GRU-PSO Forecasting Harga Emas")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Konfigurasi Model")

window = st.sidebar.number_input(
    "Window Size",
    min_value=1,
    value=1,
    step=1
)

PSOSL_particles = st.sidebar.number_input(
    "Jumlah Partikel",
    min_value=1,
    value=18
)

PSOSL_iters = st.sidebar.number_input(
    "Jumlah Iterasi",
    min_value=1,
    value=5
)

c1 = st.sidebar.number_input(
    "c1",
    value=2.0
)

c2 = st.sidebar.number_input(
    "c2",
    value=2.0
)

w = st.sidebar.number_input(
    "w",
    value=0.7
)

PSOSL_options = {
    'c1': c1,
    'c2': c2,
    'w': w
}

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
    # MISSING VALUE
    # =====================================================
    st.header("Missing Value")

    missing_table = emas.isnull().sum().reset_index()
    missing_table.columns = ['Kolom', 'Jumlah Missing']

    st.dataframe(missing_table)

    fig_missing = plt.figure(figsize=(10, 5))
    msno.matrix(emas)
    plt.title('Peta Distribusi Missing Value', fontsize=20)
    st.pyplot(fig_missing)

    # =====================================================
    # OUTLIERS
    # =====================================================
    st.header("Outlier Detection")

    fig_box = plt.figure(figsize=(10, 5))
    sns.boxplot(x=emas['Terakhir'], color='gold')
    plt.title('Boxplot Harga Emas (XAU/IDR)')
    st.pyplot(fig_box)

    Q1 = emas['Terakhir'].quantile(0.25)
    Q3 = emas['Terakhir'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = emas[
        (emas['Terakhir'] < lower_bound) |
        (emas['Terakhir'] > upper_bound)
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Q1", f"{Q1:,.2f}")
    col2.metric("Q3", f"{Q3:,.2f}")
    col3.metric("Lower Bound", f"{lower_bound:,.2f}")
    col4.metric("Upper Bound", f"{upper_bound:,.2f}")

    st.write(f"Jumlah Outlier ditemukan: {len(outliers)}")

    st.dataframe(outliers)

    # =====================================================
    # SPLIT DATA
    # =====================================================
    st.header("Split Data")

    feature_cols = ["Terakhir"]
    target_col = "Terakhir"

    data_features = emas[feature_cols].values
    data_target = emas[[target_col]].values

    values = emas[['Terakhir']].values

    n = len(values)
    n_train = int(n * 0.8)

    train_values = values[:n_train]
    test_values = values[n_train:]

    col1, col2 = st.columns(2)

    col1.metric("Jumlah Data Train", n_train)
    col2.metric("Jumlah Data Test", n - n_train)

    # =====================================================
    # SCALING
    # =====================================================
    st.header("Data Scaling")

    scaler_X = MinMaxScaler().fit(data_features[:n_train])
    scaler_y = MinMaxScaler().fit(data_target[:n_train])

    Xs = scaler_X.transform(data_features)
    ys = scaler_y.transform(data_target)

    scaled_df = pd.DataFrame({
        'Scaled_X': Xs.flatten(),
        'Scaled_y': ys.flatten()
    })

    st.dataframe(scaled_df.head())

    # =====================================================
    # WINDOWING
    # =====================================================
    st.header("Windowing Data")

    def make_sequences(X_scaled, y_scaled, window):
        X_seq, y_seq = [], []

        for i in range(window, len(X_scaled)):
            X_seq.append(X_scaled[i-window:i])
            y_seq.append(y_scaled[i])

        return np.array(X_seq), np.array(y_seq)

    X_seq_all, y_seq_all = make_sequences(Xs, ys, window=window)

    dtrain_end = n_train - window

    X_train = X_seq_all[:dtrain_end]
    y_train = y_seq_all[:dtrain_end]

    X_test = X_seq_all[dtrain_end:]
    y_test = y_seq_all[dtrain_end:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    st.write(f"Shape X_train: {X_train.shape}")
    st.write(f"Shape X_test: {X_test.shape}")

    # =====================================================
    # TRAIN BUTTON
    # =====================================================
    if st.button("Train GRU-PSO"):

        with st.spinner("Training Model..."):

            # =====================================================
            # PSO CONFIG
            # =====================================================
            PSOSL_bounds = (
                [16, 0.0001, 16, 0.01],
                [128, 0.01, 128, 0.5]
            )

            val_PSOSL = 0.2

            n_tr_samples_PSOSL = X_train.shape[0]
            n_tr_val_PSOSL = int(n_tr_samples_PSOSL * (1 - val_PSOSL))

            X_tr_PSOSL = X_train[:n_tr_val_PSOSL]
            y_tr_PSOSL = y_train[:n_tr_val_PSOSL]

            X_val_PSOSL = X_train[n_tr_val_PSOSL:]
            y_val_PSOSL = y_train[n_tr_val_PSOSL:]

            st.write("PSO training shapes:")
            st.write(X_tr_PSOSL.shape, X_val_PSOSL.shape)

            # =====================================================
            # FITNESS FUNCTION
            # =====================================================
            def make_pso_obj(X_tr, y_tr, X_va, y_va, scaler_y):

                def obj_fn(particles):

                    n_particles = particles.shape[0]
                    costs = np.zeros(n_particles)

                    for i, p in enumerate(particles):

                        units = int(np.round(p[0]))
                        lr = float(p[1])
                        batch = int(np.round(p[2]))
                        dropout = float(p[3])
                        try:
                            random.seed(49)
                            tf.random.set_seed(49)
                            clear_session()
                        
                            model = Sequential([
                                Input(shape=(X_tr.shape[1], X_tr.shape[2])),
                                GRU(
                                    units=units,
                                    activation='tanh'
                                ),
                                Dropout(dropout),
                                Dense(1)
                            ])

                            model.compile(
                                optimizer=Adam(learning_rate=lr),
                                loss='mse'
                            )

                            model.fit(
                                X_tr,
                                y_tr,
                                epochs=10,
                                batch_size=batch,
                                verbose=0
                            )

                            yv_pred = model.predict(X_va, verbose=0)

                            yv_pred_orig = scaler_y.inverse_transform(yv_pred).flatten()

                            yv_true_orig = scaler_y.inverse_transform(
                                y_va.reshape(-1, 1)
                            ).flatten()

                            costs[i] = mean_squared_error(
                                yv_true_orig,
                                yv_pred_orig
                            )

                        except Exception as e:
                            st.write("PSO eval error:", e)
                            costs[i] = 1e12

                        clear_session()
                        gc.collect()

                    return costs

                return obj_fn

            pso_obj_PSOSL = make_pso_obj(
                X_tr_PSOSL,
                y_tr_PSOSL,
                X_val_PSOSL,
                y_val_PSOSL,
                scaler_y
            )

            optimizer = GlobalBestPSO(
                n_particles=PSOSL_particles,
                dimensions=4,
                options=PSOSL_options,
                bounds=PSOSL_bounds
            )

            n_particles, dims = optimizer.swarm.position.shape

            optimizer.swarm.pbest_pos_PSOSL = optimizer.swarm.position.copy()
            optimizer.swarm.pbest_cost_PSOSL = np.full(n_particles, np.inf)

            history_positions_PSOSL = []
            history_velocity_PSOSL = []
            history_costs_PSOSL = []
            history_gbest_cost_PSOSL = []
            history_gbest_pos_PSOSL = []

            progress_bar = st.progress(0)

            iteration_results = []

            # =====================================================
            # LOOP PSO
            # =====================================================
            np.random.seed(49)
            for it in range(PSOSL_iters):

                costs_PSOSL = pso_obj_PSOSL(optimizer.swarm.position)

                mask_PSOSL = costs_PSOSL < optimizer.swarm.pbest_cost_PSOSL

                optimizer.swarm.pbest_cost_PSOSL[mask_PSOSL] = costs_PSOSL[mask_PSOSL]

                optimizer.swarm.pbest_pos_PSOSL[mask_PSOSL] = (
                    optimizer.swarm.position[mask_PSOSL].copy()
                )

                best_PSOSL = np.argmin(optimizer.swarm.pbest_cost_PSOSL)

                optimizer.swarm.best_cost_PSOSL = (
                    optimizer.swarm.pbest_cost_PSOSL[best_PSOSL]
                )

                optimizer.swarm.best_pos_PSOSL = (
                    optimizer.swarm.pbest_pos_PSOSL[best_PSOSL].copy()
                )

                history_positions_PSOSL.append(
                    optimizer.swarm.position.copy()
                )

                history_velocity_PSOSL.append(
                    optimizer.swarm.velocity.copy()
                )

                history_costs_PSOSL.append(costs_PSOSL.copy())

                history_gbest_cost_PSOSL.append(
                    float(optimizer.swarm.best_cost_PSOSL)
                )

                history_gbest_pos_PSOSL.append(
                    optimizer.swarm.best_pos_PSOSL.copy()
                )

                current_result = {
                    'Iterasi': it + 1,
                    'Best Loss': optimizer.swarm.best_cost_PSOSL,
                    'Units': int(np.round(optimizer.swarm.best_pos_PSOSL[0])),
                    'Learning Rate': optimizer.swarm.best_pos_PSOSL[1],
                    'Batch Size': int(np.round(optimizer.swarm.best_pos_PSOSL[2])),
                    'Dropout': optimizer.swarm.best_pos_PSOSL[3]
                }

                iteration_results.append(current_result)

                r1 = np.random.rand(*optimizer.swarm.position.shape)
                r2 = np.random.rand(*optimizer.swarm.position.shape)

                optimizer.swarm.velocity = (
                    PSOSL_options['w'] * optimizer.swarm.velocity
                    + PSOSL_options['c1'] * r1 * (
                        optimizer.swarm.pbest_pos_PSOSL - optimizer.swarm.position
                    )
                    + PSOSL_options['c2'] * r2 * (
                        optimizer.swarm.best_pos_PSOSL - optimizer.swarm.position
                    )
                )

                optimizer.swarm.position += optimizer.swarm.velocity

                lb, ub = np.array(PSOSL_bounds[0]), np.array(PSOSL_bounds[1])

                optimizer.swarm.position = np.clip(
                    optimizer.swarm.position,
                    lb,
                    ub
                )

                progress_bar.progress((it + 1) / PSOSL_iters)

            st.subheader("Hasil Iterasi PSO")
            st.dataframe(pd.DataFrame(iteration_results))

            # =====================================================
            # BEST PARAMETER
            # =====================================================
            best_pos_PSOSL = history_gbest_pos_PSOSL[-1]
            best_cost_PSOSL = history_gbest_cost_PSOSL[-1]

            best_units_PSOSL = int(np.round(best_pos_PSOSL[0]))
            best_lr_PSOSL = float(best_pos_PSOSL[1])
            best_batch_PSOSL = int(np.round(best_pos_PSOSL[2]))
            best_dropout_PSOSL = float(best_pos_PSOSL[3])
            best_epochs_PSOSL = 50

            st.success("PSO Finished")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Units", best_units_PSOSL)
            col2.metric("Learning Rate", round(best_lr_PSOSL, 6))
            col3.metric("Batch Size", best_batch_PSOSL)
            col4.metric("Dropout", round(best_dropout_PSOSL, 4))

            # =====================================================
            # FINAL TRAINING
            # =====================================================
            GRU_PSOSL = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
            
                GRU(
                    units=best_units_PSOSL,
                    activation='tanh'
                ),
            
                Dropout(best_dropout_PSOSL),
            
                Dense(1)
            ])

            GRU_PSOSL.compile(
                optimizer=Adam(learning_rate=best_lr_PSOSL),
                loss='mse'
            )

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True
            )
            
            history_final = GRU_PSOSL.fit(
                X_train,
                y_train,
                epochs=50,
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

            gbest_loss_PSOSL = np.array(history_gbest_cost_PSOSL)

            iterations_PSOSL = np.arange(
                1,
                len(gbest_loss_PSOSL) + 1
            )

            fig_conv = plt.figure(figsize=(8, 5))

            plt.plot(
                iterations_PSOSL,
                gbest_loss_PSOSL,
                marker='o'
            )

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

            plt.plot(
                history_final.history['loss'],
                label='Training Loss'
            )

            plt.plot(
                history_final.history['val_loss'],
                label='Validation Loss'
            )

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
                y_pred_PSOSL
            ).flatten()

            y_test_PSOSL = scaler_y.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            rmse_PSOSL = np.sqrt(mean_squared_error(
                y_test_PSOSL,
                y_pred_PSOSL
            ))

            mae_PSOSL = mean_absolute_error(
                y_test_PSOSL,
                y_pred_PSOSL
            )

            mape_PSOSL = mean_absolute_percentage_error(
                y_test_PSOSL,
                y_pred_PSOSL
            ) * 100

            train_loss_PSOSL = history_final.history['loss'][-1]
            val_loss_PSOSL = history_final.history['val_loss'][-1]
            epoch_PSOSL = len(history_final.history['loss'])

            result_entry_PSOSL = {
                'Time_Step': 1,
                'Units': best_units_PSOSL,
                'Layers': 1,
                'LR': round(best_lr_PSOSL, 6),
                'Train_Loss': round(train_loss_PSOSL, 8),
                'Val_Loss': round(val_loss_PSOSL, 8),
                'RMSE_Rp': round(rmse_PSOSL, 2),
                'MAE_Rp': round(mae_PSOSL, 2),
                'MAPE_%': round(mape_PSOSL, 4),
                'Epoch_Final': best_epochs_PSOSL
            }

            df_PSOSL_results = pd.DataFrame([
                result_entry_PSOSL
            ])

            st.dataframe(df_PSOSL_results)

            os.makedirs("results", exist_ok=True)

            result_path = "results/hasil_gru_pso.csv"

            df_PSOSL_results.to_csv(
                result_path,
                index=False
            )

            # =====================================================
            # ACTUAL VS PREDICTED
            # =====================================================
            st.header("Actual vs Predicted")

            fig_pred = plt.figure(figsize=(14, 7))

            plt.plot(
                y_test_PSOSL,
                label='Harga Aktual (Emas)',
                color='royalblue',
                linewidth=2
            )

            plt.plot(
                y_pred_PSOSL,
                label='Harga Prediksi Model GRU-PSO',
                color='green',
                linewidth=2
            )

            plt.title(
                'Model GRU-PSO: Aktual vs Prediksi Harga Emas Indonesia (IDR/Gram)',
                fontsize=14
            )

            plt.xlabel(
                'Indeks Waktu (Data Testing)',
                fontsize=12
            )

            plt.ylabel(
                'Harga Emas (Rp)',
                fontsize=12
            )

            plt.legend()
            plt.grid(True, alpha=0.2)

            st.pyplot(fig_pred)

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

else:
    st.info("Silakan upload dataset terlebih dahulu.")
