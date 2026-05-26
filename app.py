import streamlit as st
import os
import random
import numpy as np
import pandas as pd

# ==========================================================================
# 1. SET SEED GLOBAL & ATUR DETERMINISME SISTEM (WAJIB DI PALING ATAS)
# ==========================================================================
SEED = 49
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Paksa pakai CPU agar sama dengan runtime CPU Colab

import tensorflow as tf
# Atur konfigurasi thread CPU agar tidak terjadi paralelisme acak di server
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Set seed global TensorFlow & Numpy
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Input, GRU, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.backend import clear_session
import gc
from pyswarms.single import GlobalBestPSO

# Fungsi pengendali reset memory & seed internal
def reset_seeds_internal(seed=SEED):
    clear_session()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    gc.collect()

# Layout Judul Aplikasi
st.set_page_config(page_title="Prediksi Emas GRU Hybrid", layout="wide")
st.title("Prediksi Harga Emas dengan Arsitektur GRU (Sinkronisasi Colab)")
st.write("Aplikasi komputasi Statistika untuk membandingkan model GRU Standar (Adam) dengan GRU-PSO secara deterministik.")

# Input File dari User
uploaded_file = st.file_uploader("Unggah File Data Emas (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Pra-pemrosesan Data (Sinkronisasi Gaya Colab)
    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)
        
    emas = emas[['Tanggal', 'Terakhir']]
    emas.dropna(inplace=True)

    col_tanggal = emas.columns[0]
    emas[col_tanggal] = pd.to_datetime(emas[col_tanggal], dayfirst=True)

    # Urutkan data dari yang terlama ke terbaru (Lama ke Baru)
    emas = emas.sort_values(by=col_tanggal).reset_index(drop=True)
    st.success("Data berhasil diunggah dan disinkronkan!")
    
    with st.expander("Lihat Preview Data Emas"):
        st.dataframe(emas.head(10), use_container_width=True)

    # Mempersiapkan deret waktu (Time Series Sequences) untuk training
    feature_cols = ["Terakhir"]
    target_col   = "Terakhir"
    data_features = emas[feature_cols].values
    data_target = emas[[target_col]].values

    n = len(data_features)
    n_train = int(n * 0.8)

    scaler_X = MinMaxScaler().fit(data_features[:n_train])
    scaler_y = MinMaxScaler().fit(data_target[:n_train])
    Xs = scaler_X.transform(data_features)
    ys = scaler_y.transform(data_target)

    window = 1
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
    X_test  = X_seq_all[dtrain_end:]
    y_test  = y_seq_all[dtrain_end:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # ==========================================================================
    # KODE MODEL 1: TRAINING GRU-ADAM STANDAR
    # ==========================================================================
    def jalankan_training_adam():
        reset_seeds_internal(SEED)
        
        GS_epoch = 50
        GS_batch = 32
        GS_units = 16
        GS_layers = 1
        GS_dropout = 0.0
        GS_LR = 0.001
        
        model = Sequential()
        model.add(Input(shape=(window, 1)))
        model.add(GRU(units=GS_units, activation='tanh', recurrent_activation='sigmoid', reset_after=True))
        model.add(Dropout(GS_dropout))
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=GS_LR), loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            epochs=GS_epoch,
            batch_size=GS_batch,
            callbacks=[early_stop],
            validation_split=0.2,
            verbose=0
        )

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae  = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        
        return GS_units, GS_LR, GS_batch, rmse, mae, mape, y_test_inv.tolist(), y_pred_inv.tolist()

    # ==========================================================================
    # KODE MODEL 2: OPTIMASI HYPERPARAMETER DENGAN GRU-PSO
    # ==========================================================================
    def jalankan_pemodelan_pso_gru():
        # Persiapan data internal untuk fitness function PSO
        val_PSOSL = 0.2
        n_tr_samples_PSOSL = X_train.shape[0]
        n_tr_val_PSOSL = int(n_tr_samples_PSOSL * (1 - val_PSOSL))
        X_tr_PSOSL = X_train[:n_tr_val_PSOSL]
        y_tr_PSOSL = y_train[:n_tr_val_PSOSL]
        X_val_PSOSL = X_train[n_tr_val_PSOSL:]
        y_val_PSOSL = y_train[n_tr_val_PSOSL:]

        def make_pso_obj(X_tr, y_tr, X_va, y_va, scaler_y_obj):
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
                            Input(shape=(X_tr.shape[1], X_tr.shape[2])),
                            GRU(units=units, activation='tanh', recurrent_activation='sigmoid', reset_after=True),
                            Dropout(dropout),
                            Dense(1)
                        ])
                        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                        model.fit(X_tr, y_tr, epochs=10, batch_size=batch, verbose=0)
            
                        yv_pred = model.predict(X_va, verbose=0)
                        yv_pred_orig_PSOSL = scaler_y_obj.inverse_transform(yv_pred).flatten()
                        yv_true_orig_PSOSL = scaler_y_obj.inverse_transform(y_va.reshape(-1, 1)).flatten()
                        costs[i] = mean_squared_error(yv_true_orig_PSOSL, yv_pred_orig_PSOSL)
                    except Exception as e:
                        costs[i] = 1e12
                    clear_session()
                    gc.collect()
                return costs
            return obj_fn
            
        pso_obj_PSOSL = make_pso_obj(X_tr_PSOSL, y_tr_PSOSL, X_val_PSOSL, y_val_PSOSL, scaler_y)

        PSOSL_particles = 28
        PSOSL_iters = 5
        PSOSL_options = {'c1': 2.0, 'c2': 2.0, 'w': 0.7}
        PSOSL_bounds = ([16, 0.0001, 16, 0.01], [128, 0.01, 128, 0.5])
        
        np.random.seed(SEED)
        optimizer = GlobalBestPSO(
            n_particles=PSOSL_particles, dimensions=4,
            options=PSOSL_options, bounds=PSOSL_bounds
        )
        
        n_particles, dims = optimizer.swarm.position.shape
        optimizer.swarm.pbest_pos_PSOSL = optimizer.swarm.position.copy()
        optimizer.swarm.pbest_cost_PSOSL = np.full(n_particles, np.inf)
        
        history_gbest_pos_PSOSL = []
        
        # UI Streamlit untuk memantau progress iterasi PSO secara real-time seperti print log Colab
        pso_progress_box = st.empty()
        pso_log_text = "### 🔄 Log Optimasi Partikel PSO:\n"

        for it in range(PSOSL_iters):
            costs_PSOSL = pso_obj_PSOSL(optimizer.swarm.position)
            
            mask_PSOSL = costs_PSOSL < optimizer.swarm.pbest_cost_PSOSL
            optimizer.swarm.pbest_cost_PSOSL[mask_PSOSL] = costs_PSOSL[mask_PSOSL]
            optimizer.swarm.pbest_pos_PSOSL[mask_PSOSL] = optimizer.swarm.position[mask_PSOSL].copy()
            
            best_PSOSL = np.argmin(optimizer.swarm.pbest_cost_PSOSL)
            optimizer.swarm.best_cost_PSOSL = optimizer.swarm.pbest_cost_PSOSL[best_PSOSL]
            optimizer.swarm.best_pos_PSOSL = optimizer.swarm.pbest_pos_PSOSL[best_PSOSL].copy()

            history_gbest_pos_PSOSL.append(optimizer.swarm.best_pos_PSOSL.copy())
            
            # Tampilkan informasi iterasi ke UI web
            pso_log_text += (
                f"**ITERATION {it+1}** ➔ Global Best Loss: `{optimizer.swarm.best_cost_PSOSL:.6f}` | "
                f"Best Params: units=`{int(np.round(optimizer.swarm.best_pos_PSOSL[0]))}`, "
                f"lr=`{optimizer.swarm.best_pos_PSOSL[1]:.6f}`, batch=`{int(np.round(optimizer.swarm.best_pos_PSOSL[2]))}`\n\n"
            )
            pso_progress_box.markdown(pso_log_text)
            
            # Sinkronisasi urutan pergeseran random velocity seperti di loop Colab
            np.random.seed(SEED + it) 
            r1 = np.random.rand(*optimizer.swarm.position.shape)
            r2 = np.random.rand(*optimizer.swarm.position.shape)
            optimizer.swarm.velocity = (
                PSOSL_options['w'] * optimizer.swarm.velocity
                + PSOSL_options['c1'] * r1 * (optimizer.swarm.pbest_pos_PSOSL - optimizer.swarm.position)
                + PSOSL_options['c2'] * r2 * (optimizer.swarm.best_pos_PSOSL - optimizer.swarm.position)
            )
            optimizer.swarm.position += optimizer.swarm.velocity
            lb, ub = np.array(PSOSL_bounds[0]), np.array(PSOSL_bounds[1])
            optimizer.swarm.position = np.clip(optimizer.swarm.position, lb, ub)

        best_pos_PSOSL = history_gbest_pos_PSOSL[-1]
        best_units_PSOSL = int(np.round(best_pos_PSOSL[0]))
        best_lr_PSOSL = float(best_pos_PSOSL[1])
        best_batch_PSOSL = int(np.round(best_pos_PSOSL[2]))
        best_dropout_PSOSL = float(best_pos_PSOSL[3])
        
        # Retraining Model Final GRU-PSO berdasarkan parameter gbest terakhir
        reset_seeds_internal(SEED)
        
        GRU_PSOSL = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            GRU(units=best_units_PSOSL, activation='tanh', recurrent_activation='sigmoid', reset_after=True),
            Dropout(best_dropout_PSOSL),
            Dense(1)
        ])
        GRU_PSOSL.compile(optimizer=Adam(learning_rate=best_lr_PSOSL), loss='mse')
        
        early_stop_pso = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        GRU_PSOSL.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=best_batch_PSOSL, 
            callbacks=[early_stop_pso],
            validation_split=0.2, 
            verbose=0
        )
        
        y_pred_PSOSL = GRU_PSOSL.predict(X_test, verbose=0)
        y_pred_orig_PSOSL = scaler_y.inverse_transform(y_pred_PSOSL).flatten()
        y_test_orig_PSOSL = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse_PSOSL = np.sqrt(mean_squared_error(y_test_orig_PSOSL, y_pred_orig_PSOSL))
        mae_PSOSL = mean_absolute_error(y_test_orig_PSOSL, y_pred_orig_PSOSL)
        mape_PSOSL = mean_absolute_percentage_error(y_test_orig_PSOSL, y_pred_orig_PSOSL) * 100
        
        return (
            best_units_PSOSL, best_lr_PSOSL, best_batch_PSOSL, best_dropout_PSOSL,
            rmse_PSOSL, mae_PSOSL, mape_PSOSL, y_test_orig_PSOSL.tolist(), y_pred_orig_PSOSL.tolist()
        )

    # ==========================================================================
    # INTERFACE WEB: TOMBOL EKSEKUSI MODEL
    # ==========================================================================
    st.write("---")
    left_col, right_col = st.columns(2)
    
    if 'adam_done' not in st.session_state:
        st.session_state.adam_done = False
    if 'pso_done' not in st.session_state:
        st.session_state.pso_done = False

    # --- TOMBOL KIRI: MODEL STANDAR ADAM ---
    with left_col:
        st.subheader("1. Model GRU - Adam")
        st.write("Menjalankan training baseline model.")
        if st.button("Mulai Proses Training Adam"):
            with st.spinner("Sedang memproses GRU-Adam..."):
                u_a, lr_a, b_a, rmse_a, mae_a, mape_a, y_true_a, y_pred_a = jalankan_training_adam()
                st.session_state.u_a = u_a
                st.session_state.lr_a = lr_a
                st.session_state.b_a = b_a
                st.session_state.rmse_a = rmse_a
                st.session_state.mae_a = mae_a
                st.session_state.mape_a = mape_a
                st.session_state.y_true_a = np.array(y_true_a)
                st.session_state.y_pred_a = np.array(y_pred_a)
                st.session_state.adam_done = True
            st.success("Model Adam Selesai Berjalan!")
            
        if st.session_state.adam_done:
            st.metric("Units", st.session_state.u_a)
            st.metric("Learning Rate", f"{st.session_state.lr_a:.4f}")
            st.metric("Batch Size", st.session_state.b_a)
            
            st.markdown("**Metrik Evaluasi Adam:**")
            st.dataframe(pd.DataFrame([{
                'RMSE (Rp)': round(st.session_state.rmse_a, 2),
                'MAE (Rp)': round(st.session_state.mae_a, 2),
                'MAPE (%)': round(st.session_state.mape_a, 4)
            }]), use_container_width=True)

    # --- TOMBOL KANAN: MODEL OPTIMASI PSO ---
    with right_col:
        st.subheader("2. Model GRU - PSO")
        st.write("Melakukan pencarian hyperparameter terbaik dengan PSO.")
        if st.button("Mulai Optimasi & Prediksi PSO"):
            with st.spinner("Sedang menghitung GRU-PSO... Mohon ditunggu!"):
                u_p, lr_p, b_p, dr_p, rmse_p, mae_p, mape_p, y_true_p, y_pred_p = jalankan_pemodelan_pso_gru()
                st.session_state.u_p = u_p
                st.session_state.lr_p = lr_p
                st.session_state.b_p = b_p
                st.session_state.dr_p = dr_p
                st.session_state.rmse_p = rmse_p
                st.session_state.mae_p = mae_p
                st.session_state.mape_p = mape_p
                st.session_state.y_true_p = np.array(y_true_p)
                st.session_state.y_pred_p = np.array(y_pred_p)
                st.session_state.pso_done = True
            st.success("Model PSO Selesai Berjalan!")
            
        if st.session_state.pso_done:
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Optimal Units", st.session_state.u_p)
            col_p2.metric("Optimal LR", f"{st.session_state.lr_p:.6f}")
            col_p1.metric("Optimal Batch", st.session_state.b_p)
            col_p2.metric("Optimal Dropout", f"{st.session_state.dr_p:.4f}")
            
            st.markdown("**Metrik Evaluasi PSO:**")
            st.dataframe(pd.DataFrame([{
                'RMSE (Rp)': round(st.session_state.rmse_p, 2),
                'MAE (Rp)': round(st.session_state.mae_p, 2),
                'MAPE (%)': round(st.session_state.mape_p, 4)
            }]), use_container_width=True)

    # ==========================================================================
    # VISUALISASI PERBANDINGAN AKHIR
    # ==========================================================================
    if st.session_state.adam_done or st.session_state.pso_done:
        st.write("---")
        st.subheader("📈 Grafik Visualisasi Hasil Perbandingan")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if st.session_state.adam_done:
            ax.plot(st.session_state.y_true_a, label='Harga Aktual', color='black', linewidth=2)
            ax.plot(st.session_state.y_pred_a, label='Prediksi GRU-Adam', color='darkorange', linestyle='--', linewidth=1.5)
        elif st.session_state.pso_done:
            ax.plot(st.session_state.y_true_p, label='Harga Aktual', color='black', linewidth=2)
            
        if st.session_state.pso_done:
            ax.plot(st.session_state.y_pred_p, label='Prediksi GRU-PSO', color='crimson', linestyle='-.', linewidth=1.5)
            
        ax.set_title("Perbandingan Performa Model Aktual vs Prediksi", fontsize=14)
        ax.set_xlabel("Indeks Data Testing")
        ax.set_ylabel("Harga Emas (Rp)")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
