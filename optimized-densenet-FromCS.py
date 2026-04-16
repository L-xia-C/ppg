import re, gc, os, math, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PIL import Image
from datetime import datetime
from tqdm import tqdm


# Disable eager execution for better performance
# tf.compat.v1.disable_eager_execution()

def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covariance = np.sum((x - mean_x) * (y - mean_y)) / len(x)
    std_x = np.sqrt(np.sum((x - mean_x) ** 2) / len(x))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2) / len(y))
    pcc = covariance / (std_x * std_y)
    return pcc

def get_Model(i):
    script_path = os.path.dirname(os.path.abspath(__file__))
    weight_path=os.path.join(script_path,"weight/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5")
    """Create model architecture"""
    if i == 1:
        base_model = keras.applications.VGG16(
            weights='D:/LX/hr_estimation_2d/Python Script/new data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False,
            input_shape=(224, 224, 3))
    if i == 2:
        base_model = tf.keras.applications.DenseNet121(
            weights= weight_path,
            include_top=False,
            input_shape=(224, 224, 3)
        )
    base_model.trainable = True
    v_alpha = 0.1
    v_dropout = 0.5
    model = Sequential()
    model.add(base_model)
    model.add(keras.layers.BatchNormalization())

    if i == 0:
        model.add(keras.layers.AveragePooling2D(pool_size=(3, 3)))
    else:
        model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Flatten())
    s_layer_more = 2

    if s_layer_more == 1:
        model.add(layers.Dense(4096))
        model.add(keras.layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=v_alpha))
        model.add(layers.Dropout(v_dropout))

        model.add(layers.Dense(4096))
        model.add(keras.layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=v_alpha))
        model.add(layers.Dropout(v_dropout))

        model.add(layers.Dense(1000))
        model.add(keras.layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=v_alpha))
        model.add(layers.Dropout(v_dropout))
    elif s_layer_more == 2:
        if i == 0:
            model.add(keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=v_alpha)))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LeakyReLU(alpha=v_alpha))
            model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=v_alpha)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=v_alpha))
        model.add(keras.layers.Dropout(0.1))
    else:
        model.add(keras.layers.Dense(64))

    model.add(keras.layers.Dense(1, activation=keras.layers.LeakyReLU(alpha=v_alpha)))
    return model

def get_spllit_index_v2(s_list_file, i):
    """Split data for LOSO validation by subject"""
    all_index = list(range(len(s_list_file)))
    test_index = [s_list_file.index(item) for item in s_list_file if int(item.split("-")[0]) == i]
    train_index = [item for item in all_index if item not in test_index]
    return train_index, test_index

def extract_hr_from_filename(filename):
    """Extract heart rate value from filename"""
    match = re.search(r'\[([0-9\.]+)\]', filename)
    if match:
        return float(match.group(1))
    return None

def create_data_generators(file_list, hr_values, train_indices, test_indices, batch_size=16):
    """Create data generators for training and testing"""
    def preprocess_image(filename):
        """Preprocess a single image file"""
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0  # Normalize to [0,1]
        return img

    # Create lists for training
    train_files = [os.path.join(cwt_root, file_list[i]) for i in train_indices]
    train_labels = [hr_values[i] for i in train_indices]

    # Create lists for testing
    test_files = [os.path.join(cwt_root, file_list[i]) for i in test_indices]
    test_labels = [hr_values[i] for i in test_indices]

    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    num_parallel_calls = 8
    train_dataset = train_dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create testing dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_dataset = test_dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

def plot_training_history(history, fold_num, output_folder):
    """绘制训练历史损失曲线"""
    plt.figure(figsize=(10, 6))

    # 绘制损失曲线
    plt.plot(np.log(history.history['loss']), label='train')
    if 'val_loss' in history.history:
        plt.plot(np.log(history.history['val_loss']), label='val')

    plt.title(f'loss - Subject {fold_num}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"training_loss_{fold_num}.png"))
    plt.close()

def evaluate_model(model, train_dataset, test_dataset, train_indices, test_indices,
                   file_list, raw_hr_values, scaler_yy, fold_num, output_folder):
    """评估模型并保存结果"""
    # 获取测试集预测结果
    y_pred_test = model.predict(test_dataset)
    y_pred_test_orig = scaler_yy.inverse_transform(y_pred_test)
    
    # 获取测试集真实值
    y_test = np.array([raw_hr_values[i] for i in test_indices]).reshape(-1, 1)
    
    # 收集训练集的标签和预测值
    y_train_scaled = []
    y_pred_train = []
    
    # 遍历训练数据集，保持批处理
    for x_batch, y_batch in train_dataset:
        # 收集批次标签
        y_train_scaled.append(y_batch.numpy())
        # 收集批次预测
        pred = model.predict(x_batch, verbose=0)
        y_pred_train.append(pred)
    
    # 合并所有批次
    y_train_scaled = np.concatenate(y_train_scaled).reshape(-1, 1)
    y_pred_train = np.concatenate(y_pred_train).reshape(-1, 1)
    
    # 反归一化获取原始尺度
    y_train = scaler_yy.inverse_transform(y_train_scaled)
    y_pred_train_orig = scaler_yy.inverse_transform(y_pred_train)

    # 计算测试集指标
    mse_v = mean_squared_error(y_pred_test_orig, y_test)
    rmse_v = math.sqrt(mse_v)
    mae_v = mean_absolute_error(y_pred_test_orig, y_test)
    me_v = np.mean(y_pred_test_orig - y_test)
    pcc_v = pearson_correlation(y_pred_test_orig.squeeze(), y_test.squeeze())

    # 计算训练集指标
    train_mae = mean_absolute_error(y_pred_train_orig, y_train)
    train_pcc = pearson_correlation(y_pred_train_orig.squeeze(), y_train.squeeze())

    # 获取片段名称
    seg_test = [file_list[i].split('[')[0] for i in test_indices]

    # 保存结果到CSV
    fold_data = {
        'seg': seg_test,
        'ecg': y_test.squeeze(),
        'hr': y_pred_test_orig.squeeze()
    }
    df_fold = pd.DataFrame(fold_data)
    df_fold.to_csv(os.path.join(output_folder, f"subject_{fold_num}.csv"), index=False)

    # 创建图表 - 训练集和测试集的散点图
    plt.figure(figsize=(12, 10))

    # 创建 2x1 子图布局
    plt.subplot(2, 1, 1)

    # 训练集散点图 - 按样本索引排序
    x_train = np.arange(len(y_train))
    # 排序以便更好地可视化
    ind_train = np.argsort(y_train.squeeze())

    plt.scatter(x_train, y_train[ind_train].squeeze(), color='red', alpha=0.7, label='label', s=0.1)
    plt.scatter(x_train, y_pred_train_orig[ind_train].squeeze(), color='blue', alpha=0.7, label='pre', s=0.1)

    plt.title(f'train- (MAE: {train_mae:.3f}')
    plt.xlabel('sample')
    plt.ylabel('hr')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 测试集散点图
    plt.subplot(2, 1, 2)

    x_test = np.arange(len(y_test))
    # 排序以便更好地可视化
    ind_test = np.argsort(y_test.squeeze())

    plt.scatter(x_test, y_test[ind_test].squeeze(), color='red', alpha=0.7, label='label', s=0.1)
    plt.scatter(x_test, y_pred_test_orig[ind_test].squeeze(), color='blue', alpha=0.7, label='pre', s=0.1)

    plt.title(f'val - Subject {fold_num} (MAE: {mae_v:.3f})')
    plt.xlabel('sample')
    plt.ylabel('hr')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"prediction_scatter_{fold_num}.png"))
    plt.close()

    # 记录指标
    with open(os.path.join(output_folder, "LOSO.txt"), "a+") as file:
        file.write(f"subject:{fold_num},mae:{mae_v:.3f},rmse:{rmse_v:.3f},corr:{pcc_v:.3f},"
                   f"train_mae:{train_mae:.3f},train_corr:{train_pcc:.3f},"
                   f"test_samples:{len(y_test)},train_samples:{len(y_train)}\n")

    return mae_v, rmse_v, me_v, pcc_v, df_fold

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-l', '--lr', type=float, default=0.0025)
    parser.add_argument('-c', '--choose', type=str, default="0")
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-ss', '--subject_start', type=int, default=1)
    parser.add_argument('-se', '--subject_end', type=int, default=8)
    parser.add_argument('-o','--output_dir', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    epochs = args.epochs
    choose_gpu = args.choose
    lr = args.lr
    batch_size = args.batch_size
    begin = args.subject_start
    end = args.subject_end
    output_index_best_test_folder=args.output_dir

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = choose_gpu

    # Enable memory growth to avoid allocating all memory at once
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Constants
    N_pixels = 224
    cwt_root = r"D:\XZX\PPG-more\self_cwt\picture\self_highlight"
    CNN = ['densenet', 'vgg16_model2']
    flag_CNN = CNN[0]

    # Create output directory
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    # output_index_best_test_folder = rf"/ai/xzx/PPG-more/ppg_dalia_cwt/checkpoints_{formatted_time}_epochs{epochs}_lr{lr}"
    os.makedirs(output_index_best_test_folder, exist_ok=True)

    # Get file list - only read filenames and labels, not images
    print("拿到所有数据的label")
    s_list_file = os.listdir(cwt_root)
    ecg = [extract_hr_from_filename(file) for file in tqdm(s_list_file)]
    ecg = np.array([hr for hr in ecg if hr is not None])

    # Normalize labels
    print("Normalizing labels...")
    scaler_yy = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ecg.reshape(-1, 1))
    yy = scaler_yy.transform(ecg.reshape(-1, 1)).squeeze()

    # Lists to store results
    list_mae_v = []
    list_rmse_v = []
    list_me_v = []
    list_pcc_v = []
    pd_list = []

    # For each subject (LOSO)
    for k_count in range(begin, end):
        print(f"\n{'=' * 50}")
        print(f"Processing Subject {k_count}")
        print(f"{'=' * 50}")

        # Get train/test indices
        begin_time = time.time()
        train_index, test_index = get_spllit_index_v2(s_list_file, k_count)
        print(f"Split data: {len(train_index)} training samples, {len(test_index)} test samples")
        print(f"Split time: {(time.time() - begin_time):.2f} seconds")

        # Create data generators
        print("Creating data generators...")
        begin_time = time.time()
        train_dataset, test_dataset = create_data_generators(
            s_list_file, yy, train_index, test_index, batch_size)
        print(f"Data generator creation time: {(time.time() - begin_time):.2f} seconds")

        # Create model
        print("Creating model...")
        model_flag = 2  # DenseNet
        model = get_Model(model_flag)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.Huber(delta=2.0),
            metrics=['mse', 'mae']
        )

        # Create callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            patience=50,
            verbose=1,
            factor=0.9,
            min_lr=0.00000001
        )

        checkpoint_save_path = os.path.join(output_index_best_test_folder, f"model_{k_count}.h5")
        cp_callback = ModelCheckpoint(
            filepath=checkpoint_save_path,
            monitor='val_mae',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        )

        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=50,
        #     restore_best_weights=True
        # )

        # Train model
        print("Training model...")
        begin_time = time.time()
        history = model.fit(
            train_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=[cp_callback, reduce_lr],
            validation_data=test_dataset
        )
        plot_training_history(history, k_count, output_index_best_test_folder)
        # Load best weights
        model.load_weights(checkpoint_save_path)

        # Evaluate model
        print("Evaluating model...")
        begin_time = time.time()
        mae_v, rmse_v, me_v, pcc_v, df_fold = evaluate_model(
            model, train_dataset, test_dataset, train_index, test_index,
            s_list_file, ecg, scaler_yy, k_count, output_index_best_test_folder)
        print(f"Evaluation time: {(time.time() - begin_time):.2f} seconds")

        # Print results
        print(f"MAE: {mae_v:.3f}")
        print(f"RMSE: {rmse_v:.3f}")
        print(f"ME: {me_v:.3f}")
        print(f"PCC: {pcc_v:.3f}")

        # Append results
        list_mae_v.append(mae_v)
        list_rmse_v.append(rmse_v)
        list_me_v.append(me_v)
        list_pcc_v.append(pcc_v)
        pd_list.append(df_fold)

        # Clear memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Combine all results
    print("\nCombining results...")
    df_all = pd.concat(pd_list)
    df_all.to_csv(os.path.join(output_index_best_test_folder, "all.csv"), index=False)

    # Calculate and print overall metrics
    mae_average_v = np.mean(list_mae_v)
    rmse_average_v = np.mean(list_rmse_v)
    me_average_v = np.mean(list_me_v)
    pcc_average_v = np.mean(list_pcc_v)

    mae_std_v = np.std(list_mae_v, ddof=1)
    rmse_std_v = np.std(list_rmse_v, ddof=1)
    me_std_v = np.std(list_me_v, ddof=1)
    pcc_std_v = np.std(list_pcc_v, ddof=1)

    print("_________________________________________________________________________")
    print("average mae_v   :  ", '(%.3f ± %.3f)' % (mae_average_v, mae_std_v))
    print("average rmse_v  :  ", '(%.3f ± %.3f)' % (rmse_average_v, rmse_std_v))
    print("average me_v    :  ", '(%.3f ± %.3f)' % (me_average_v, me_std_v))
    print("average pcc     :  ", '(%.3f ± %.3f)' % (pcc_average_v, pcc_std_v))
    print("_________________________________________________________________________")

    # Save overall results
    # with open(os.path.join(output_index_best_test_folder, "LOSO.txt"), "a+") as file:
    #     file.write(f"ALL,mae:{mae_average_v:.3f},rmse:{rmse_average_v:.3f},corr:{pcc_average_v:.3f}\n")
