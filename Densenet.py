import re,gc
import logging
import os, csv, scipy.stats, math, pycwt, pywt, cv2
import time

from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import Sequential
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage import transform
from PIL import Image
from datetime import datetime
from tqdm import tqdm
# from merge import merge_1D_yolo_DenseAfterYolo
import argparse
from sklearn.model_selection import KFold

def pearson_correlation(x, y):
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 计算协方差
    covariance = np.sum((x - mean_x) * (y - mean_y)) / len(x)

    # 计算标准差
    std_x = np.sqrt(np.sum((x - mean_x) ** 2) / len(x))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2) / len(y))

    # 计算Pearson相关系数
    pcc = covariance / (std_x * std_y)

    return pcc
def gen_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal = signal + noise
    return signal
def read_specturm(file_dir, N_pixels, s_image_flag):
    ecg   = []
    files = os.listdir(file_dir)  # 读入文件夹
    num_imag = len(files)  # 统计文件夹中的文件个数
    pbar=tqdm(total=num_imag,desc="read_specturm")
    if s_image_flag == "gray":
        out_image = np.ones((num_imag, N_pixels, N_pixels))
    elif s_image_flag == "rgb":
        out_image = np.ones((num_imag, N_pixels, N_pixels, 3))
    for i,file in enumerate(files):
        # print(files[i])
        s_name = os.path.join(file_dir, file)
        match = re.search(r'\[([0-9\.]+)\]', file)
        if match:
            number_str = match.group(1)  # 提取数字部分
            ecg.append(float(number_str))  # 转换为 float
            # print(ecg[i])
        else:
            print("没有找到数字")
        image = Image.open(s_name)  # 读取图片文件
        if s_image_flag == "gray":
            image = image.convert("1")
        # =============================================================================
        #plt.imshow(image)
        #plt.show()            #将图片输出到屏幕
        # =============================================================================
        image_arr0 = np.array(image)  # 将图片以数组的形式读入变量
        image_arr = transform.resize(image_arr0, (N_pixels, N_pixels))
        # print (image_arr)
        if s_image_flag == "gray":
            out_image[i, :, :] = image_arr
        elif s_image_flag == "rgb":
            out_image[i, :, :, :] = image_arr
        pbar.update(1)
    return files, out_image,ecg
def get_Model(i):
    if i == 1:
        base_model = keras.applications.VGG16(
            weights='D:/LX/hr_estimation_2d/Python Script/new data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False,
            input_shape=(224, 224, 3))
    if i == 2:
        base_model = tf.keras.applications.DenseNet121(
            weights='D:\XZX\mmWave-2D\densenet-wPE\model/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
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
        model.add(keras.layers.AveragePooling2D(pool_size=(3, 3)))  # pool_size=(3, 3)
    else:
        model.add(keras.layers.GlobalAveragePooling2D())  # pool_size=(3, 3)
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
    elif s_layer_more == 2:  #
        if i == 0:
            model.add(keras.layers.Dense(128, activation=keras.layers.LeakyReLU(
                alpha=v_alpha)))  # ,kernel_regularizer=l2(0.0001)
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LeakyReLU(alpha=v_alpha))
            model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=v_alpha)))  # ,kernel_regularizer=l2(0.0001)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=v_alpha))
        model.add(keras.layers.Dropout(0.1))
    else:
        model.add(keras.layers.Dense(64))


    model.add(keras.layers.Dense(1, activation=keras.layers.LeakyReLU(alpha=v_alpha)))
    return model
def get_spllit_index(s_list_file,i):
    all_index=list(range(len(s_list_file)))
    cwt_all_split=rf"D:\XZX\mmWave-2D\densenet-wPE\cwt_all_split\fold{i}"
    fold_list=os.listdir(cwt_all_split)
    test_index=[s_list_file.index(seg) for seg in fold_list]
    train_index=[item for item in all_index if item not in test_index]
    '''
    Parameters
    ----------
    i:第几折
    Returns
    -------
    对应的 trian_index / test_inex
    '''
    return train_index,test_index
def get_spllit_index_v2(s_list_file,i):
    all_index=list(range(len(s_list_file)))
    # cwt_all_split=rf"D:\XZX\mmWave-2D\densenet-wPE\cwt_all_split\fold{i}"
    # fold_list=os.listdir(cwt_all_split)
    test_index=[ s_list_file.index(item) for item in s_list_file if int(item.split("-")[0])==i]
    train_index=[item for item in all_index if item not in test_index]
    '''
    Parameters
    ----------
    i:第几折
    Returns
    -------
    对应的 trian_index / test_inex
    '''
    return train_index,test_index
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-l', '--lr', type=float, default=0.002)
    parser.add_argument('-c', '--choose', type=str, default="0")
    return parser.parse_args()

if __name__ == "__main__":
    path = os.getcwd()
    args=get_args()
    epochs=args.epochs
    choose_gpu=args.choose
    lr=args.lr
    os.environ["CUDA_VISIBLE_DEVICES"] = choose_gpu
    N_pixels = 224

    cwt_root=r"D:\XZX\PPG-more\self_cwt\picture\self_highlight"

    CNN = ['densenet', 'vgg16_model2']
    flag_CNN = CNN[0]

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    output_index_best_test_folder = rf"D:\XZX\PPG-more\ppg_dalia_cwt\checkpoints_{formatted_time}_epochs{epochs}_lr{lr}"
    os.makedirs(output_index_best_test_folder, exist_ok=True)

    ## 评估读图片用时
    beginTime=time.time()
    s_list_file, xx ,ecg = read_specturm(cwt_root, N_pixels, "rgb")
    print(f"read_specturm:{(time.time() - beginTime)/60}")

    yy=np.array(ecg)

    ## 评估归一化用时
    beginTime = time.time()
    scaler_yy = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(yy.reshape(-1, 1))
    yy = scaler_yy.transform(yy.reshape(-1, 1))
    print(f"scaler_transform:{(time.time() - beginTime)/60}")

    yy = yy.astype('float')
    yy = yy.reshape(-1,)

    list_mse = []
    list_mse_v = []
    list_rmse = []
    list_rmse_v = []
    list_mae = []
    list_mae_v = []
    list_me = []
    list_me_v = []
    list_pcc_v= []
    list_corr = []
    list_corr_v = []

    list_y = []
    list_y_pred = []

    pd_list=[]

    for k_count in range(1,16):

        beginTime=time.time()
        train_index,test_index=get_spllit_index_v2(s_list_file,k_count)
        seg_test = [s_list_file[i].split('[')[0] for i in test_index]
        print(f"get_index:{(time.time() - beginTime)/60}")

        x_train, y_train = np.array(xx[train_index, :, :]), np.array(yy[train_index])
        x_test, y_test = np.array(xx[test_index, :, :]), np.array(yy[test_index])
        print([ s_list_file[i] for i in test_index ])
        if flag_CNN == 'densenet':
            model_flag = 2
            model = get_Model(model_flag)  # 3
        else:
            model_flag = 1
            model = get_Model(model_flag)  # get_Model(1)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.Huber(delta=2.0, reduction="auto", name="huber_loss"),
                      metrics=['mse', 'mae'])
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  # 'val_mae',
                                                         patience=20,
                                                         verbose=1,
                                                         factor=0.9,
                                                         min_lr=0.00000001)



        checkpoint_save_path=os.path.join(output_index_best_test_folder,f"model_{k_count}.h5")
        cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                                      monitor='val_mae',  # 'val_loss', #
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='min')  # 分类的参数设置


        flag_fitModel = 1  # 默认=1:训练网络      =0:不训练网络(为了在后面加载训练过的模型，然后输出这个模型的精度)
        flag_loadModel = 1
        if flag_fitModel == 1:
            model_lstm = model.fit(x_train,
                                   y_train,
                                   batch_size=16,  # 32
                                   epochs=epochs,
                                   verbose=2,
                                   callbacks=[cp_callback, reduce_lr],
                                   validation_data=(x_test, y_test),
                                   validation_freq=1,
                                   shuffle=True)  #
        else:
            model_lstm = model.fit(x_train, y_train,
                                   batch_size=16,  # 32
                                   epochs=epochs,
                                   verbose=2,
                                   callbacks=[reduce_lr],
                                   validation_data=(x_test, y_test),
                                   validation_freq=1,
                                   shuffle=True)  #
        model.summary()

        # (4.1)checkpoint model
        if flag_loadModel == 1:
            new_model = get_Model(model_flag)  # 函数  源码搭建的函数
            folder = os.path.exists(checkpoint_save_path)
            new_model.load_weights(checkpoint_save_path)

            # Step3 visulization
            # (1)loss
            loss_v      = model_lstm.history['loss']
            val_loss_v  = model_lstm.history['val_loss']

            # (2)regression
            yy_train_v  = new_model.predict(x_train)
            yy_pred_v   = new_model.predict(x_test)
            y_train_v   = scaler_yy.inverse_transform(y_train.reshape(-1,1))
            y_test_v    = scaler_yy.inverse_transform(y_test.reshape(-1,1))

            yy_pred_v   = scaler_yy.inverse_transform(yy_pred_v.reshape(-1,1))
            yy_train_v  = scaler_yy.inverse_transform(yy_train_v.reshape(-1,1))

            # (3)figure plot
            ind_plot_train_v  = np.argsort(y_train_v.reshape(-1,))
            ind_plot_test_v   = np.argsort(y_test_v.reshape(-1,))

    #=============================================================================
            plt.figure()
            plt.subplot(311)
            plt.plot(scipy.log(loss_v),color='blue', label='log10, Training Loss')
            plt.plot(scipy.log(val_loss_v),color='red', label='log10, Validation Loss')
            plt.title('Training and Validation Loss')

            plt.subplot(312)
            plt.plot(y_train_v[ind_plot_train_v],  color='red', label='real_HR')
            plt.plot(yy_train_v[ind_plot_train_v], color='blue',label='Trained HR')
            plt.title('HR Trained')
            plt.xlabel('sample')
            plt.ylabel('real HR')
            plt.legend()

            plt.subplot(313)
            plt.plot(y_test_v[ind_plot_test_v], color='red', label='real_HR')
            plt.plot(yy_pred_v[ind_plot_test_v],color='blue',label='Predicted HR')
            plt.title('HR Prediction')
            plt.xlabel('sample')
            plt.ylabel('real HR')
            plt.legend()
            plt.savefig(os.path.join(output_index_best_test_folder,f"{k_count}.png"))
            plt.close()
    #=============================================================================

            #(4)results
            mse_v  = mean_squared_error(yy_pred_v, y_test_v)            # calculate MSE 均方误差 ----->E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
            rmse_v = math.sqrt(mean_squared_error(yy_pred_v, y_test_v)) # calculate RMSE 均方根误差--->sqrt[MSE]           (对均方误差开方)
            mae_v  = mean_absolute_error(yy_pred_v, y_test_v)           # calculate MAE 平均绝对误差-->E[|预测值-真实值|]    (预测值减真实值求绝对值后求均值）
            me_v   = sum(yy_pred_v-y_test_v)/len(yy_pred_v)
            PCC_v  = pearson_correlation(yy_pred_v, y_test_v)
            y_test_v = y_test_v.squeeze()
            yy_pred_v = yy_pred_v.squeeze()
            fold_data = {
                'seg': seg_test,
                'ecg': y_test_v,
                'hr': yy_pred_v
            }
            df_fold = pd.DataFrame(fold_data)
            df_fold.to_csv(os.path.join(output_index_best_test_folder,f"subject_{k_count}.csv"), index=False)
            pd_list.append(df_fold)

            print('mse_v : %.3f' % mse_v)
            print('rmse_v: %.3f' % rmse_v)
            print('mae_v : %.3f' % mae_v)
            print('me_v  : %.3f' % me_v)
            print('PCC_v   : %.3f' % PCC_v)
            list_mse_v.append(mse_v)
            list_rmse_v.append(rmse_v)
            list_mae_v.append(mae_v)
            list_me_v.append(me_v)
            list_pcc_v.append(PCC_v)
            with open(os.path.join(output_index_best_test_folder,"LOSO.txt"), "a+") as file:
                file.write(f"subject:{k_count},mae:{mae_v:.3f},rmse:{rmse_v:.3f},corr:{PCC_v:.3f},length:{len(y_test_v)}\n")

            # 删除模型和清理会话
            del model
            del new_model
            del model_lstm
            tf.keras.backend.clear_session()
            gc.collect()

    df_all=pd.concat(pd_list)
    df_all.to_csv(os.path.join(output_index_best_test_folder,"all.csv"),index=False)

    if flag_loadModel == 1:
        mse_average_v = sum(list_mse_v) / len(list_mse_v)
        rmse_average_v = sum(list_rmse_v) / len(list_rmse_v)
        mae_average_v = sum(list_mae_v) / len(list_mae_v)
        me_average_v = sum(list_me_v) / len(list_me_v)
        pcc_average_v=sum(list_pcc_v)/len(list_pcc_v)

        mse_std_v = np.std(list_mse_v, ddof=1)  # 无偏标准差，/(n-1)
        rmse_std_v = np.std(list_rmse_v, ddof=1)  # 无偏标准差，/(n-1)
        mae_std_v = np.std(list_mae_v, ddof=1)  # 无偏标准差，/(n-1)
        me_std_v = np.std(list_me_v, ddof=1)  # 无偏标准差，/(n-1)
        pcc_std_v=np.std(list_pcc_v,ddof=1)

        print("_________________________________________________________________________")
        print("average mse_v   :  ", '(%.3f ± %.3f)' % (mse_average_v, mse_std_v))
        print("average rmse_v  :  ", '(%.3f ± %.3f)' % (rmse_average_v, rmse_std_v))
        print("average mae_v   :  ", '(%.3f ± %.3f)' % (mae_average_v, mae_std_v))
        print("average me_v   :  ", '(%.3f ± %.3f)' % (me_average_v, me_std_v))
        print("average pcc   :  ", '(%.3f ± %.3f)' % (pcc_average_v, pcc_std_v))
        print("_________________________________________________________________________")
with open(os.path.join(output_index_best_test_folder, "LOSO.txt"), "a+") as file:
    file.write(f"ALL,mae:{mae_average_v:.3f},rmse:{rmse_average_v:.3f},corr:{pcc_average_v:.3f}\n")

