""" Training module for Echo State Network """
# Third-partyモジュール
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# 自作モジュール
from datasets import *
from echo_state_network import ReservoirNetWork


# パラメータ群
TRAIN_DATA_RATIO    = 0.8            # 学習データに使用する割合
EPOCH_NUM           = 1              # Epoch数
LEAK_RATE           = 0.5            # 漏れ率
NUM_INPUT_NODES     = 1              # 入力層のサイズ
NUM_RESERVOIR_NODES = 128            # Reservoir層のサイズ
NUM_OUTPUT_NODES    = 1              # 出力層のサイズ
ITER_NUM            = 1000           # 繰り返し回数
SKIP_NUM            = 100            # 繰り返し処理をskipする回数
T                   = np.pi*16       # 時間
dt                  = np.pi*0.01     # 時間刻み
NUM_TIME_STEPS      = int(T/dt)
X_INIT              = 0.001           # Xの初期値
Y_INIT              = 0.001           # Yの初期値
LOGISTIC_ALPHA      = 3.9            # Logistic Map alpha
HENON_ALPHA         = 1.4            # Henon Map alpha
HENON_BETA          = 0.3            # Henon Map beta


def train(dataset_type:str):
    # Load dataset
    if dataset_type == 'logistic_map':
        dataset = LogisticMapDataset(iter_num=ITER_NUM,
                                     skip_num=SKIP_NUM,
                                     alpha=LOGISTIC_ALPHA,
                                     x_init=X_INIT
                                    )
    elif dataset_type == 'sin':
        dataset = InputGenerator(0, T, NUM_TIME_STEPS)
    elif dataset_type == 'henon_map':
        dataset = HenonMapDataset(iter_num=ITER_NUM,
                                  skip_num=SKIP_NUM,
                                  x_init=X_INIT,
                                  y_init=Y_INIT,
                                  alpha=HENON_ALPHA,
                                  beta=HENON_BETA
                                )
    else:
        print(f"不明なdataset_tyoe : {dataset_type}")
        exit()

    # データを学習用とtest用に分割
    if dataset_type == 'henon_map':
        (data, _), t = dataset.get_data()
    else:
        data, t = dataset.get_data()
    train_data_num = int(len(data) * TRAIN_DATA_RATIO)
    train_data = data[:train_data_num]
    test_data  = data[train_data_num:]
    teacher_data = np.concatenate([train_data[1:], test_data[:1]])

    # モデルを定義
    model = ReservoirNetWork(
        inputs=train_data,
        teacher=teacher_data,
        num_input_nodes=NUM_INPUT_NODES,
        num_reservoir_nodes=NUM_RESERVOIR_NODES,
        num_output_nodes=NUM_OUTPUT_NODES,
        leak_rate=LEAK_RATE)

    # 学習
    for epoch in tqdm(range(EPOCH_NUM)):
        model.train(split_num=train_data_num) # 訓練
    train_result = model.get_train_result() # 訓練の結果を取得

    # 推論
    num_predict = int(len(test_data))
    predict_result = model.predict(num_predict)

    ## plot
    plt.plot(t, data, label="input_data")
    plt.plot(t[:train_data_num], train_result, label="trained")
    plt.plot(t[train_data_num:], predict_result, label="predicted")
    if dataset_type == 'sin':
        plt.axvline(x=int(T*TRAIN_DATA_RATIO), label="end of train", color="red") # border of train and prediction
    else:
        plt.axvline(x=train_data_num, label="end of train", color="red") # border of train and prediction
    plt.legend()
    plt.title(f"Echo State Network {dataset_type} Prediction")
    plt.xlabel("time step")
    plt.ylabel("x")
    plt.savefig(f"images/{dataset_type}_lr{LEAK_RATE}_plot.png")
    plt.clf()
    plt.close()



if __name__=="__main__":
    # logistic_map, sin, henon_map
    a = []
    ai = Decimal(0.2)
    while ai <= 0.91:
        a.append(ai)
        ai += Decimal(0.01)
    # for i in a:
    # LEAK_RATE = float(i)
    train(dataset_type='henon_map')