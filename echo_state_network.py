""" Echo State Network """
import random

import numpy as np
from scipy import linalg



class ReservoirNetWork:
    def __init__(self, inputs:'numpy.ndarray',
                 teacher:'numpy.ndarray',
                 num_input_nodes:int,
                 num_reservoir_nodes:int,
                 num_output_nodes:int,
                 leak_rate:float=0.1,
                 activator:'numpy.ufunc'=np.tanh
                ):
        self.inputs = inputs        # 学習で使う入力
        self.teacher = teacher      # 教師データ
        self.log_reservoir_nodes = np.array([np.zeros(num_reservoir_nodes)]) # reservoir層のノードの状態を記録

        # init weights
        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)

        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        for i, layer in enumerate(self.weights_reservoir):
            for j, node in enumerate(layer):
                if abs(self.weights_reservoir[i][j]) < 0.00001 :
                    self.weights_reservoir[i][j] = 0
                if i == j:
                    self.weights_reservoir[i][j] = 0
        cnt = 0
        for i, layer in enumerate(self.weights_reservoir):
            for j, node in enumerate(layer):
                if node == 0:
                    cnt += 1
        print(f"(reservoir_weights)cnt = {cnt}")
        print(f"(reservoir_weights)0 rate = {cnt*100/(num_reservoir_nodes*num_reservoir_nodes)}[%]")
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])

        # それぞれの層のノードの数
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.num_output_nodes = num_output_nodes

        self.leak_rate = leak_rate # 漏れ率
        self.activator = activator # 活性化関数

    def _get_next_reservoir_nodes(self, input:'numpy.ndarray', current_state:'numpy.ndarray'):
        """reservoir層のノードの次の状態を取得
        """
        # next_state = (1 - self.leak_rate) * current_state
        # next_state += self.leak_rate * (np.array([input]) @ self.weights_input
        #     + current_state @ self.weights_reservoir)
        # return self.activator(next_state)
        next_state = (1 - self.leak_rate) * current_state
        U = np.array([input]) @ self.weights_input + current_state @ self.weights_reservoir
        next_state = next_state + self.leak_rate * self.activator(U)
        return next_state

    def _update_weights_output(self, lambda0:float):
        """出力層の重みを更新
        """
        # Ridge Regression
        # E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0 # lambda0
        # inv_x = np.linalg.inv(self.log_reservoir_nodes.T @ self.log_reservoir_nodes + E_lambda0)
        # # update weights of output layer
        # self.weights_output = (inv_x @ self.log_reservoir_nodes.T) @ self.inputs
        E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0 # lambda0
        inv_x = np.linalg.inv(self.log_reservoir_nodes.T @ self.log_reservoir_nodes + E_lambda0)
        # # update weights of output layer
        self.weights_output = (inv_x @ self.log_reservoir_nodes.T) @ self.teacher

    def train(self, split_num:int, lambda0:float=0.01):
        """学習"""
        for input in self.inputs:
            current_state = np.array(self.log_reservoir_nodes[-1])
            self.log_reservoir_nodes = np.append(self.log_reservoir_nodes,
                [self._get_next_reservoir_nodes(input, current_state)], axis=0)
        if self.log_reservoir_nodes.shape[0] > split_num+1:
            self.log_reservoir_nodes = self.log_reservoir_nodes[split_num-1:,:]
        self.log_reservoir_nodes = self.log_reservoir_nodes[1:]
        self._update_weights_output(lambda0)

    def get_train_result(self):
        """学習で得られた重みを基に訓練データを学習できているかを出力"""
        outputs = []
        reservoir_nodes = np.zeros(self.num_reservoir_nodes)
        for input in self.inputs:
            reservoir_nodes = self._get_next_reservoir_nodes(input, reservoir_nodes)
            outputs.append(self.get_output(reservoir_nodes))
        return outputs

    def predict(self, length_of_sequence:int):
        """予測する"""
        predicted_outputs = [self.inputs[-1]] # 最初にひとつ目だけ学習データの最後のデータを使う
        reservoir_nodes = self.log_reservoir_nodes[-1] # 訓練の結果得た最後の内部状態を取得
        for _ in range(length_of_sequence):
            reservoir_nodes = self._get_next_reservoir_nodes(predicted_outputs[-1], reservoir_nodes)
            predicted_outputs.append(self.get_output(reservoir_nodes))
        return predicted_outputs[1:] # 最初に使用した学習データの最後のデータを外して返す

    def get_output(self, reservoir_nodes:int):
        """get output of current state"""
        # return self.activator(reservoir_nodes @ self.weights_output) 修正前
        return reservoir_nodes @ self.weights_output # 修正後

    def _generate_variational_weights(self, num_pre_nodes:int, num_post_nodes:int):
        """重みを0.5(25%)か-0.5(25%)か0(50%)で初期化したものを返す"""
        # return (np.random.randint(0, 2, num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]) * 2 - 1) * 0.1
        weights = []
        total_node_num = num_pre_nodes * num_post_nodes
        cnt_5  = 0
        cnt_0  = 0
        cnt_m5 = 0
        c_list = [0.5, 0, -0.5]
        for i in range(num_pre_nodes):
            for j in range(num_post_nodes):
                if cnt_5 > int(0.25*total_node_num):
                    try:
                        c_list.remove(0.5)
                    except:
                        print()
                if cnt_0 > int(0.5*total_node_num):
                    try:
                        c_list.remove(0)
                    except:
                        print()
                if cnt_m5 > int(0.25*total_node_num):
                    try:
                        c_list.remove(-0.5)
                    except:
                        print()
                if c_list == []:
                    break

                val = random.choice(c_list)
                if val == 0.5:
                    cnt_5 += 1
                elif val == 0:
                    cnt_0 += 1
                else:
                    cnt_m5 += 1
                weights.append(val)
        return np.array([weights])

    def _generate_reservoir_weights(self, num_nodes:int):
        """Reservoir層の重みを初期化"""
        weights = np.random.normal(0, 1, num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        spectral_radius = max(abs(linalg.eigvals(weights)))
        return weights / spectral_radius



if __name__ == "__main__":
    model = ReservoirNetWork()