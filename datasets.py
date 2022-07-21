# 3rd party
import numpy as np
# 自作
from visualize import plot_2d



class InputGenerator:
    def __init__(self, start_time:float, end_time:float, num_time_steps:int):
        self.start_time = start_time
        self.end_time = end_time
        self.num_time_steps = num_time_steps 

    def get_data(self, amplitude:float=1.0):
        t = np.linspace(self.start_time, self.end_time, self.num_time_steps)
        return np.sin( np.linspace(self.start_time, self.end_time, self.num_time_steps) ) * amplitude, t


class LogisticMapDataset:
    def __init__(self, iter_num:int, skip_num:int, alpha:float, x_init:float):
        """データセットを作成"""
        self.__alpha = alpha
        self.__x = np.array([])
        self.__t = np.array([])
        t = 0
        x = x_init
        for i in range(iter_num + skip_num):
            if i >= skip_num:
                self.__x = np.append(self.__x, x)
                self.__t = np.append(self.__t, t)
                t += 1
            x = self.calc_logistic_map(x=x)
    
    def get_data(self):
        """データセットを入手"""
        return self.__x, self.__t
    
    def calc_logistic_map(self, x:'numpy.ndarray'):
        """Logistic Mapの漸化式"""
        return self.__alpha * x * (1.0 - x)
        


class HenonMapDataset:
    def __init__(self, iter_num:int, skip_num:int,
                 x_init:float, y_init:float,
                 alpha:float, beta:float):
        self.__alpha = alpha
        self.__beta = beta
        self.__x = np.array([])
        self.__y = np.array([])
        self.__t = np.array([])
        t = 0
        x = x_init
        y = y_init
        for i in range(iter_num + skip_num):
            if i >= skip_num:
                self.__x = np.append(self.__x, x)
                self.__y = np.append(self.__y, y)
                self.__t = np.append(self.__t, t)
                t += 1
            x, y = self.calc_henon_map(x=x, y=y)
        
    def get_data(self):
        """データセットを入手"""
        return (self.__x, self.__y), self.__t
    
    def calc_henon_map(self, x:'numpy.ndarray'):
        """Logistic Mapの漸化式"""
        return self.__alpha * x * (1.0 - x)



if __name__ == "__main__":
    dataset = LogisticMapDataset(iter_num=300, skip_num=0, alpha=3.9, x_init=0.01)
    # dataset = HenonMapDataset(iter=300, alpha=1.4, beta=0.3)
    x, t = dataset.get_data()
    plot_2d(x=x[:-1], y=x[1:], label_x="t", label_y="x",
            title="logistic_map", legend=None, plot_type="scatter",
            color_list=None, save_fig_path="logistic_map.png"
            )
    plot_2d(x=t, y=x, label_x="t", label_y="x",
            title="logistic_map", legend=None, plot_type="plot",
            color_list=None, save_fig_path="logistic_map_orbit.png"
            )
    