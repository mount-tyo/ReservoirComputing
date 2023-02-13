""" 可視化機能をまとめたモジュール """
import numpy as np
import matplotlib.pyplot as plt


def plot_2d(x:'numpy.ndarray',      # (X軸)plotデータ
            y:'numpy.ndarray',      # (Y軸)plotデータ
            label_x:str,            # X軸ラベル名
            label_y:str,            # Y軸ラベル名
            title:str,              # タイトル名
            plot_type:str,          # プロットタイプ(scatter, etc.)
            save_fig_path:str,      # 画像を保存するときのpath
            legend:list=None,       # 凡例名のリスト
            color_list:list=None,   # 複数の凡例がある場合は、使用する色を指定可能
            ):
    if plot_type == "scatter":
        plt.scatter(x, y, color=color_list)
    else:
        plt.plot(x, y, color=color_list)
    if legend is not None:
        plt.legend(legend)
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.savefig(save_fig_path)
    plt.clf()
    plt.close()