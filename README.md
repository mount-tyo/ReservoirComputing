# Reservoir Computing

## 概要

Reservoir Networkの1つであるecho state networkを構築し、規則的と非規則的な時系列データセットそれぞれを用意し、学習させる。

## Reservoir Computingとは？

計算量が少なく、時系列データに対して有用な再帰的なニューラルネットワーク(Recurrent Neural Network; RNN)の1種である。
詳細な説明は以下の記事に譲る。
- [Qiita, ゼロから作るReservoir Computing](https://qiita.com/pokotsun/items/dd8eb48fadeee052110b)

## Datasets

### 規則的な時系列データ

※周期的な挙動が見られるデータセットを採用した
- Sin Wave

### 不規則な時系列データ

※カオス現象が見られる時系列データセットを採用した
- Logistic Map
- Henon Map

## Environment

- OS: MacOS Ventura 13.1
- Python 3.9.6

## Installation

- `venv`で仮想環境を作成

```bash
# venvの作成
python -m venv venv
# venvの有効化
source venv/bin/activate
# pipのアップグレード
pip install --upgrade pip
# パッケージのインストール
pip install -r requirements.txt
```

## Usage

- training

```bash
# 実行例
python train.py
```

## Directory Structure

```txt
.
├── README.md                       # 最初に読むべきファイル
├── datasets.py                     # データセットを定義
├── echo_state_network.py           # モデルを定義
├── images                          # 可視化結果のdir(自身で作成する必要あり)
├── requirements.txt                # パッケージの依存関係
├── train.py                        # リザバーコンピューティングの学習
└── visualize.py                    # 可視化機能
```