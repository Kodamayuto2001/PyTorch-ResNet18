# PyTorch-ResNet18
Residual Neural Network（18層）を作っています

## 2回目：失敗
- 入力する画像：MNISTの手書き文字0～9まで10種類
- 入力するチャンネル：1チャンネル
- 最適化関数：Adam
- 学習率：0.001
- epoch：10
- 
#### 損失関数のグラフ

![loss](https://user-images.githubusercontent.com/55943803/129902275-4834f77d-1da1-4ac0-bb56-c3fc822b1331.png)

#### 精度（学習が進むほど精度が悪くなっている）
![acc](https://user-images.githubusercontent.com/55943803/129902268-add268e9-9ff3-47ff-ba8a-0c0a4a045546.png)

## 1回目：失敗
- 入力する画像のサイズ：150px × 150px
- 入力するチャンネル：3チャンネル
- 最適化関数：Adam
- 学習率：0.001
- epoch：10

#### 下の画像が損失関数のepochごとのグラフ

![loss](https://user-images.githubusercontent.com/55943803/129806431-6f5194fe-72ae-47d9-a37f-c9a6c9574224.png)

#### 下の画像が精度のepochごとのグラフ

![acc](https://user-images.githubusercontent.com/55943803/129806428-23f48052-6893-4f3d-9c00-45cde62f34b9.png)
