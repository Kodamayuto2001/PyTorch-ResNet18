# PyTorch-ResNet18
Residual Neural Network（18層）を作っています

## 検証：4回目
### とりあえず動いた
- 1層目(conv1,bn1,relu)：128 × 1 x 256 x 256 -> 128 x 64 x 128 x 128
- 2層目(maxpool)：-> 128 x 64 x 64 x 64
- 3層目と4層目(conv3x3,bn1,relu,スキップコネクションがある)：-> 128 x 64 x 64 x 64 
- 5層目と6層目(conv3x3,bn2,relu,スキップコネクションがある)：-> 128 x 64 x 64 x 64
- 7層目と8層目(conv3x3,bn1,relu,スキップコネクションがある)：-> 128 x 128 x 32 x 32
- 9層目と10層目(conv3x3,bn2,relu,スキップコネクションがある)：-> 128 x 128 x 32 x 32
- 11層目と12層目(conv3x3,bn1,relu,スキップコネクションがある)：-> 128 x 128 x 16 x 16 
- 13層目と14層目(conv3x3,bn2,relu,スキップコネクションがある)：-> 128 x 128 x 16 x 16
- 15層目と16層目(conv3x3,bn1,relu,スキップコネクションがある)：-> 128 x 512 x 8 x 8
- 17層目と18層目(conv3x3,bn2,relu,スキップコネクションがある)：-> 128 x 512 x 8 x 8
- avgpool：128 x 512 x 2 x 2
- 線形変換：512 x 2 x 2 -> 3
- イノシシかカラスかシカに分類

## 検証：3回目
- データセット：MNIST
- 参考にした記事：https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb
- 学習回数：10回
- モデル：ResNet-18
- 最適化関数：Adam
- 学習率：0.001
- バッチサイズ：128
- チャンネル：1(GrayScale)
- 最後の活性化関数：F.softmax関数
- 損失関数：F.cross_entropy関数
- 入力のサイズ：28（MNISTのまま）
- 分類数：10クラス

#### 損失関数と評価関数とかかった時間
##### 学習：1回目
![image](https://user-images.githubusercontent.com/55943803/130061656-33a84950-8ce7-4806-bd50-557ce477dee9.png)

##### 学習：2回目
![image](https://user-images.githubusercontent.com/55943803/130061812-6455339e-b8eb-4aff-96b8-bdccc68e5620.png)

##### 学習：3回目
![image](https://user-images.githubusercontent.com/55943803/130061883-a97be024-0afd-42c4-9c63-b507d6f62c17.png)

##### 学習：4回目
![image](https://user-images.githubusercontent.com/55943803/130061941-69f5358d-5e4d-45c0-85b4-0c2b1b685781.png)

##### 学習：5回目
![image](https://user-images.githubusercontent.com/55943803/130061995-228f88be-f45d-49ba-9f53-9bacbddf501c.png)

##### 学習：6回目
![image](https://user-images.githubusercontent.com/55943803/130062055-4bf72960-e48c-456a-999e-3272bcebd7e8.png)

##### 学習：7回目
![image](https://user-images.githubusercontent.com/55943803/130062175-83e3a3c2-e934-49ff-bd27-f55d78384e26.png)

##### 学習：8回目
![image](https://user-images.githubusercontent.com/55943803/130062287-4ace7ab1-ebce-44f1-a099-ab756cf3e596.png)

##### 学習：9回目
![image](https://user-images.githubusercontent.com/55943803/130062353-812cf60a-40f9-4624-b792-3213146fff50.png)

##### 学習：10回目
![image](https://user-images.githubusercontent.com/55943803/130062418-2b6c988b-6e6f-4f46-9204-d5727a27ffd1.png)

#### 全体のテストデータの評価の結果
##### 精度：99.20%
![image](https://user-images.githubusercontent.com/55943803/130062517-77692e5b-58f9-4d65-88a1-123ec2daa05a.png)

##### テストデータから1枚選んで評価した結果
##### 精度：100%
![image](https://user-images.githubusercontent.com/55943803/130062678-a69a69ef-7994-48db-8cfd-797559d07988.png)


## 検証：2回目
- 入力する画像：MNISTの手書き文字0～9まで10種類
- 入力するチャンネル：1チャンネル
- 最適化関数：Adam
- 学習率：0.001
- epoch：10

#### 損失関数のグラフ
![loss](https://user-images.githubusercontent.com/55943803/129902275-4834f77d-1da1-4ac0-bb56-c3fc822b1331.png)

#### 精度（学習が進むほど精度が悪くなっている）
![acc](https://user-images.githubusercontent.com/55943803/129902268-add268e9-9ff3-47ff-ba8a-0c0a4a045546.png)

#### 考察
- ResNet18のモデルに問題がある可能性が高い


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
