# ToDo
- [ ] 汎化性能チェック
- [ ] --characterで日本語テキストを参照するようにする
- [ ] 編集距離を表示する
- [ ] Rosettaモデルで学習-テスト
- [ ] STAR-Netモデルで学習-テスト

# シーンテキスト認識モデルの比較、データセットおよびモデル分析では、何が間違っているか?	
大半の既存STRモデルが採用されている4段階のSTRフレームワークの公式PyTorch実装により、一貫したトレーニングと評価データセットの下で、精度、スピード、メモリ需要の面でモジュールごとに性能に貢献することが可能となり、現在の比較に対する障害を解消し、既存のモジュールの性能利得を理解することができます。   
<br><br>
<img src="./figure/trade-off.jpg" width="1000" title="trade-off">

# Getting Started
## 依存関係
- [元のソースコード](https://github.com/clovaai/deep-text-recognition-benchmark)では  
PyTorch 1.1.0, CUDA 9.0, python 3.6 and Ubuntu 16.04. でテストを実行
- ~`pip3 install torch==1.1.0`が必要~
- `pip3 install torch==1.2.0`が必要
- 要件 : lmdb, pillow, torchvision, nltk, natsort
`pip3 install lmdb pillow torchvision nltk natsort`

## トレーニングと評価用のimdbデータセットを[ここ](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt)からダウンロードします。
- data_lmdb_release.zipには以下が含まれます。　
- training datasets : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)[1] and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)[2] 　
- validation datasets : トレーニングセットの結合 [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6]
- evaluation datasets : ベンチマーク評価データセットは[IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6], [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)[7], [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)[8], [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)[9]

## 学習済みモデルのデモ
1. 学習済みモデルを[ここ](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)からダウンロード
2. `demo_image/`にテストしたい画像を入れる
3. `demo.py`を実行(大文字と小文字を区別するモデルを使用する場合は--sensitiveオプションを追加)   
```
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ --saved_model TPS-ResNet-BiLSTM-Attn.pth
```

## 学習と評価
1. CRNN[10]モデルの学習   
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training \
--valid_data data_lmdb_release/validation --select_data MJ --batch_ratio 1.0 \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --data_filtering_off
```   

2. CRNN[10]モデルのテスト
```
CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC \
--saved_model saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth
```

3. 最良の精度の組み合わせ(TPS-ResNet-BiLSTM-Attn)もトレーニングし、テストするようにしましょう。
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
```
```
CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
```

~__2019/08/29現在、テストまでやっていない__~   
__2019/09/06現在、テスト結果は3割ぐらい正解__ ==> Model:None-VGG-BiLSTM-CTC iter:10000

## [ここ](https://drive.google.com/drive/folders/1W84gS9T5GU5l5Wp3VV1aeXIIKV87yjRm)からダウンロードした故障ケースと洗浄したラベル
image_release.zipには、不具合事例画像と、清浄化されたラベルを持つベンチマーク評価画像が含まれています。   
<img src="./figure/failure-case.jpg" width="1000" title="failure cases">

# 引数
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--continue_model', default='', help="path to model to continue training")
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
## lmdbデータセットを作成する必要がある場合	
```
pip3 install fire
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```   
この時点で、gt.txtは{imagepath}\t{label}\nである必要があります。   
例：
```
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...
```

## グランドトゥルースの正規化
`python3 text_normalize.py --inputpath data/SynthText800000/train_text.txt --outputpath data/SynthText_800000_normalize \
 --output_file_name normalized_train_text.txt`
 
 グランドトゥルースを**NFKC**で正規化する。
