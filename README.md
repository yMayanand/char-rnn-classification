# 🧠 char-rnn-classification
predicting country of origin from names 👱 ➡️ 🏴󠁧󠁢󠁥󠁮󠁧󠁿, 👱‍♂️ ➡️ 🇮🇳
we use **lstm** to classify names wherein names are fed to it character by character and then after end of the sequence we get vector encoded with the name which will be used to classify its country of origin.

## usage
### Training

```
python train.py --epoch 1000 --bs 128 --lr 1e-5 --opt AdamW --emb_size 512 --hidden_size 512
```

* `--epoch`: number of epochs
* `--bs`: batch size
* `--lr`: learning rate
* `--opt`: pytorch optimizer name
* `--emb_size`: embedding size
* `--hidden_size`: hidden layer size of lstm

you can run training on cpu or gpu, code will automatically detect and run training on best available device

### Inference

```
python predict.py --name wang --emb_size 512 --hidden_size 512
```

* `--dropout`: dropout rate
* `--name`: name to predict
* `--emb_size`: embedding size
* `--hidden_size`: hidden layer size of lstm

Note: checkpoint available in `./checkpoint` directory has `--emb_size` 512 and `--hidden_size` 512
