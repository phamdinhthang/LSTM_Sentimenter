# LSTM Sentimenter


### Sentiment analysis using Long-short Term Memory, inplemented in tensorflow
Long Short Term Memory (LSTM) is one type of Recurrent Neural Network, which has the ability to capture long-term dependencies of sequence data (such as words).
In LSTM, output of the previous hidden state not only connect to the next state via weights (which is vulnerable to Vanishing Gradient), but also via a chain connection called "cell memory state". Cell memory state flows through the entire recurrent process of the network through simple matrix element-wise operation such as add, multiple, hence memory state is somehow invulnerable to Vanishing Gradient.

![LSTM Cell](https://github.com/phamdinhthang/LSTM_Sentimenter/blob/master/misc/lstm.png "")

Long Short Term Memory cell. Image [source](https://cdn-images-1.medium.com/max/1600/1*Niu_c_FhGtLuHjrStkB_4Q.png)

In a LSTM cell, the forget gate and update gate, which is put under Sigmoid activation, controls how "much" a past cell state can flow to the next cell.

### Usage

For demonstration, two public datasets were used:
- IMDB review dataset. ([source](http://ai.stanford.edu/~amaas/data/sentiment/))
- Movie review dataset. ([Source](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data))

To train the LSTM model:

```
python train_sentimenter.py data_path learning_rate epochs batch_size lstm_units device_name
```

Train parameters:
* data_path: absolute path to the dataset folder
* learning_rate: Adam Optimizer learning rate
* epochs: Number of epochs
* batch_size: size of the mini_batch
* lstm_units: number of hidden units in a LSTM cell
* device_name: train the model using CPU/GPU ('cpu' or 'gpu')
