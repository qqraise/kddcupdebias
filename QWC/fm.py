import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from load_data import load_data, load_item, gen_full_data, gen_train_eval
from conf import opt
import pandas as pd
import time, datetime


class SampledSoftmaxLoss(object):
  """ The loss function implements the Dense layer matmul and activation
  when in training mode.
  """
  def __init__(self, model, num_classes, num_sampled):
    self.model = model
    output_layer = model.layers[-1]
    self.input = output_layer.input
    self.weights = output_layer.weights
    self.num_classes = num_classes
    self.num_sampled = num_sampled

  def loss(self, y_true, y_pred, **kwargs):
    labels = tf.argmax(y_true, axis=1)
    labels = tf.expand_dims(labels, -1)
    loss = tf.nn.sampled_softmax_loss(
        weights=self.weights[0],
        biases=self.weights[1],
        labels=labels,
        inputs=self.input,
        num_sampled = self.num_sampled,
        num_classes = self.num_classes,
        #partition_strategy = "div",
    )
    return loss

def get_embed(x_input, x_size, k_latent):
    if x_size > 0: #category
        embed = Embedding(x_size, k_latent, input_length=1, 
                          embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed

def build_model(X, feature_map, nums_neg=5):
    
    f_size = [dim for name, dim in feature_map.items()]
    dim_input, dim_output = len(f_size), (nums_neg+1) * 2
    
    #y_true = Input(shape=(feature_map['item_id'], ))
    input_x = [Input(shape=(1,)) for i in range(dim_input)]     
    biases = [get_embed(x, size, 1) for (x, size) in zip(input_x, f_size)]
    factors = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]
    s = Add()(factors)
    diffs = [Subtract()([s, x]) for x in factors]
    dots = [Dot(axes=1)([d, x]) for d,x in zip(diffs, factors)]
    x = Concatenate()(biases + dots)
    x = BatchNormalization()(x)
    output = Dense(dim_output, activation='relu', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    
    loss_calculator = SampledSoftmaxLoss(model, feature_map['item_id'], nums_neg)
    opt = Adam(clipnorm=0.5)
    #print(type(y_true), type(output))
    #model.add_loss(loss_calculator.loss(y_true, output))
   
    model.compile(optimizer=opt, loss=loss_calculator.loss, experimental_run_tf_function=False)
    output_f = factors + biases
    model_features = Model(inputs=input_x, outputs=output_f)
    return model, model_features

t = (2020, 4, 10, 0, 0, 0, 0, 0, 0)
time_end = time.mktime(t)
def time_info(time_delta):
    timestamp = time_end * time_delta
    struct_time = time.gmtime(timestamp)
    return (struct_time.tm_wday, struct_time.tm_hour, struct_time.tm_min)

if __name__ == '__main__':
    #1, add pretrained weights 
    #
    k_latent = 8
    embedding_reg = 0.0002
    kernel_reg = 0.1
    train_data = load_data(opt.data.base_dir+opt.data.train_file, opt.phrase)
    test_data = load_data(opt.data.base_dir+opt.data.test_file, opt.phrase)
    # qtime_data = load_data(opt.data.base_dir+opt.data.predict_file, opt.phrase, qtime=True)
    # user_df = pd.read_csv(opt.data.base_dir+opt.data.user_feat, names=opt.data.user_cols, header=None)
    # item_df = load_item(opt.data.base_dir+opt.data.item_feat)
    
    train_data['flag'] = 0
    test_data['flag'] = 1
    data = gen_full_data([train_data, test_data])
    data['pharse_id'] = data['user_id'] % 11
    data['day'],  data['hour'], data['minute'] = zip(*data['qtime'].apply(time_info))
    feature_cols = ['user_id', 'item_id', 'pharse_id', 'day', 'hour', 'minute']
    feature_map = (data[feature_cols].max()+1).to_dict()
    
    model, model_features = build_model(data, feature_map, nums_neg=5)
    print("model build complete")

    nb_classes = feature_map['item_id']
    targets = data['item_id'].values.reshape(-1)
    one_hot_targets = np.eye(nb_classes, dtype=np.int8)[targets]
    r = [data[col] for col, dim in feature_map.items()]
    print("input data gen complete")

    model.fit(x=r, y=one_hot_targets)
    print("model train done")