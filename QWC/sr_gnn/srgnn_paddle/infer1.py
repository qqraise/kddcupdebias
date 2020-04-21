import argparse
import logging
import numpy as np
import os
import paddle
import paddle.fluid as fluid
import reader
import network

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DIN example")
    parser.add_argument(
        '--model_path', type=str, default='./saved_model/', help="path of model parameters")
    parser.add_argument(
        '--test_path', type=str, default='./data/diginetica/test.txt', help='dir of test file')
    parser.add_argument(
        '--config_path', type=str, default='./data/diginetica/config.txt', help='dir of config')
    parser.add_argument(
        '--use_cuda', type=int, default=1, help='whether to use gpu')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument(
        '--start_index', type=int, default='0', help='start index')
    parser.add_argument(
        '--last_index', type=int, default='10', help='end index')
    parser.add_argument(
        '--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument(
        '--step', type=int, default=1, help='gnn propogation steps')
    return parser.parse_args()

def nlargest(arr, top_k=50):
    print(arr.shape, arr)
    ind = np.argpartition(arr, -top_k)[-top_k:]
    return ind[np.argsort(arr[ind])][::-1]

def infer(args):
    batch_size = args.batch_size
    items_num = reader.read_config(args.config_path)
    test_data = reader.Data(args.test_path, False)
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    loss, acc, py_reader, feed_datas, logits= network.network(items_num, args.hidden_size, args.step, args.batch_size)
    exe.run(fluid.default_startup_program())
    
    
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=args.model_path, executor=exe)
    
    feed_list = [e.name for e in feed_datas]

    infer_reader = test_data.reader(batch_size, batch_size*20, False)
    feeder = fluid.DataFeeder(place=place, feed_list=feed_list)
    for iter, data in enumerate(infer_reader()):
        
        res = exe.run(infer_program,
                      feed=feeder.feed(data),
                              fetch_list=target_var)
        logits, acc = res
        if iter == 0:
            break
    print('session:', data[0][:-1], 'label:',np.argmax(logits))
    print('label:',np.argmax(logits), 'label-50:', nlargest(logits[0]))



if __name__ == "__main__":
    args = parse_args()
    infer(args)