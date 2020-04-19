ln -s ../../data/ ./
cd sr_gnn/srgnn_pyg && python gen_data.py
ln -s sr_gnn/srgnn_pyg/datasets
cd sr_gnn/srgnn_paddle/data/diginetica && ln -s sr_gnn/srgnn_pyg/datasets/debias/raw/ ./

# srgnn_pyg train & predict
# python main.py --epoch=10  --lr=0.05
# python main.py --predict=True --model_path=../model/debias/epoch30

# srgnn_paddle train & eval & predict
# python train.py --use_cuda 1 --epoch_num 3 --model_path './saved_model_1' --lr 0.01
# python eval.py --model_path './saved_model_1/epoch_2' --test_path './data/diginetica/test.txt' --use_cuda 1