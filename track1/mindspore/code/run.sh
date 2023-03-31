# python code_mindspore/pre/pre_main.py
# python code_mindspore/train.py --workers 12 --end_epoch 25 --batchsize 8 --total_epoch 100 --end_epoch 23 --lr 0.00005
python code_mindspore/infer_20221114.py --checkpoint_path "./model/epoch_22.ckpt" --checkpoint_path2 "./model/epoch_23.ckpt"
# python code_mindspore/infer_20221120.py --checkpoint_path "./model/epoch_22.ckpt" --checkpoint_path2 "./model/epoch_23.ckpt" --checkpoint_path3 "./model/epoch_21.ckpt"