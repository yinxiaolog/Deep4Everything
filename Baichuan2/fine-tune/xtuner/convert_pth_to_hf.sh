CONFIG=/data/yinxiaoln/code/Baichuan2/fine-tune/xtuner/bc2_7b_cqae3.py
PATH_TO_PTH_MODEL=/data/yinxiaoln/save/epoch_10.pth
SAVE_PATH_TO_HF_MODEL=/data/yinxiaoln/save/epoch_10_hf
xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL
