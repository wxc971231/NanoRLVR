# train_rlvr.py
# 主入口：DDP/单卡、训练循环、日志、保存/加载ckpt
# 目标：你改算法主要改这里的 compute_loss 分支即可

import os
import sys
from dotenv import load_dotenv
load_dotenv()

def main():
    print("NanoRLVR training script")
    # TODO: Implement training loop

if __name__ == "__main__":
    main()
