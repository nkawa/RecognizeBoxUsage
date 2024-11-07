

import torch
from torch import jit
from torchvision.transforms import functional as TF
from scipy.special import softmax
from datetime import datetime

import cv2
from enum import Enum
import numpy as np
import os

from ptokai_box_info import ptokai_box_info

class Box(Enum):
    FLOOR = 0
    GOODS = 1
    EMPTY = 2
    HAND = 3
    OTHER = 4

device = torch.device("cuda", 0)
model = jit.load("/mnt/bigdata/01_projects/2024_trusco/box_model/model.pt", map_location=device)


def do_recog(vfile, save_file):
    cap = cv2.VideoCapture(vfile)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Recognizing", vfile, "With:" ,frames, "frames")

    with open(save_file, "w") as f:
        f.write("frame_id,box_no,pred_result,pred_prob, pred_name,second_res,second_prob, second_name\n")
        for frm_count in range(int(frames)):
            ok ,frm = cap.read()
            if not ok:
                print("Can't read frame", frm_count,ok)
                break

            for box in range(1,len(ptokai_box_info)+1):
                x, y = ptokai_box_info[box][0]
                if x - 3885 - 50 < 0 or y - 813 - 50 < 0:
                    continue
                with torch.no_grad():
            # 1. フレームをnumpy配列からテンソルに変換
            #    0-255のuint8から0-1のfloat32に、(h, w, c)から(c, h, w)に自動で変換される
            # 2. 1つのボックスがちょうど収まるようにクロップ
            # 3. 64x64にリサイズ

                    input = TF.resized_crop(TF.to_tensor(frm), y - 813 - 50, x - 3885 - 50, 100, 100, (64, 64))
                    output = model(input.to(device=device))
                    outcpu = output.cpu()
                    pred = softmax(outcpu.numpy())
                    fargs = np.argsort(pred)[0]
                    plist = pred.tolist()[0]
                    probs = [int(p*1000)/10 for p in plist]
                    #print(probs, fargs)
                    pmax = fargs[-1]
                    psec = fargs[-2]
#                print(frm_count, box, probs[-1],  Box(args[-1]).name)
                    if frm_count % 100 == 0 and box == 1:
                        print(f"{frm_count},{box},{pmax},{probs[pmax]},{Box(pmax).name},{psec},{probs[psec]},{Box(psec).name}")
                    f.write(f"{frm_count},{box},{pmax},{probs[pmax]},{Box(pmax).name},{psec},{probs[psec]},{Box(psec).name}\n")

        

#video_file = "/mnt/gazania/trusco-stitch/2024/2024-10-03/new_small_overlap_1100_1200_2x_nkawa1934.mp4"
#""/mnt/bigdata/01_projects/2024_trusco/box_result/2024-10-03_1100_1200_2x_frame.csv"

video_index = ["0700_0800", "0800_0900", "0900_1100"]

for vi in video_index:
    vname = f"/mnt/gazania/trusco-stitch/2024/2024-10-03-h265/new_small_overlap_{vi}_2x_nkawa1934.mp4"
    savename = f"/mnt/bigdata/01_projects/2024_trusco/box_result/2024-10-03_{vi}_2x_frame.csv"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("start",current_time)
    print(vname, os.path.exists(vname))
    print(savename)
    do_recog(vname, savename)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("done",current_time)