import os
import glob

FOLDER_PATH = "/home/henry/learning2/LIRA-Federated-Learning/experiments/timagenet-fl-defenses"

files_timg = glob.glob(os.path.join(FOLDER_PATH, "**", "timage*"), recursive=True)

# print(files_timg)
print(len(files_timg))
d = [0, 0, 0]

for idf, ft in enumerate(files_timg):
    if "iba_pgd_mr" in ft:
        # print(idf, os.path.basename(ft))
        cmd = f"conda activate dungnt && CUDA_VISIBLE_DEVICES={d[2]} {ft}"
        # print(cmd)
        d[2] += 1
        # CUDA_VISIBLE_DEVICES=4 ./experiments/iba-replacement/mnist-krum-pgd.sh
        
    elif "iba_pgd" in ft:
        
        cmd = f"conda activate dungnt && CUDA_VISIBLE_DEVICES={d[1]} {ft}"
        # print(cmd)
        d[1] += 1
        # 
        # print(idf, os.path.basename(ft))
    else:
        cmd = f"conda activate dungnt && CUDA_VISIBLE_DEVICES={d[0]} {ft}"
        print(cmd)
        d[0] += 1
        pass
    