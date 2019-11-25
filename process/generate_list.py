import os
import os.path as osp

def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)

num_class = 500
skeleton_root = "/home/liweijie/Data/SLR_dataset/xf500_body_color_txt"
input_path = "../input"
create_path(input_path)
train_list = open("../input/train_list.txt","w")
val_list = open("../input/val_list.txt","w")

skeleton_path_list = os.listdir(skeleton_root)
skeleton_path_list.sort()
n = len(skeleton_path_list)
for i,skeleton_path in enumerate(skeleton_path_list):
    print("%d/%d"%(i,n))
    label = skeleton_path
    abs_skeleton_path = osp.join(skeleton_root,skeleton_path)
    skeleton_list = os.listdir(abs_skeleton_path)
    skeleton_list.sort()
    index = int(label)
    if index<num_class:
        for skeleton in skeleton_list:
            abs_skeleton = osp.join(abs_skeleton_path,skeleton)
            p = skeleton.split('_')
            person = int(p[0].lstrip('P'))
            repeat_time = int(p[3])
            record = osp.join(skeleton_path,skeleton)+"\t"+skeleton_path+"\n"
            if repeat_time==0:
                if person<=36:
                    train_list.write(record)
                else:
                    val_list.write(record)
