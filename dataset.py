import torch.utils.data as data

from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import os
import os.path as osp
import time

class VideoRecord(object):
    def __init__(self,row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def skeleton_path(self):
        return self._data[1]

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class iSLR_Dataset(data.Dataset):
    
    def __init__(self, list_file,
                transform=None,  args=None,
                modality='RGB', width=1280,height=720
               ):
        self.video_root = args.video_root
        self.skeleton_root = args.skeleton_root
        self.list_file = list_file
        self.transform = transform
        self.length = args.length
        self.image_length = args.image_length
        self.train_mode = args.train_mode
        self.augmentation = args.augmentation
        #TODO not hard code
        self.modality = modality
        self.width = width
        self.height = height
        
        self._parse_list()

    def _load_image(self, directory, idx):
        path_list = os.listdir(osp.join(self.video_root,directory))
        path_list.sort()
        image_name = osp.join(self.video_root,directory,path_list[idx])
        if self.modality == 'RGB':
            try: 
                return [Image.open(image_name).convert('RGB')]
            except Exception:
                print('error loading image:', osp.join(self.video_root, directory, path_list[idx]))
                return [Image.open(osp.join(self.video_root, directory, path_list[0])).convert('RGB')]
        
    def _parse_list(self):
        tmp = [x.strip().split('\t') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[2])>4]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def get_sample_indices(self,num_frames):
        indices = np.linspace(1,num_frames-1,self.length).astype(int)
        interval = (num_frames-1)//self.length
        if interval>0:
            jitter = np.random.randint(0,interval,self.length)
        else:
            jitter = 0
        jitter = (np.random.rand(self.length)*interval).astype(int)
        indices = np.sort(indices+jitter)
        indices = np.clip(indices,0,num_frames-1)
        skeleton_indices = indices
        image_indices = indices[::self.length//self.image_length]
        return skeleton_indices,image_indices
    
    def _load_data(self, path):
        # 对于openpose数据集要处理训练列表
        path = path.rstrip("body.txt")+"color"
        path = osp.join(self.skeleton_root, path)
        file_list = os.listdir(path)
        file_list.sort()
        mat = []
        for i,file in enumerate(file_list):
            # 第一帧有问题，先排除
            if i>0:
                filename =  osp.join(path,file)
                f = open(filename,"r")
                content = f.readlines()
                try:
                    mat_i = self.content_to_mat(content)
                    mat.append(mat_i)
                except:
                    print("can not convert this file to mat: "+filename)
        mat = np.array(mat)
        end = time.time()
        mat = mat.astype(np.float32)
        return mat

    def content_to_mat(self,content):
        mat = []
        for i in range(len(content)):
            if "Body" in content[i]:
                for j in range(25):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)
            elif "Face" in content[i]:
                for j in range(70):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)

            elif "Left" in content[i]:
                for j in range(21):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)

            elif "Right" in content[i]:
                for j in range(21):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)
                break
        # for i in range(1,26):
        #     record = content[i].lstrip().lstrip("[").rstrip("\n").rstrip("]")
        #     joint = [float(x) for x in record.split()]
        #     mat.append(joint)
        # for i in range(27,97):
        #     record = content[i].lstrip().lstrip("[").rstrip("\n").rstrip("]")
        #     joint = [float(x) for x in record.split()]
        #     mat.append(joint)
        # for i in range(98,119):
        #     record = content[i].lstrip().lstrip("[").rstrip("\n").rstrip("]")
        #     joint = [float(x) for x in record.split()]
        #     mat.append(joint)
        # for i in range(120,141):
        #     record = content[i].lstrip().lstrip("[").rstrip("\n").rstrip("]")
        #     joint = [float(x) for x in record.split()]
        #     mat.append(joint)

        mat = np.array(mat)
        # 第三维是置信度，不需要
        mat = mat[:,0:2]
        return mat

    def __getitem__(self, index):
        record = self.video_list[index]
        # get mat
        mat = self._load_data(record.skeleton_path)
        num_frames = record.num_frames if record.num_frames<mat.shape[0]\
            else mat.shape[0]
        skeleton_indices,image_indices = self.get_sample_indices(num_frames)
        MatForImage = mat[image_indices,:,:]
        mat = mat[skeleton_indices,:,:]
        # mat: T J D
        
        if self.augmentation:
            # view invarianttransform
            # mat = view_invariant_transform(mat)
            # select the hand joint
            # data augmentation
            # mat = self.random_augmentation(mat)

            # get the four corner
            x = MatForImage[:,:,0]
            y = MatForImage[:,:,1]
            min_x =  int(np.min(x[x>0]))
            min_y =  int(np.min(y[y>0]))
            max_x = int(np.max(x[x>0]))
            max_y = int(np.max(y[y>0]))
            min_x,min_y,max_x,max_y = self.random_generate_min(min_x,min_y,max_x,max_y)
            MatForImage = MatForImage-np.array([min_x,min_y])
            MatForImage = MatForImage/np.array([max_x-min_x,max_y-min_y])
        else:
            MatForImage = mat
            MatForImage = MatForImage/np.array([self.width,self.height])

        if "rgb" in self.train_mode or "fusion" in self.train_mode:
            # generate heatmaps
            heat_maps = []
            for i in range(MatForImage.shape[0]):
                heat_map =[]
                for j in range(MatForImage.shape[1]):
                    x,y = MatForImage[i,j,:]
                    z = self.generate_gaussian(x,y)
                    # if j==0:
                    #     plt.subplot(4,4,i+1)
                    #     plt.imshow(z)
                    heat_map.append(z)
                heat_map = np.stack(heat_map,0)
                heat_maps.append(heat_map)
            # plt.show()
            heat_maps = np.stack(heat_maps,0)
        
            # get images
            images = list()
            for i,ind in enumerate(image_indices):
                img = self._load_image(record.path, ind)
                if self.augmentation:
                    img = crop_img(img,min_x,min_y,max_x,max_y)
                else:
                    pass
                images.extend(img)
            
            images = self.transform(images)
        
            return mat, images, heat_maps, record.label
        
        elif self.train_mode == "single_skeleton":
            return mat, 0, 0, record.label

    def __len__(self):
        return len(self.video_list)

    def generate_gaussian(self,x,y):
        class Distribution():
            def __init__(self,mu,Sigma):
                self.mu = mu
                self.sigma = Sigma

            def two_d_gaussian(self,x):
                mu = self.mu
                Sigma =self.sigma
                n = mu.shape[0]
                Sigma_det = np.linalg.det(Sigma)
                Sigma_inv = np.linalg.inv(Sigma)
                N = np.sqrt((2*np.pi)**n*Sigma_det)

                fac = np.einsum('...k,kl,...l->...',x-mu,Sigma_inv,x-mu)

                Z = np.exp(-fac/2)/N
                Z = (Z-Z.min())/(Z.max()-Z.min())
                if Z.max()-Z.min()==0:
                    print("waring, div zero")
                    Z = (Z-Z.min())/(Z.max()-Z.min()+1e-6)
                return Z
        N = 10
        X = np.linspace(0,1,N)
        Y = np.linspace(0,1,N)
        X,Y = np.meshgrid(X,Y)
        mu = np.array([x,y])
        Sigma = np.array([[0.01,0],[0,0.01]])
        pos = np.empty(X.shape+(2,))
        pos[:,:,0]= X
        pos[:,:,1] = Y

        p2 = Distribution(mu,Sigma)
        Z = p2.two_d_gaussian(pos)
        del p2
        return Z

    def random_augmentation(self,mat):
        choice = np.random.randint(0,2,4)
        if choice[0]==1:
            mat = self.random_jitter(mat)
        # elif choice[1]==1:
        #     mat = self.random_shift(mat)
        return mat

    def random_jitter(self,mat):
        # input: T J D
        jitter_amp = 10
        delta  = np.random.randint(0,jitter_amp,mat.shape)
        mat = mat+delta
        return mat

    def random_scale(self, mat):
        # input: T J D
        min = 0.8
        max = 1.2
        import random
        scale =  random.random()*(max-min)+min
        mat = mat*scale
        return mat

    def random_shift(self,mat):
        shift_amp = 20
        xshift = np.random.randint(-shift_amp,shift_amp)
        yshift = np.random.randint(-shift_amp,shift_amp)
        mat[:,:,0] = mat[:,:,0]+xshift
        mat[:,:,1] = mat[:,:,1]+yshift
        return mat

    def random_generate_min(self,min_x,min_y,max_x,max_y):
        min_thre = 100
        max_thre = 200
        random_x = np.random.randint(min_thre,max_thre)
        random_y = np.random.randint(min_thre,max_thre)
        min_x = max(min_x-random_x,0)
        min_y = max(min_y-random_y,0)
        max_x = min(max_x+random_x,self.width)
        max_y = min(max_y+random_y,self.height)
        return min_x,min_y,max_x,max_y

def view_invariant_transform(mat):
    '''
      @params mat: T J D
    '''
    # index1 = 12
    # index2 = 16
    index1 = 9
    index2 = 12

    new_mat = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        if 0 in mat[i,[index1,index2],:]:
            print(mat[i,[index1,index2],:])
        delta_x,delta_y = mat[i,index2,:]-mat[i,index1,:]
        center_x,center_y = 0.5*(mat[i,index2,:]+mat[i,index1,:])
        length = (delta_x*delta_x+delta_y*delta_y)**0.5
        cos_theta = delta_x/length
        sin_theta = delta_y/length
        T = np.array([
            [cos_theta,-sin_theta],
            [-sin_theta,-cos_theta]
        ])
        t = np.array([center_x,center_y])
        # 对一帧中所有坐标进行具有视角不变性的变换
        # x'=Tx
        # origin_coord: J D
        origin_coord = mat[i,:,:]-t
        new_coord = np.matmul(T,origin_coord.transpose())
        new_coord = new_coord.transpose()
        new_mat[i,:,:] = new_coord
    return new_mat


def crop_img(image,min_x,min_y,max_x,max_y):
    '''
        @param image: PIL.Image
    '''
    new_image = []
    for img in image:
        img = np.array(img)
        img = img[min_y:max_y,min_x:max_x]
        img = Image.fromarray(img)
        new_image.append(img)
    return new_image