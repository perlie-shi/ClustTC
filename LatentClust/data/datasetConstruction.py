import logging
import os
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
import os
import platform
from collections import defaultdict
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# logger = logging.getLogger(__name__)


def inner_data_processing(inner_data):
    inner_data_merge = {}
    batch = len(inner_data)
    for key in inner_data[0]:
        inner_data_merge[key] = []
    for inner_data_item in inner_data:
        for key in inner_data_item:
            inner_data_merge[key].append(torch.tensor(inner_data_item[key]))

    for key in inner_data_merge:
        inner_data_merge[key] = torch.stack(inner_data_merge[key],dim=0).reshape(batch,-1).type(torch.float)

    return inner_data_merge

def seq_collate(data):
    # 将一个batch的数据打包在一起
    (obs_seq_list,      pred_seq_list, 
     obs_seq_rel_list,  pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list,
     obs_traj_Me,       pred_traj_gt_Me, 
     obs_traj_rel_Me,   pred_traj_gt_rel_Me,
     obs_date_mask,     pred_date_mask,
     meteo_obs,meteo_pre,inner_data,ty_seqInfoZip) = zip(*data)

    #obs_seq_list: 一个batch的obs轨迹 16*[1,2,8]
    _len = [len(seq) for seq in obs_seq_list] 
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # 将一个batch的轨迹拼接在一起
    # Data format: [batch, input_size, seq_len]
    # transpose to: [seq_len, batch, input_size]  (LSTM input)
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end) # batch中轨迹的index
    obs_traj_Me = torch.cat(obs_traj_Me, dim=0).permute(2, 0, 1)
    pred_traj_Me = torch.cat(pred_traj_gt_Me, dim=0).permute(2, 0, 1)
    obs_traj_rel_Me = torch.cat(obs_traj_rel_Me, dim=0).permute(2, 0, 1)
    pred_traj_rel_Me = torch.cat(pred_traj_gt_rel_Me, dim=0).permute(2, 0, 1)
    obs_date_mask = torch.cat(obs_date_mask, dim=0).permute(2, 0, 1)
    pred_date_mask = torch.cat(pred_date_mask, dim=0).permute(2, 0, 1)
    meteo_obs = torch.stack(meteo_obs, dim=0).permute(0,4,1,2,3) # [16,2,8,64,64] [batch, meteo_num, obs_len, h, w]
    meteo_pre = torch.stack(meteo_pre, dim=0).permute(0,4,1,2,3)
    inner_data = inner_data_processing(inner_data) # dict: {[batchsize, -1]，...}
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, 
        non_linear_ped, loss_mask, seq_start_end, 
        obs_traj_Me, pred_traj_Me, obs_traj_rel_Me, pred_traj_rel_Me,
        obs_date_mask, pred_date_mask, meteo_obs, meteo_pre, inner_data, ty_seqInfoZip
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    add = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if line[6].endswith(('00', '06', '12', '18')):
                add.append(line[-2:])
                main_data = [float(i) for i in line[:-2]]
                data.append(main_data)
            else:
                continue
    return {'main':np.asarray(data),'addition':add}


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of pre trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


# 调用
# TrajectoryDataset(
#         path, #这里传入的是self-data，数据是归一化之后的单个tc的目标txt文件
#         obs_len=args.obs_len,
#         pred_len=args.pred_len,
#         stride=args.stride,
#         delim=args.delim,
#     meteo = args.meteo)

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=4, stride=1, threshold=0.002,
        min_ped=1, delim='\t', meteo='gph', save_path=None
    ):
        """
        Args:
        - data_dir: self-data path
        - obs_len: length of input trajectories
        - pred_len: length of output trajectories
        - stride: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len # 8
        self.pred_len = pred_len # 4
        self.stride = stride
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.meteo_name = meteo
        self.save_path = save_path

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] #获取所有txt文件路径
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_list_date_mask = []
        loss_mask_list = []
        non_linear_ped = []
        ty_seqInfoZip = []

        # construct sequence of self-data
        for path in all_files:
            _,x = os.path.split(path)
            tyname = os.path.splitext(x)[0]
            data = read_file(path, delim)
            addinf = data['addition'] # timestamp, tc_name
            data = data['main'] # timeNUM，TC_id，lon，lat, pre, wind
            frames = np.unique(data[:, 0]).tolist() #timeNUM去重（每条record为一帧）
            frame_data = [] #存储所有的轨迹
            for frame in frames:
                # 将txt文件中同一帧的，没有目标的坐标点保存在同一个frame_data的同一个index中
                # 对于tc来说，这些操作是不必要的，因为每一个record都不会有重复
                frame_data.append(data[frame == data[:, 0], :])

            # 获取一个txt文件中的轨迹可以分割成多少个子轨迹
            # 如第一个子轨迹：1,2,3...20，第二2,3,4，...21
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / stride))
            # 迭代子轨迹
            for idx in range(0, num_sequences * self.stride + 1, stride):
                
                # axis=0  照着row的方向叠在一起
                curr_seq_data = np.concatenate( #轨迹序列，长度为12
                    frame_data[idx:idx + self.seq_len], axis=0)
                
                # peds_in_curr_seq  存储当前子序列的目标id，   ty的话   应该就是1
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) #[1.]

                # 第一维的1 表示的是目标id， 第二维表示的是lon lat pre wind，第三维表示的是时间
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4, self.seq_len)) #(1,4,12)
                curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))    #(1,4,12)

                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len)) #(1,12)
                # 获取时间信息
                curr_date_mask = np.zeros((len(peds_in_curr_seq), 4, self.seq_len)) #(1,4,12)
                
                #每一条轨迹中的 num_peds 是多少,与peds_in_curr_seq的长度一致
                num_peds_considered = 0

                # 存储当前子轨迹的pre轨迹是否是线性轨迹
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    
                    # 后续如果考虑TC的交互影响，就比较有意义
                    #这两句对单TC并无参考意义，因为单个TC的ped_id都是1
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == #获取序列12中相同id的轨迹
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # 判断当前子轨迹的长度是否满足要求，否则舍弃
                    # idx是当前子轨迹的起始位置
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len: #这个判断没看懂含义
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) #将 lon lat pre wind 转置
                    curr_ped_seq = curr_ped_seq

                    #取出当前轨迹中所有的时间信息并embedding
                    curr_ped_date_mask = [x[0] for x in addinf[idx:idx+pred_len+obs_len]]
                    curr_ped_date_mask = self.embed_time(curr_ped_date_mask) # （1，4，12）将每一个时间embedding成四维的
                    
                    # Make coordinates relative
                    # rel_curr_ped_seq 存相邻两个坐标点中间的差  后-前
                    # 计算相邻两个record之间的差值（包括经纬度和风速气压差）
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_date_mask[_idx, :, pad_front:pad_end] = curr_ped_date_mask

                    # 判断当前子轨迹中的pre轨迹是否是线性轨迹
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                # if num_peds_considered > min_ped: 源码---
                # - min_ped: Minimum number of pedestrians that should be in a seqeunce
                # 最小的行人个数，1的话  应该是至少有一个  但是源码缺大于1，有点问题。。。
                # 但是因为台风基本上同个时间都只有一个，所以我改成>=1
                # 判断每一个子轨迹中的num_peds_considered是否大于min_ped
                if num_peds_considered >= min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered]) # 当前子轨迹
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_list_date_mask.append(curr_date_mask[:num_peds_considered])
                    
                    ty_seqInfoZip.append({'tyInfoSeqIdx':[tyname,idx],\
                                 'lastobsName':addinf[idx+self.obs_len-1],\
                                 'seqdate':[x[0] for x in addinf[idx:idx+pred_len+obs_len]]})

        self.num_seq = len(seq_list) # 所有的子轨迹序列
        seq_list = np.concatenate(seq_list, axis=0) #(29037, 4, 12)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_date_mask = np.concatenate(seq_list_date_mask, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy( #所有子轨迹的 obs部分
            seq_list[:, :2, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(#所有子轨迹的 pred部分
            seq_list[:, :2, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :2, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :2, self.obs_len:]).type(torch.float)
        self.obs_traj_Me = torch.from_numpy( # pre wind
            seq_list[:, 2:, :self.obs_len]).type(torch.float)
        self.pred_traj_Me = torch.from_numpy(
            seq_list[:, 2:, self.obs_len:]).type(torch.float)
        self.obs_traj_rel_Me = torch.from_numpy(
            seq_list_rel[:, 2:, :self.obs_len]).type(torch.float)
        self.pred_traj_rel_Me = torch.from_numpy(
            seq_list_rel[:, 2:, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float) #[29037, 12]
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        self.obs_date_mask = torch.from_numpy(
            seq_list_date_mask[:, :, :self.obs_len]).type(torch.float) #[29037, 4, 8]
        self.pred_date_mask = torch.from_numpy(
            seq_list_date_mask[:, :, self.obs_len:]).type(torch.float) #[29037, 4, 4]

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()

        # [(0,1), (1,2), (2,3)...]
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.ty_seqInfoZip = ty_seqInfoZip

    def __len__(self):
        return self.num_seq

    def embed_time(self,date_list):
        data_embed = []
        for date in date_list:
            year = (float(date[:4]) - 1949) / (2019 - 1949) - 0.5
            month = (float(date[4:6]) - 1) / 11.0 - 0.5
            day = (float(date[6:8]) - 1) / 30.0 - 0.5
            hour = float(date[8:10]) / 18 - 0.5
            data_embed.append([year, month, day, hour])
        return np.array(data_embed).transpose(1, 0)[np.newaxis, :, :]

    def transforms(self,img):
        all_min = img.min()
        all_max = img.max()
        img = (img-all_min)/(all_max-all_min)
        img[img>1] = 1
        img[img<0] = 0
        return img

    def img_read(self,img_path):
        try:
            img = np.load(img_path)
        except Exception as e:
            print(f"ERROR {e} loading image from path: {img_path}")
            raise RuntimeError(f"Critical error: failed to load image from {img_path}") from e
        img = cv2.resize(img,(64,64))
        img = self.transforms(img)  #对crop后的meteo进行归一化，缩放到0，1之间
        img = img.astype(np.float32)
        img = img[:,:,np.newaxis]   #[64,64,1]

        return img

    def get_meteo(self,seqInfoZip):
        tyname = seqInfoZip['lastobsName'][1]
        year = seqInfoZip['tyInfoSeqIdx'][0][2:6]
        seqdate = seqInfoZip['seqdate']
        if platform.system() == 'Windows':
            inner_root = r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Inner-data'
        else:
            inner_root = '/data/gaoliang/dataset_shared/Inner-data'
        inner_time = seqInfoZip['lastobsName'][0]
        inner_path = os.path.join(inner_root,year,tyname,inner_time+'.npy')
        inner_data = np.load(inner_path,allow_pickle=True).item()
        if platform.system() == 'Windows':
            meteo_path = {'500gph':r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\500_Geopotential_year_centercrop',
                        '500t':r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\500_Temperature_year_centercrop',
                        '500u':r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\500_U component of wind_year_centercrop',
                        '500v':r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\500_V component of wind_year_centercrop',
                        '850rh':r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\850_Relative humidity_year_centercrop',
                        '850sh': r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\850_Specific humidity_year_centercrop',
                        '850u': r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\850_U component of wind_year_centercrop',
                        '850v': r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\850_V component of wind_year_centercrop',
                        '200u': r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\200_U component of wind_year_centercrop',
                        '200v': r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\200_V component of wind_year_centercrop',
                        'wshear':r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\Wind shear',
                        'sst': r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Meteo-data\SST_year_centercrop' 
                        }
        else:
            meteo_path = {'500gph':'/data/gaoliang/dataset_shared/Meteo-data/500_Geopotential_year_centercrop',
                          '500t':'/data/gaoliang/dataset_shared/Meteo-data/500_Temperature_year_centercrop',
                          '500u':'/data/gaoliang/dataset_shared/Meteo-data/500_U component of wind_year_centercrop',
                          '500v':'/data/gaoliang/dataset_shared/Meteo-data/500_V component of wind_year_centercrop',
                          '850rh':'/data/gaoliang/dataset_shared/Meteo-data/850_Relative humidity_year_centercrop',
                          '850sh': '/data/gaoliang/dataset_shared/Meteo-data/850_Specific humidity_year_centercrop',
                          '850u': '/data/gaoliang/dataset_shared/Meteo-data/850_U component of wind_year_centercrop',
                          '850v': '/data/gaoliang/dataset_shared/Meteo-data/850_V component of wind_year_centercrop',
                          '200u': '/data/gaoliang/dataset_shared/Meteo-data/200_U component of wind_year_centercrop',
                          '200v': '/data/gaoliang/dataset_shared/Meteo-data/200_V component of wind_year_centercrop',
                          'wshear':'/data/gaoliang/dataset_shared/Meteo-data/Wind shear',
                          'sst': '/data/gaoliang/dataset_shared/Meteo-data/SST_year_centercrop'
                          }

        if self.meteo_name == 'all':
            data_dir = [os.path.join(path, year, tyname) for path in meteo_path.values()]
        elif self.meteo_name in meteo_path:
            data_dir = [os.path.join(meteo_path[self.meteo_name], year, tyname)]
        else:
            raise ValueError(f"Unknown meteo name: {self.meteo_name}")

        meteo_obs = []
        meteo_pre = []
        obs_list = seqdate[:self.obs_len]
        pre_list = seqdate[self.obs_len:]

        for obs_date in obs_list:
            meteo_obs_all = []
            for path in data_dir:
                img_path = os.path.join(path, obs_date + '.npy')
                img = self.img_read(img_path) #[64,64,1]
                meteo_obs_all.append(img)
            meteo_obs_all = np.concatenate(meteo_obs_all, axis=2) # [64,64,meteo_num]
            meteo_obs.append(meteo_obs_all) # [8,64,64,meteo_num]
        for pre_date in pre_list:
            meteo_pre_all = []
            for path in data_dir:
                img_path = os.path.join(path, pre_date + '.npy')
                img = self.img_read(img_path)
                meteo_pre_all.append(img)
            meteo_pre_all = np.concatenate(meteo_pre_all, axis=2)
            meteo_pre.append(meteo_pre_all)

        # meteo_obs = torch.tensor(np.array(meteo_obs), dtype=torch.float)
        # meteo_pre = torch.tensor(np.array(meteo_pre), dtype=torch.float)
        meteo_obs = torch.from_numpy(np.array(meteo_obs, dtype=np.float32))
        meteo_pre = torch.from_numpy(np.array(meteo_pre, dtype=np.float32))

        return {'obs': meteo_obs, 'pre': meteo_pre,'inner':inner_data}

    def __getitem__(self, index):

        start, end = self.seq_start_end[index]

        # 获取当前子序列的所有气象crop数据以及最后一个时间步的inner data
        # 返回的是字典 {'obs': meteo_obs, 'pre': meteo_pre,'inner':inner_data}
        # [8, 64, 64, 1] [4, 64, 64, 1] dict{inner_data}
        meteo = self.get_meteo(self.ty_seqInfoZip[start:end][0])
        out = [
            self.obs_traj[start:end, :],        self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],    self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],     self.loss_mask[start:end, :],
            self.obs_traj_Me[start:end,:],      self.pred_traj_Me[start:end,:],
            self.obs_traj_rel_Me[start:end, :], self.pred_traj_rel_Me[start:end, :],
            self.obs_date_mask[start:end, :],   self.pred_date_mask[start:end, :],
            meteo['obs'],meteo['pre'],meteo['inner'],
            self.ty_seqInfoZip[start:end]
        ]

        # save traj_segment if path is specified
        # if self.save_path:

        return out


if __name__ == '__main__':

    path = r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Self-data\train'
    # path = r'D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\datasets\Self-data\test2019-2023_V1'

    save_path = r'D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\AllTrajSequences\tc_trajs_train_time.npz'
    dset = TrajectoryDataset(path, obs_len=8, pred_len=4, stride=1, delim='\t', meteo='all')

    
    all_trajs = []        # 存储所有轨迹 [total_seq, 12, 4]
    all_times = []        # 存储所有轨迹对应的时间字符串 [total_seq, 12]
    tc_names = []         # 所有 TC 名称
    tc_lens = []          # 每个 TC 的轨迹数量

    tc_traj_dict = defaultdict(list)
    tc_time_dict = defaultdict(list)

    print("开始遍历数据集...")
    for i in tqdm(range(len(dset)), desc="Processing Trajectories"):
        sample = dset[i]
        obs = sample[0]               # [1, 2, 8]
        pred = sample[1]              # [1, 2, 4]
        wind_pre_obs = sample[6]      # [1, 2, 8]
        wind_pre_pred = sample[7]     # [1, 2, 4]
        info = sample[-1][0]
        tyname = info['tyInfoSeqIdx'][0]

        # 轨迹本身
        traj_pos = torch.cat([obs, pred], dim=2)           # [1, 2, 12]
        traj_wind = torch.cat([wind_pre_obs, wind_pre_pred], dim=2)  # [1, 2, 12]
        full_traj = torch.cat([traj_pos, traj_wind], dim=1)[0].T.numpy()  # [12, 4]

        # 保存原始时间字符串列表
        seq_dates = info['seqdate']  # 长度12的原始时间字符串列表

        tc_traj_dict[tyname].append(full_traj)
        tc_time_dict[tyname].append(seq_dates)

    # 合并所有轨迹，同时记录 TC 信息
    for tc, traj_list in tc_traj_dict.items():
        traj_arr = np.stack(traj_list, axis=0)  # [N, 12, 4]
        time_arr = np.array(tc_time_dict[tc])   # [N, 12]，字符串数组

        all_trajs.append(traj_arr)
        all_times.append(time_arr)
        tc_names.append(tc)
        tc_lens.append(len(traj_arr))

    # 合并为一个大矩阵
    all_trajs = np.concatenate(all_trajs, axis=0)  # [total_N, 12, 4]
    all_times = np.concatenate(all_times, axis=0)  # [total_N, 12]

    # 保存为一个 .npz 文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, 
                        tc_trajs=all_trajs, 
                        tc_times=all_times,  # 保存原始时间字符串
                        tc_names=np.array(tc_names),
                        tc_lens=np.array(tc_lens))

    print(f"保存完成：共 {len(tc_names)} 个台风，{len(all_trajs)} 段轨迹，每段含原始时间信息")
