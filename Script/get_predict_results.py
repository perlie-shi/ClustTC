import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
import cv2
from matplotlib import animation
import matplotlib.image as img
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

from scipy.spatial import ConvexHull
from matplotlib import patches
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LatentClust.data.loader import data_loader
from LatentClust.models import TrajectoryGenerator
from LatentClust.utils import relative_to_abs, get_dset_path,dic2cuda

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default=r'model_save/mgchooser_pipre5lr1e4_evn_envshare_noclip_trainall_relu_tripchkecl_gph', type=str)
# parser.add_argument('--model_path',default=r'D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\model_save\bs=96_Self-data_meteo=all_CCM\checkpoint_with_model_05200.pt', type=str)


parser.add_argument('--num_samples', default=6, type=int)
parser.add_argument('--dset_type', default='test2019-2023', type=str)

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        lstm_numlayer=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def toNE(pred_traj,pred_Me):
    # 0  经度  1纬度 ，lon lat
    pred_traj[:, :,0] = (pred_traj[:, :,0] / 10 * 500 + 1300)/10
    pred_traj[:,:,1] = (pred_traj[:,:,1] / 6 * 300 + 300)/10
    # 0 气压 1 风速
    pred_Me[:, :, 0] = pred_Me[:, :, 0]*50+960
    pred_Me[:, :, 1] = pred_Me[:, :, 1] * 25 + 40
    return pred_traj,pred_Me

plot_dic = {'tyid':[],'gt_data_all':[],'pred_traj':[],'gt_data_all_pw':[],'pred_pw':[]}
def get_plot_data(tyid, gt_data_all, pred_list, gt_data_all_pw, pred_list_Me):
    '''
    -> tyid: batch dict (12timestamps)
    -> gt_data_all:     [12,96,2]
    -> pred_list:       [4 96 2] list_len=6,6是6个生成器
    -> gt_data_all_pw:  [12,96,2]
    -> pred_list_Me:    [4 96 2] list_len=6
    '''
    plot_dic['tyid'].append(tyid)
    plot_dic['gt_data_all'].append(gt_data_all)
    plot_dic['pred_traj'].append(pred_list)
    plot_dic['gt_data_all_pw'].append(gt_data_all_pw)
    plot_dic['pred_pw'].append(pred_list_Me)
    print(tyid)

def evaluate(args, loader, generator, num_samples,modelPath,save_path,Class,flag):
    with torch.no_grad():
        plotCount = 0
        for batch in loader:
            innerdata = dic2cuda(batch[-2])
            tyID = batch[-1]
            batch = [tensor.cuda() for tensor in batch[:-2]]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me,
             pred_traj_gt_rel_Me,
             obs_date_mask, pred_date_mask, image_obs, image_pre) = batch
            gt = pred_traj_gt[:, :, :].data
            input_a = obs_traj[:, :, :].data

            pred_list,pred_list_Me = [],[]
            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            obs_traj_real, obs_traj_Me_real = toNE(obs_traj.cuda().data.cpu().numpy()[:, :, :2],obs_traj.cuda().data.cpu().numpy()[:, :, 2:])
            pred_traj_gt_real,pred_traj_gt_real_Me = toNE(pred_traj_gt.cuda().data.cpu().numpy()[:, :, :2],pred_traj_gt.cuda().data.cpu().numpy()[:, :, 2:])
            seq_traj_gt = np.concatenate((obs_traj_real, pred_traj_gt_real), axis=0)
            seq_me_gt = np.concatenate((obs_traj_Me_real, pred_traj_gt_real_Me), axis=0)
            #for _ in range(6):#num_samples
            pred_traj_fake_rel,_,_,sampled_gen_idxs = generator(
                obs_traj, obs_traj_rel, seq_start_end, image_obs,innerdata,
                num_samples=num_samples, all_g_out=False
            )
            pred_traj_fake_all = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )
            pred_traj_fake = pred_traj_fake_all[:4, :, :,:2]
            pred_traj_fake_Me = pred_traj_fake_all[:4, :, :,2:]
            time_step,num_s,batch,_ = pred_traj_fake.shape
            for num_i in range(num_s):
                pred_traj_fake_one,pred_traj_fake_Me_one = toNE(pred_traj_fake[:,num_i],pred_traj_fake_Me[:,num_i])
                pred_list.append(pred_traj_fake_one.data.cpu().numpy())
                pred_list_Me.append(pred_traj_fake_Me_one.data.cpu().numpy())
            get_plot_data(tyID, seq_traj_gt, pred_list, seq_me_gt, pred_list_Me)
        np.save(os.path.join(save_path, f'{flag}_prediction_results.npy'), plot_dic)
        # 26个batch batchsize=96

def main(args):
    # sava_path = args.model_path
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:

        if 'no_' in path or 'pt' not in path:
            continue
        modelpath = path
        checkpoint = torch.load(modelpath)
        print(checkpoint['args'])
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        _args.batch_size = 8   # 或者 8、16，任意小尺寸

        # path = get_dset_path(_args.dataset_name, args.dset_type)

        #分类专用测试路径
        Class = '6'
        # path = rf"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\datasets\Self-data\Norm_class\{Class}"
        # save_path = rf"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\model_save\bs=96_Self-data_meteo=all_CCM\{Class}"

        flag = 'offshore'#'RI'  #'offshore'  #'nearcoast'
        path = rf"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\datasets\Self-data\test2019-2023_V1"
        save_path = rf"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\model_save\bs=96_Self-data_meteo=all_CCM\{flag}"
        coastline_path = r"D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\ne_10m_coastline\ne_10m_coastline.shp"
          
        _, loader = data_loader( #针对性取出哪些轨迹
            _args,
            path,
            test=True,
            only_RI=True if flag=='RI' else False,
            near_coast=True if flag=='nearcoast' else False,
            offshore=True if flag=='offshore' else False,
            coastline_path=coastline_path,
            coast_threshold_km=150
        )
        evaluate(_args, loader, generator, args.num_samples,modelpath,save_path,Class,flag)



def seed_torch():
    seed = 1024 # 用户设定
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
