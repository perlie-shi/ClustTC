import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy.spatial import cKDTree
import shapefile  # pyshp
from LatentClust.data.datasetConstruction import TrajectoryDataset, seq_collate

def select_RI_indices(dataset, obs_len=8, pred_len=4):
    """
    根据快速增强条件选择轨迹段索引：
        24h风速增加≥15 m/s 或 12h风速增加≥10 m/s
    风速需反归一化: wind_real = wind_norm*25+40
    """
    ri_indices = []

    for idx in range(len(dataset)):
        start, end = dataset.seq_start_end[idx]
        
        # 风速在 obs_traj_Me / pred_traj_Me 的第二个通道
        obs_wind_norm = dataset.obs_traj_Me[start:end, 1, :]  # [num_peds, obs_len]
        pred_wind_norm = dataset.pred_traj_Me[start:end, 1, :]  # [num_peds, pred_len]

        # 拼接完整轨迹风速并反归一化
        full_wind_norm = torch.cat([obs_wind_norm, pred_wind_norm], dim=1)
        full_wind_real = full_wind_norm * 25 + 40  # 反归一化

        # 遍历该轨迹段的每条轨迹
        for wind_seq in full_wind_real:
            # 遍历轨迹序列检查增幅
            seq_len = wind_seq.shape[0]
            for i in range(seq_len):
                for j in range(i+1, seq_len):
                    dt = j - i  # 时间步间隔
                    dv = wind_seq[j] - wind_seq[i]  # 风速差
                    # 12h = 2步，24h = 4步
                    if (dt >= 2 and dv >= 10) or (dt >= 4 and dv >= 15):
                        ri_indices.append(idx)
                        break  # 当前轨迹段满足条件即可
                else:
                    continue
                break

    return sorted(list(set(ri_indices)))

def haversine(lat1, lon1, lat2, lon2):
    """计算球面两点间的距离，单位 km"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def classify_coastal_trajectories_no_gpd(
    tc_trajs: np.ndarray,
    coastline_path: str,
    threshold_km: float = 150,
    sample_step: int = 50
    ):
    """
    使用 KDTree + Haversine 分类轨迹为近岸或远洋，不依赖 geopandas
    tc_trajs: [N, seq_len, 2] (lon, lat)
    """
    # 1. 读取海岸线 shapefile
    sf = shapefile.Reader(coastline_path)
    coast_points = []
    for shape in sf.shapes():
        coords = np.array(shape.points)
        coords = coords[::sample_step]  # 采样降低计算量
        coast_points.append(coords)
    coast_points = np.vstack(coast_points)  # [M, 2] -> (lon, lat)
    

    # 反归一化
    lons_deg = (tc_trajs[:, :, 0]*50+1300)
    lats_deg = (tc_trajs[:, :, 1]*50+300)
    
    lons = (lons_deg*0.1).ravel()
    lats = (lats_deg*0.1).ravel()
    # 2. KDTree 近邻查找
    tree = cKDTree(coast_points)
    traj_points = np.column_stack([lons, lats])
    _, idx = tree.query(traj_points, k=1)
    nearest_coast = coast_points[idx]

    # 3. Haversine 计算精确距离
    distances_km = haversine(traj_points[:,1], traj_points[:,0],
                             nearest_coast[:,1], nearest_coast[:,0])
    distances_km = distances_km.reshape(tc_trajs.shape[0], tc_trajs.shape[1])

    # 4. 每条轨迹最小距离
    traj_min_distances_km = distances_km.min(axis=1)

    # 5. 分类
    is_coastal_traj = traj_min_distances_km <= threshold_km
    coastal_ids = np.where(is_coastal_traj)[0]
    offshore_ids = np.where(~is_coastal_traj)[0]

    return coastal_ids, offshore_ids, traj_min_distances_km

def data_loader(args, path, test=None, only_RI=False,
                near_coast=False, offshore=False,
                coastline_path=None, coast_threshold_km=150):
    """
    通用数据加载器：
      - only_RI: 是否仅快速增强轨迹
      - near_coast/offshore: 是否筛选近岸或远洋
    """
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
        delim=args.delim,
        meteo=args.meteo
    )

    if test is None: # train
        shuffle = True
    else:           # test
        shuffle = False
        args.loader_num_workers = 0

    indices = np.arange(len(dset))  # 初始保留所有轨迹段

    # 快速增强筛选
    if only_RI:
        ri_indices = select_RI_indices(dset, obs_len=args.obs_len, pred_len=args.pred_len)
        indices = np.intersect1d(indices, ri_indices)
        print(f"[INFO] RI轨迹段数: {len(indices)}")

    # 近岸/远洋筛选
    if near_coast or offshore:
        # 提取所有轨迹段的 [lon, lat]
        traj_lonlat = []
        for idx in indices:
            start, end = dset.seq_start_end[idx]
            obs = dset.obs_traj[start:end]    # [num_peds, 2, obs_len]
            pred = dset.pred_traj[start:end]  # [num_peds, 2, pred_len]
            full = torch.cat([obs, pred], dim=2)[0, :, :].T  # [seq_len, 2] (lon, lat)
            traj_lonlat.append(full.numpy())
        traj_lonlat = np.stack(traj_lonlat, axis=0)  # [N, seq_len, 2]

        coastal_ids, offshore_ids, _ = classify_coastal_trajectories_no_gpd(
            traj_lonlat, coastline_path, threshold_km=coast_threshold_km
        )

        if near_coast:
            indices = indices[coastal_ids]
            print(f"[INFO] 近岸轨迹段数: {len(indices)}")
        elif offshore:
            indices = indices[offshore_ids]
            print(f"[INFO] 远洋轨迹段数: {len(indices)}")

    # 构造 Subset
    dset = Subset(dset, indices.tolist())

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate
    )
    return dset, loader


