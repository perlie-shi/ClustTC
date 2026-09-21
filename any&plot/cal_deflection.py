# -*- coding: utf-8 -*-
"""
登陆前后偏折角对比分析（修正版）
- txt文件中的经纬度已经是归一化后的值
- 只需乘以0.1反归一化为度数
- 找到A和B的max相差最大的轨迹并绘制
"""

import os, math, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shapefile
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.unicode_minus'] = False

try:
    from scipy.spatial import cKDTree as KDTree
    _have_kdtree = True
except Exception:
    _have_kdtree = False

def _fmt_lon(lon):
    # 经度：E 正、W 负
    hemi = 'E' if lon >= 0 else 'W'
    return f"{abs(lon):.1f}°{hemi}"

def _fmt_lat(lat):
    # 纬度：N 正、S 负
    hemi = 'N' if lat >= 0 else 'S'
    return f"{abs(lat):.1f}°{hemi}"


# ====================== CONFIG ======================
RAW_TC_DIR = r"D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\Self-data\1950-2025"
COAST_SHP  = r"D:\MacauPrograms\TrajPrediction\MGTCF\dataset_shared\ne_10m_coastline\ne_10m_coastline.shp"

SAVE_DIR = r"D:\MacauPrograms\TrajPrediction\MGTCF\my_method\Picture of papers\tune"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PREFIX = "fixedwin_from_eachfile_A48p48_vs_B144p48"

# 时间窗口配置
STEP_HOURS = 6
A_LEFT_H, A_RIGHT_H = 48, 48
B_LEFT_H, B_RIGHT_H = 144, 48
LAND_THRESHOLD_KM = 30.0

# 平滑参数
SMOOTH_TRAJ = True
SMOOTH_WIN = 3

BOOT_B = 2000
# ====================================================


def denormalize_to_deg(coords_norm):
    """
    反归一化：txt中的归一化值直接 *0.1 转为度数
    输入: [N, 2] 归一化值
    输出: [N, 2] 度数
    """
    return coords_norm * 0.1


def parse_each_txt_as_storm(raw_dir):
    """
    读取每个txt文件为一条TC轨迹
    txt中已经是归一化值，直接读取
    返回: dict {filename: np.array([[lon_norm, lat_norm], ...])}
    """
    storm_tracks = {}
    files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if len(files) == 0:
        raise RuntimeError(f"No txt files in {raw_dir}")
    
    for fp in tqdm(files, desc="Reading TC files"):
        coords = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    lat_norm = float(parts[2])  # 已经是归一化值
                    lon_norm = float(parts[3])  # 已经是归一化值
                except Exception:
                    continue
                
                coords.append([lon_norm, lat_norm])
        
        coords = np.array(coords, dtype=float)
        
        # 去除相邻重复点
        if coords.shape[0] > 1:
            keep = [0]
            for i in range(1, coords.shape[0]):
                if not np.allclose(coords[i], coords[i-1]):
                    keep.append(i)
            coords = coords[keep]
        
        fname = os.path.splitext(os.path.basename(fp))[0]
        storm_tracks[fname] = coords
    
    return storm_tracks


def moving_average_1d(x, k=3):
    if k <= 1:
        return x
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    kern = np.ones(k) / k
    return np.convolve(xpad, kern, mode='valid')


def smooth_traj(traj, k=3):
    if (not SMOOTH_TRAJ) or traj.shape[0] < k:
        return traj
    lon_s = moving_average_1d(traj[:, 0], k)
    lat_s = moving_average_1d(traj[:, 1], k)
    return np.stack([lon_s, lat_s], axis=1)


def initial_bearing_deg(lon1, lat1, lon2, lat2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return np.degrees(np.arctan2(y, x))


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def deflection_series_deg(traj_deg):
    """输入度数坐标，输出偏折角序列"""
    if traj_deg.shape[0] < 3:
        return np.array([])
    
    lons = traj_deg[:, 0]
    lats = traj_deg[:, 1]
    
    bearings = [initial_bearing_deg(lons[i], lats[i], lons[i+1], lats[i+1])
                for i in range(len(lons) - 1)]
    bearings = np.asarray(bearings)
    
    dtheta = np.diff(bearings)
    dtheta = (dtheta + 180.0) % 360.0 - 180.0
    return dtheta


def compute_metrics_from_angles(dtheta):
    a = np.abs(dtheta)
    if a.size == 0:
        return None
    return (float(np.mean(a)), float(np.max(a)), 
            float(np.percentile(a, 95)), float(np.sum(a)))


def compute_metrics_for_window(traj_norm):
    """
    输入归一化坐标，反归一化后计算指标
    返回: (mean, max, p95, total) 或 None
    """
    if traj_norm is None or traj_norm.shape[0] < 3:
        return None
    
    # 反归一化为度数
    traj_deg = denormalize_to_deg(traj_norm)
    
    # 平滑
    traj_smooth = smooth_traj(traj_deg, SMOOTH_WIN)
    
    # 计算偏折角
    dtheta = deflection_series_deg(traj_smooth)
    if dtheta is None or dtheta.size == 0:
        return None
    
    return compute_metrics_from_angles(dtheta)


# ---------- 海岸线与登陆点判定 ----------
def load_coast_points_deg(shp_path):
    sf = shapefile.Reader(shp_path)
    pts = []
    for s in sf.shapes():
        pts.extend(s.points)
    return np.array(pts, dtype=float)


def build_coast_kdtree(coast_pts_deg):
    pts_rad = np.radians(coast_pts_deg)
    if _have_kdtree:
        tree = KDTree(pts_rad)
        return tree, pts_rad
    else:
        return None, pts_rad


def nearest_coast_distance_km(lon_deg, lat_deg, tree_and_pts):
    tree, pts_rad = tree_and_pts
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)
    
    if tree is not None:
        dist_rad, idx = tree.query([lon_rad, lat_rad])
        lon2, lat2 = pts_rad[idx, 0], pts_rad[idx, 1]
        return haversine_km(lon_deg, lat_deg, np.degrees(lon2), np.degrees(lat2))
    else:
        dlats = pts_rad[:, 1] - lat_rad
        dlons = pts_rad[:, 0] - lon_rad
        a = np.sin(dlats/2.0)**2 + np.cos(lat_rad) * np.cos(pts_rad[:, 1]) * np.sin(dlons/2.0)**2
        d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        return float(np.min(d))


def find_landfall_index(track_norm, tree_and_pts, threshold_km=30.0):
    if track_norm.shape[0] == 0:
        return None
    
    track_deg = denormalize_to_deg(track_norm)
    for i in range(track_deg.shape[0]):
        dkm = nearest_coast_distance_km(track_deg[i, 0], track_deg[i, 1], tree_and_pts)
        if dkm <= threshold_km:
            return i
    return None


def hours_to_steps(h, step_hours):
    return int(round(h / step_hours))


def slice_fixed_windows(track, land_idx, step_hours, A_left, A_right, B_left, B_right):
    t0 = land_idx
    A_L = max(0, t0 - hours_to_steps(A_left, step_hours))
    A_R = min(track.shape[0], t0 + hours_to_steps(A_right, step_hours) + 1)
    B_L = max(0, t0 - hours_to_steps(B_left, step_hours))
    B_R = max(0, t0 - hours_to_steps(B_right, step_hours) + 1)
    
    A = track[A_L:A_R] if (A_R - A_L) >= 3 else None
    B = track[B_L:B_R] if (B_R - B_L) >= 3 else None
    return A, B


def find_max_diff_track(storm_tracks, tree_and_pts):
    """
    只找 Δ = A_MaxTurn - B_MaxTurn > 0 的样本，并最大化 Δ。
    A: [t0-48h, t0+48h]（含两端）  B: [t0-144h, t0-48h]（含两端）
    其它保持与你的主程序兼容。
    """
    max_delta = -np.inf
    best_rec = None
    skipped_counts = {
        "short_track": 0, "no_landfall": 0, "A_win_bad": 0, "B_win_bad": 0, 
        "metric_none": 0, "A_not_greater": 0
    }

    for name, track in tqdm(storm_tracks.items(), desc="Finding A>B max-diff track"):
        if track.shape[0] < 3:
            skipped_counts["short_track"] += 1
            continue

        t0 = find_landfall_index(track, tree_and_pts, LAND_THRESHOLD_KM)
        if t0 is None:
            skipped_counts["no_landfall"] += 1
            continue

        A_win, B_win = slice_fixed_windows(track, t0, STEP_HOURS,
                                           A_LEFT_H, A_RIGHT_H, B_LEFT_H, B_RIGHT_H)
        if A_win is None:
            skipped_counts["A_win_bad"] += 1
            continue
        if B_win is None:
            skipped_counts["B_win_bad"] += 1
            continue

        mA = compute_metrics_for_window(A_win)
        mB = compute_metrics_for_window(B_win)
        if (mA is None) or (mB is None):
            skipped_counts["metric_none"] += 1
            continue

        A_max, B_max = mA[1], mB[1]
        delta = A_max - B_max
        if not (delta > 0):
            skipped_counts["A_not_greater"] += 1
            continue

        if delta > max_delta:
            max_delta = delta
            best_rec = {
                "name": name, "track": track, "t0": t0,
                "A_win": A_win, "B_win": B_win,
                "A_metrics": mA, "B_metrics": mB,
                "delta": float(delta)
            }

    if best_rec is None:
        print("[WARN] 未找到符合 Δ=A_max-B_max>0 的轨迹")
        print("[DEBUG] 跳过统计：", skipped_counts)
    return best_rec

def find_top_diff_tracks(storm_tracks, tree_and_pts, topN=20):
    """
    找到 Δ = A_MaxTurn - B_MaxTurn 最大的前 N 条轨迹（只保留 Δ>0）
    返回: list[dict]，每个元素和 find_max_diff_track 的 best_rec 格式相同
    """
    results = []

    for name, track in tqdm(storm_tracks.items(), desc="Finding top diff tracks"):
        if track.shape[0] < 3:
            continue

        t0 = find_landfall_index(track, tree_and_pts, LAND_THRESHOLD_KM)
        if t0 is None:
            continue

        A_win, B_win = slice_fixed_windows(track, t0, STEP_HOURS,
                                           A_LEFT_H, A_RIGHT_H, B_LEFT_H, B_RIGHT_H)
        if A_win is None or B_win is None:
            continue

        mA = compute_metrics_for_window(A_win)
        mB = compute_metrics_for_window(B_win)
        if (mA is None) or (mB is None):
            continue

        A_max, B_max = mA[1], mB[1]
        delta = A_max - B_max
        if delta <= 0:
            continue

        results.append({
            "name": name, "track": track, "t0": t0,
            "A_win": A_win, "B_win": B_win,
            "A_metrics": mA, "B_metrics": mB,
            "delta": float(delta)
        })

    # 排序取前 topN
    results.sort(key=lambda r: r["delta"], reverse=True)
    return results[:topN]



def load_coastline_shapes(shp_path):
    sf = shapefile.Reader(shp_path)
    return sf.shapes()


def plot_track_with_windows(rec, coast_shapes, save_dir, save_prefix, margin_deg=5.0):
    """
    只绘制轨迹周边的海岸线分段，并把坐标轴范围限定在
    [轨迹经纬度最小/最大 ± margin_deg]；这样刻度自然“对味儿”。
    """
    # 反归一化
    track_deg = denormalize_to_deg(rec['track'])
    A_deg = denormalize_to_deg(rec['A_win'])
    B_deg = denormalize_to_deg(rec['B_win'])

    # 轨迹包络框 + 外扩
    lon_min = np.min(track_deg[:,0]) - margin_deg
    lon_max = np.max(track_deg[:,0]) + margin_deg
    lat_min = np.min(track_deg[:,1]) - margin_deg
    lat_max = np.max(track_deg[:,1]) + margin_deg

    def in_bbox(lon, lat):
        return (lon_min <= lon) & (lon <= lon_max) & (lat_min <= lat) & (lat <= lat_max)

    plt.figure(figsize=(8, 6))

    # 1) 海岸线（仅画落在 bbox 内的点段）
    for s in coast_shapes:
        pts = np.asarray(s.points, float)
        if pts.size == 0:
            continue
        mask = in_bbox(pts[:,0], pts[:,1])
        if not np.any(mask):
            continue
        # 为了保留线段连贯性，取 mask 的连通段
        idx = np.where(mask)[0]
        # 简单做法：直接把 bbox 内的点画出来（已经足够表达海岸）
        plt.plot(pts[mask,0], pts[mask,1], color='0.7', linewidth=0.6, zorder=0)

    # 2) 全轨迹
    plt.plot(track_deg[:,0], track_deg[:,1], color='lightgray', linewidth=2.5, 
             label='Full track', zorder=1)

    # 3) B 窗口
    plt.plot(B_deg[:,0], B_deg[:,1], color='#d62728', linewidth=3.5, 
             label='Pre-Landfall (B)', zorder=2)

    # 4) A 窗口
    plt.plot(A_deg[:,0], A_deg[:,1], color='#1f77b4', linewidth=3.5, 
             label='Near-Landfall (A)', zorder=3)

    # 5) 登陆点
    t0 = rec['t0']
    plt.scatter(track_deg[t0,0], track_deg[t0,1],
                color='black', marker='x', s=140, linewidths=3.5, 
                label='Landfall', zorder=4)

    # 6) MaxTurn 位置标注（和你原逻辑一致）
    def find_maxturn_idx(traj_deg):
        if traj_deg.shape[0] < 3:
            return None
        dtheta = np.abs(deflection_series_deg(traj_deg))
        if dtheta.size == 0:
            return None
        return int(np.argmax(dtheta)) + 1

    idxA = find_maxturn_idx(A_deg)
    idxB = find_maxturn_idx(B_deg)
    if idxA is not None and 0 <= idxA < A_deg.shape[0]:
        plt.scatter(A_deg[idxA,0], A_deg[idxA,1],
                    edgecolors='#1f77b4', facecolors='none', s=180, linewidths=2.5,
                    label=f"MaxTurn(A)={rec['A_metrics'][1]:.1f}°", zorder=5)
    if idxB is not None and 0 <= idxB < B_deg.shape[0]:
        plt.scatter(B_deg[idxB,0], B_deg[idxB,1],
                    edgecolors='#d62728', facecolors='none', s=180, linewidths=2.5,
                    label=f"MaxTurn(B)={rec['B_metrics'][1]:.1f}°", zorder=5)

    # 坐标轴与风格
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    # 不再使用 axis('equal')，让坐标轴根据范围自适应；若坚持等比例，可把下面一行打开：
    # plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("Longitude (°)", fontsize=28, fontname='Arial')
    plt.ylabel("Latitude (°)",  fontsize=28, fontname='Arial')
    plt.title(f"{rec['name'][9:]}",
              fontsize=28, fontweight='bold', fontname='Arial')
    plt.legend(prop={'family':'Arial', 'size': 28}, loc='upper right')

    # ===== 只保留 MaxTurn(A) / MaxTurn(B) 的 legend =====
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    keep_handles = []
    keep_labels  = []
    for h, l in zip(handles, labels):
        if l.startswith("MaxTurn(A)") or l.startswith("MaxTurn(B)"):
            keep_handles.append(h)
            keep_labels.append(l)
    ax.legend(
        keep_handles,
        keep_labels,
        prop={'family': 'Arial', 'size': 28},
        loc='upper right'
    )
    ax.tick_params(
    axis='both',
    which='major',
    labelsize=24,   # 主刻度字号
    length=6,
    width=1.5
)


    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: _fmt_lon(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: _fmt_lat(v)))

    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{save_prefix}_max_diff_{rec['name']}.png")
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] 轨迹图保存：{out_path}")
    plt.show()



def collect_A_B_metrics(storm_tracks, tree_and_pts):
    """收集所有A/B窗口的指标"""
    A_list, B_list = [], []
    stats = {
        "total_storms": 0,
        "no_landfall": 0,
        "A_computed": 0,
        "B_computed": 0,
    }
    
    for name, track in tqdm(storm_tracks.items(), desc="Processing storms"):
        stats["total_storms"] += 1
        if track.shape[0] < 3:
            stats["no_landfall"] += 1
            continue
        
        li = find_landfall_index(track, tree_and_pts, LAND_THRESHOLD_KM)
        if li is None:
            stats["no_landfall"] += 1
            continue
        
        A_win, B_win = slice_fixed_windows(track, li, STEP_HOURS, 
                                          A_LEFT_H, A_RIGHT_H, B_LEFT_H, B_RIGHT_H)
        
        if A_win is not None:
            ma = compute_metrics_for_window(A_win)
            if ma is not None:
                A_list.append(ma)
                stats["A_computed"] += 1
        
        if B_win is not None:
            mb = compute_metrics_for_window(B_win)
            if mb is not None:
                B_list.append(mb)
                stats["B_computed"] += 1
    
    A_arr = np.array(A_list, float) if A_list else np.empty((0, 4))
    B_arr = np.array(B_list, float) if B_list else np.empty((0, 4))
    return A_arr, B_arr, stats


def simple_kde(x, grid, h=None):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.zeros_like(grid)
    if h is None:
        std = np.std(x)
        n = len(x)
        h = 1.06 * std * (n ** (-1/5)) if std > 0 else max(1.0, (grid[1]-grid[0])*3)
    diff2 = (grid[:, None] - x[None, :])**2
    kern = np.exp(-0.5 * diff2 / (h**2)) / (math.sqrt(2*math.pi) * h)
    return np.mean(kern, axis=1)


def ecdf(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    return x, np.arange(1, n+1) / n


def bootstrap_diff(a, b, stat="median", B=BOOT_B, random_state=123):
    rng = np.random.default_rng(random_state)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    f = np.mean if stat == "mean" else np.median
    obs = f(a) - f(b)
    
    Ab = rng.choice(a, size=(B, a.size), replace=True)
    Bb = rng.choice(b, size=(B, b.size), replace=True)
    boot = f(Ab, axis=1) - f(Bb, axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    
    pool = np.concatenate([a, b])
    na = a.size
    cnt = 0
    for _ in range(B):
        rng.shuffle(pool)
        aa = pool[:na]
        bb = pool[na:]
        if abs(f(aa) - f(bb)) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (B + 1)
    
    return float(obs), (float(ci_low), float(ci_high)), float(p)


def describe(name, x):
    x = np.asarray(x, float)
    if x.size == 0:
        return f"{name}: N=0"
    return (f"{name}: N={x.size}, mean={x.mean():.2f}°, "
            f"median={np.median(x):.2f}°, P95={np.percentile(x,95):.2f}°")


def plot_max_total_distributions(A, B, save_dir, save_prefix):
    """保持原有的绘图风格"""
    metric_names = ["Maximum Turning Angle (°)", "Total Curvature (°)"]
    col_index = [1, 3]
    x_max_hint = [80, 250]
    bins_hint = [np.linspace(0, x_max_hint[i], 31) for i in range(2)]
    
    plt.figure(figsize=(10, 9))
    for j in range(2):
        i = col_index[j]
        plt.subplot(1, 2, j+1)
        bins = bins_hint[j]
        
        if A.size:
            plt.hist(A[:, i], bins=bins, density=True, alpha=0.45, label="Near-Landfall Period")
            grid = np.linspace(0, bins[-1], 801)
            kde = simple_kde(np.clip(A[:, i], 0, bins[-1]), grid)
            plt.plot(grid, kde, linewidth=2, color='C0')
        
        if B.size:
            plt.hist(B[:, i], bins=bins, density=True, alpha=0.45, label="Pre-Landfall History")
            grid = np.linspace(0, bins[-1], 801)
            kde = simple_kde(np.clip(B[:, i], 0, bins[-1]), grid)
            plt.plot(grid, kde, linewidth=2, color='C1')
        
        plt.xlabel(metric_names[j], fontsize=20, fontname='Arial')
        plt.ylabel("Probability Density", fontsize=20, fontname='Arial')
        plt.title(metric_names[j] + "Distribution", fontsize=20, fontname='Arial')
        plt.xticks(fontsize=14, fontname='Arial')
        plt.yticks(fontsize=14, fontname='Arial')
        plt.legend(prop={'family': 'Arial', 'size': 16})
    
    # plt.suptitle("Turning Angle Distribution: Near-Landfall vs Pre-Landfall Periods",
    #              fontsize=20, fontweight='bold', fontname='Arial')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = os.path.join(save_dir, f"{save_prefix}_max_total_pdf.png")
    plt.savefig(out_pdf, dpi=300)
    print(f"[INFO] PDF 图保存：{out_pdf}")
    plt.show()


def plot_max_total_ecdf(A, B, save_dir, save_prefix):
    """保持原有的绘图风格"""
    metric_names = ["MaxTurn (°)", "TotalCurvature (°)"]
    col_index = [1, 3]
    x_max_hint = [80, 250]
    
    plt.figure(figsize=(8, 6))
    for j in range(2):
        i = col_index[j]
        plt.subplot(1, 2, j+1)
        if A.size:
            x1, y1 = ecdf(np.clip(A[:, i], 0, x_max_hint[j]))
            plt.plot(x1, y1, label="Near-Landfall Period")
        if B.size:
            x2, y2 = ecdf(np.clip(B[:, i], 0, x_max_hint[j]))
            plt.plot(x2, y2, label="Pre-Landfall History")
        
        plt.xlabel(metric_names[j], fontsize=12, fontname='Arial')
        plt.ylabel("ECDF", fontsize=12, fontname='Arial')
        plt.title(metric_names[j] + " ECDF", fontsize=12, fontname='Arial')
        plt.xticks(fontsize=9, fontname='Arial')
        plt.yticks(fontsize=9, fontname='Arial')
        if j == 0:
            plt.legend(fontsize=9, prop={'family': 'Arial'})
    
    plt.suptitle("Turning Angle ECDF: Near-Landfall vs Pre-Landfall Periods",
                 fontsize=16, fontweight='bold', fontname='Arial')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_ecdf = os.path.join(save_dir, f"{save_prefix}_max_total_ecdf.png")
    plt.savefig(out_ecdf, dpi=300)
    print(f"[INFO] ECDF 图保存：{out_ecdf}")
    plt.show()


# ========================== 主流程 ==========================
def main():
    print("[INFO] 读取TC轨迹文件...")
    storm_tracks = parse_each_txt_as_storm(RAW_TC_DIR)
    print(f"[INFO] 共读取 {len(storm_tracks)} 条轨迹")
    
    print(f"[INFO] 加载海岸线...")
    coast_pts = load_coast_points_deg(COAST_SHP)
    tree = KDTree(np.radians(coast_pts)) if _have_kdtree else None
    tree_and_pts = (tree, np.radians(coast_pts))
    
    print("[INFO] 计算登陆前后窗口指标...")
    A, B, stats = collect_A_B_metrics(storm_tracks, tree_and_pts)
    
    print("\n[统计信息]")
    print(f"  总轨迹数: {stats['total_storms']}")
    print(f"  未识别登陆: {stats['no_landfall']}")
    print(f"  Near-Landfall窗口有效样本: {stats['A_computed']}")
    print(f"  Pre-Landfall窗口有效样本: {stats['B_computed']}")
    
    metric_names = ["MeanAbsTurn", "MaxTurn", "P95Turn", "TotalCurvature"]
    
    print("\n[Near-Landfall Period (t0-48h ~ t0+48h)]")
    for i, nm in enumerate(metric_names):
        print("  " + describe(nm, A[:, i] if A.size else []))
    
    print("\n[Pre-Landfall Period (t0-144h ~ t0-48h)]")
    for i, nm in enumerate(metric_names):
        print("  " + describe(nm, B[:, i] if B.size else []))
    
    # 绘制分布图
    if A.size and B.size:
        plot_max_total_distributions(A, B, SAVE_DIR, SAVE_PREFIX)
        # plot_max_total_ecdf(A, B, SAVE_DIR, SAVE_PREFIX)
        
        # 显著性检验
        print("\n[显著性检验] Δmedian = Near-Landfall - Pre-Landfall")
        for i, nm in enumerate(metric_names):
            diff, ci, p = bootstrap_diff(A[:, i], B[:, i], stat="median", 
                                        B=BOOT_B, random_state=123)
            sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
            print(f"  {nm}: Δ={diff:.2f}°, 95%CI[{ci[0]:.2f}, {ci[1]:.2f}], p={p:.4f}{sig}")
    

    # 找到并绘制MaxTurn差值最大的轨迹
    # print("\n[INFO] 寻找MaxTurn差值最大的轨迹...")
    # best_rec = find_max_diff_track(storm_tracks, tree_and_pts)
    
    # if best_rec:
    #     print(f"[INFO] 找到最大差值轨迹: {best_rec['name']}")
    #     print(f"       A_MaxTurn = {best_rec['A_metrics'][1]:.2f}°")
    #     print(f"       B_MaxTurn = {best_rec['B_metrics'][1]:.2f}°")
    #     print(f"       差值(Δ=A-B) = {best_rec['delta']:.2f}°")
    #     coast_shapes = load_coastline_shapes(COAST_SHP)
    #     plot_track_with_windows(best_rec, coast_shapes, SAVE_DIR, SAVE_PREFIX)
    # else:
    #     print("[WARN] 未找到符合条件的轨迹")
    

    # # 找到并绘制前 20 个 A>B 差值最大的轨迹
    # "Manny", "Sarah", "Tilda"
    print("\n[INFO] 寻找前 20 个 A>B 差值最大的轨迹...")
    top_recs = find_top_diff_tracks(storm_tracks, tree_and_pts, topN=20)

    if not top_recs:
        print("[WARN] 没有找到符合条件的轨迹")
    else:
        coast_shapes = load_coastline_shapes(COAST_SHP)
        for i, rec in enumerate(top_recs, 1):
            print(f"Top{i:02d}: {rec['name']} Δ={rec['delta']:.2f}° "
                f"(A={rec['A_metrics'][1]:.2f}°, B={rec['B_metrics'][1]:.2f}°)")
            if rec['name'][9:] in ["Manny", "Sarah", "Tilda"]:
                plot_track_with_windows(rec, coast_shapes, SAVE_DIR, f"{SAVE_PREFIX}_Top{i:02d}")



    print("\n[完成]")


if __name__ == "__main__":
    main()