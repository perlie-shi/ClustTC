'''
画气象变量之间的相似矩阵
并计算ICS和IDS
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd 
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# ==================== 导入你的 Loader ====================
from LatentClust.data.datasetConstruction import TrajectoryDataset, seq_collate
from LatentClust.data.loader import data_loader

# ==================== 导入你的模型 =========================
from LatentClust.models import TrajectoryGenerator
from attrdict import AttrDict
from LatentClust.utils import dic2cuda
import seaborn as sns
from matplotlib import gridspec
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
plt.rcParams['font.family'] = 'Arial'

# =============== 你需要改的部分 =====================
MODEL_PATH = r"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\model_save\bs=96_Self-data_meteo=all_CCM\checkpoint_with_model_05200.pt"
TEST_DATA_PATH = r"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\datasets\Self-data\test2017-2018"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 按 evaluation.py 的方式构造 generator ====================
def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        meteo_num=args.meteo_num,
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
        batch_norm=args.batch_norm,
    )
    generator.load_state_dict(checkpoint['g_state'])
    generator.to(DEVICE)
    generator.eval()
    return generator, args

# ================================================================

def visualize_channel_cloud(feats, channel_names, save_name=r"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\Picture of papers\tsne_channel_cloud.png"):
    """
    feats: [B, V, D]   # sample-level channel embeddings
    channel_names: (V,) list
    Only visualize the channel cloud distribution (按通道分布).
    """
    B, V, D = feats.shape

    # ---------- 1) 展开成 sample-level embedding ----------
    # 每个通道有 B 个点 → 按通道着色
    F = feats.cpu().numpy().reshape(B * V, D)   # [B*V, D]
    labels = np.repeat(np.arange(V), B)

    # ---------- 2) t-SNE ----------
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=2000,
        random_state=42
    )
    Z = tsne.fit_transform(F)    # [B*V, 2]

    # ---------- 3) 绘图 ----------
    plt.figure(figsize=(9, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, V))

    for v in range(V):
        idx = (labels == v)
        plt.scatter(
            Z[idx, 0], Z[idx, 1],
            s=60, alpha=0.55,
            color=colors[v],
            label=channel_names[v]
        )

    plt.title("t-SNE of Meteorology Embeddings", fontsize=20, fontweight='bold')
    plt.xlabel("t-SNE 1", fontsize=15)
    plt.ylabel("t-SNE 2", fontsize=15)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()
    print(f"🎉 Saved channel cloud figure: {save_name}")


#=========================画CCM聚类的图==========================
def extract_ccm_feats():
    # 1) 加载 checkpoint & generator
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    generator, ckpt_args = get_generator(checkpoint)

    # 2) 构建测试集 DataLoader
    print("⏳ Loading test dataset from:", TEST_DATA_PATH)
    _, loader_test = data_loader(
        ckpt_args,
        TEST_DATA_PATH,
        test=True,
        only_RI=False,
        near_coast=False,
        offshore=False,
        coast_threshold_km=150
    )
    # ============================================================
    #  只取前两个 batch，提取 CCM feats
    # ============================================================
    all_before = []
    all_after = []    
    all_assign =[]
    max_batches = 6   # 只要前 2 个 batch

    for batch_i, batch in enumerate(loader_test):
        if batch_i >= max_batches:
            break

        print(f"➡ Processing batch {batch_i+1}/{max_batches}")

        # === 解析 batch ===
        env_data = dic2cuda(batch[-2])
        info = batch[-1]

        batch_tensors = [tensor.to(DEVICE) for tensor in batch[:-2]]
        (
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            non_linear_ped, loss_mask, seq_start_end,
            obs_Me, pred_Me_gt, obs_Me_rel, pred_Me_gt_rel,
            obs_date_mask, pred_date_mask, image_obs, image_pre
        ) = batch_tensors

        # === 拼接输入 ===
        obs = torch.cat([obs_traj, obs_Me], dim=2)
        obs_rel = torch.cat([obs_traj_rel, obs_Me_rel], dim=2)

        with torch.no_grad():
            _pred_fake_rel, _, _, _ = generator(
                obs, obs_rel, seq_start_end,
                image_obs, env_data,
                num_samples=ckpt_args.num_samples if 'num_samples' in ckpt_args else 1,
                all_g_out=False
            )

            # feats = generator.self_fuse_meteo.ccm.feats_cache   # [B, V, D]
            feats_before = generator.self_fuse_meteo.ccm.feats_before
            feats_after  = generator.self_fuse_meteo.ccm.feats_after
            assign_prob  = generator.self_fuse_meteo.ccm.assign_prob_cache

            all_before.append(feats_before.cpu().numpy())
            all_after.append(feats_after.cpu().numpy())
            all_assign.append(assign_prob.cpu().numpy())

    # all_feats = np.concatenate(all_feats, axis=0)
    all_before = np.concatenate(all_before, axis=0)
    all_after = np.concatenate(all_after, axis=0)
    all_assign = np.concatenate(all_assign, axis=0)

    np.save(rf"ccm_feats_before_b={max_batches}.npy", all_before)
    print("💾 Saved ccm_feats_before: ccm_feats_before.npy")
    np.save(rf"ccm_feats_after_b={max_batches}.npy", all_after)
    print("💾 Saved ccm_feats_after: ccm_feats_after.npy")
    np.save(rf"ccm_assign_prob_b={max_batches}.npy", all_assign)
    print("all feats before shape:", all_before.shape, "after shape:", all_after.shape, "assign shape:", all_assign.shape)



# ============================================================
#                子图绘制功能函数（不创建 figure）
# ============================================================
def plot_similarity_subplot(ax, H_proto, channel_names, title):
    """
    在指定 ax 上画通道相似度矩阵
    H_proto: [V, D]
    """
    V, D = H_proto.shape

    # --- 余弦相似度矩阵（与你原来完全一致）---
    sim_matrix = np.matmul(H_proto, H_proto.T)
    norm = np.linalg.norm(H_proto, axis=1, keepdims=True)
    sim_matrix = sim_matrix / norm
    sim_matrix = sim_matrix / norm.T

# --- 若为 After，则按物理簇重新排序 ---
    if title == "After Clustering":
        new_order = [
            r"geopotential$_{500}$",
            r"temperature$_{500}$",
            r"Uwind$_{500}$", r"Vwind$_{500}$",
            r"r_humidity$_{850}$", r"s_humidity$_{850}$",
            r"Uwind$_{850}$", r"Vwind$_{850}$",
            "sst",
            r"Uwind$_{200}$", r"Vwind$_{200}$",
            "shear"
        ]
        # 找到索引
        idx = [channel_names.index(name) for name in new_order]

        # 重排矩阵和标签
        sim_matrix = sim_matrix[idx][:, idx]
        channel_names = new_order

    # --- 热力图绘制 ---
    sns.heatmap(
        sim_matrix,
        cmap='viridis',
        square=True,
        xticklabels=channel_names,
        yticklabels=channel_names,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        ax=ax
    )

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=12)


# ============================================================
#                主函数：对外调用这个即可
# ============================================================
def plot_before_after_similarity(before_path,
                                 after_path,
                                 channel_names,
                                 save_path):
    """
    载入 before / after 特征，计算均值原型，并绘制双子图
    """

    # --- 加载数据 ---
    before_feats = np.load(before_path)    # [N, V, D]
    after_feats = np.load(after_path)      # [N, V, D]

    H_before = before_feats.mean(axis=0)   # [V, D]
    H_after = after_feats.mean(axis=0)     # [V, D]

    # --- 创建画布 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图
    plot_similarity_subplot(
        axes[0],
        H_before,
        channel_names,
        title="Before Clustering"
    )

    # 右图
    plot_similarity_subplot(
        axes[1],
        H_after,
        channel_names,
        title="After Clustering"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"🎉 Saved combined similarity figure: {save_path}")



#=====================开始计算 ICS / IDS ==========================

# ==================================================
# 计算余弦相似度矩阵
# ==================================================
def compute_channel_similarity(feats):   # feats: [B,V,D]
    feats_mean = feats.mean(axis=0)      # [V,D]
    norm = np.linalg.norm(feats_mean, axis=1, keepdims=True) + 1e-8
    feats_norm = feats_mean / norm       # [V,D]
    sim = feats_norm @ feats_norm.T      # [V,V]
    return sim


# ==================================================
# 得到 cluster_id（每个变量属于哪个 cluster）
# ==================================================
def get_cluster_id(assign_prob):  # [B,V,K]
    assign_mean = assign_prob.mean(axis=0)  # [V,K]
    cluster_id = assign_mean.argmax(axis=1) # [V]
    return cluster_id


# ==================================================
# ICS / IDS（自动忽略空簇，方法 1）
# ==================================================
def compute_ics_ids(sim, cluster_id, K):
    V = len(cluster_id)
    ICS_list = []
    IDS_list = []

    # ---- 识别非空簇 ----
    valid_clusters = [k for k in range(K) if np.sum(cluster_id == k) > 0]
    # print("valid clusters:", valid_clusters)

    # -------------------------
    # ① 计算 ICS（类内相似度）
    # -------------------------
    for k in valid_clusters:
        idx = np.where(cluster_id == k)[0]
        if len(idx) < 2:
            continue
        block = sim[np.ix_(idx, idx)]
        vals = block[np.triu_indices_from(block, k=1)]
        ICS_list.append(vals.mean())

    ICS = np.mean(ICS_list)

    # -------------------------
    # ② 计算 IDS（类间分离度）
    # -------------------------
    for p in valid_clusters:
        for q in valid_clusters:
            if p == q:
                continue
            idx_p = np.where(cluster_id == p)[0]
            idx_q = np.where(cluster_id == q)[0]
            block = sim[np.ix_(idx_p, idx_q)]
            IDS_list.append(block.mean())

    IDS = np.mean(IDS_list)

    return ICS, IDS


# ==================================================
# 主函数（你给的代码基础上改的）
# ==================================================
def calculate_ICS_IDS():
    feats_before = np.load("ccm_feats_before_b=6.npy")   # [B,V,D]
    feats_after  = np.load("ccm_feats_after_b=6.npy")    # [B,V,D]
    assign_prob  = np.load("ccm_assign_prob_b=6.npy")    # [B,V,K]

    K = assign_prob.shape[-1]
    print("Number of clusters K =", K)

    # similarity matrices
    sim_before = compute_channel_similarity(feats_before)
    sim_after  = compute_channel_similarity(feats_after)

    # cluster id（唯一来源）
    cluster_id = get_cluster_id(assign_prob)
    print("cluster counts =", np.bincount(cluster_id), "\n")

    # ICS / IDS（自动忽略空簇）
    ics_before, ids_before = compute_ics_ids(sim_before, cluster_id, K)
    ics_after,  ids_after  = compute_ics_ids(sim_after,  cluster_id, K)

    print("Before CCM: ICS =", ics_before, " IDS =", ids_before)
    print("After  CCM: ICS =", ics_after,  " IDS =", ids_after)

    df = pd.DataFrame({
        "Version": ["Before CCM", "After CCM"],
        "ICS (↑)": [ics_before, ics_after],
        "IDS (↓)": [ids_before, ids_after]
    })

    print(df)
    df.to_csv("ics_ids_results.csv", index=False)
    print("已生成 ics_ids_results.csv")




# ============================================================
#                示例：主程序调用方式
# ============================================================

if __name__ == "__main__":

    '''
    计算 feats_before / feats_after 并保存为 .npy 文件
    '''
    # extract_ccm_feats()

    CHANNEL_NAMES = [
        r"geopotential$_{500}$", r"temperature$_{500}$", r"Uwind$_{500}$", r"Vwind$_{500}$",
        r"r_humidity$_{850}$", r"s_humidity$_{850}$", r"Uwind$_{850}$", r"Vwind$_{850}$",
        r"Uwind$_{200}$", r"Vwind$_{200}$", "shear", "sst"
    ]

    plot_before_after_similarity(
        before_path="ccm_feats_before.npy",
        after_path="ccm_feats_after.npy",
        channel_names=CHANNEL_NAMES,
        save_path=r"D:\MacauPrograms\TrajPrediction\MGTCF\my_method_V3_CCM\Picture of papers\similarity_reorder.png"
    )

    # ========== 计算 ICS / IDS ==========
    # calculate_ICS_IDS()