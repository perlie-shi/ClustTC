import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from LatentClust.Unet import Unet3D
from LatentClust.env_net import Env_net
import math

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

class ChannelClusteringModule(nn.Module):
    """
    CCM (Channel Clustering Module) - 带 Cross-Attention Pooling
    输入:
        feats: [B, V, D]   -- 各气象通道的嵌入
        query: [B, D]      -- 来自时序/自数据模块的全局特征 (用于 cross-attention)
    输出:
        out:   [B, D]      -- 聚合后的通道表示
    """
    def __init__(self, d_model=64, num_vars=12, num_clusters=4, momentum=0.99):
        super(ChannelClusteringModule, self).__init__()
        self.num_clusters = num_clusters
        self.d_model = d_model
        self.momentum = momentum

        # 通道 → 聚类分配
        self.cluster_assign = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, num_clusters)
        )

        # 每簇一个独立 FFN
        self.cluster_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_clusters)
        ])

        # 全局簇中心 (EMA)
        self.register_buffer('cluster_centers', torch.randn(num_clusters, d_model))

        # Cross-Attention 参数
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
    def forward(self, feats, query):
        """
        feats:  [B, V, D]
        query:  [B, D]
        """
        device = feats.device  # 获取当前 batch 的 device
        B, V, D = feats.size()

        # === Step1. 软聚类分配 ===
        assign_logits = self.cluster_assign(feats)       # [B,V,K]
        assign_prob = F.softmax(assign_logits, dim=-1)   # [B,V,K]

        # === Step2. EMA 更新全局簇中心 ===
        batch_centers = torch.einsum('bvk,bvd->bkd', assign_prob, feats)
        batch_centers /= (assign_prob.sum(dim=1, keepdim=True).permute(0, 2, 1) + 1e-8)
        batch_centers = batch_centers.mean(0)

        with torch.no_grad():
            # 🔧 保证在同一设备
            self.cluster_centers = (
                self.momentum * self.cluster_centers.to(batch_centers.device)
                + (1 - self.momentum) * batch_centers
            )

        # === Step3. Cluster-aware FFN ===
        cluster_feats = []
        for k in range(self.num_clusters):
            hk = self.cluster_ffn[k](feats)
            cluster_feats.append(assign_prob[..., k:k + 1] * hk)
        feats_new = torch.stack(cluster_feats, dim=-1).sum(-1)  # [B,V,D]

        # === Step4. Cross-Attention Pooling ===
        Q = self.query_proj(query).unsqueeze(1)        # [B,1,D]
        K = self.key_proj(feats_new)                   # [B,V,D]
        V_ = self.value_proj(feats_new)                # [B,V,D]

        # 🔧 保证 cluster_centers 在同一设备
        c = self.cluster_centers.to(device)

        dist = ((feats.unsqueeze(2) - c) ** 2).sum(-1)  # [B,V,K]
        attn_score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)  # [B,1,V]
        attn_weight = F.softmax(attn_score, dim=-1)                  # [B,1,V]
        out = torch.bmm(attn_weight, V_).squeeze(1)                  # [B,D]

        self.assign_prob_cache = assign_prob.detach()

        self.feats_before = feats.detach()         # 聚类前
        self.feats_after  = feats_new.detach()     # 聚类后
        return out, assign_prob, self.cluster_centers

    
class GeneratorEncoder(nn.Module):
    """GeneratorEncoder is part of TrajectoryGenerator"""
    def __init__(
        self, embedding_dim=32, h_dim=64, mlp_dim=128, lstm_numlayer=1,
        dropout=0.0, meteo_num=12, cnn_out_dim=64, fused_dim=32, num_heads=2, trans_numlayer=2, trans_dropout=0.1
    ):
        super(GeneratorEncoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.lstm_numlayer = lstm_numlayer
        self.meteo_num = meteo_num

        #为每个meteo变量定义一个独立的cnn，即图像编码
        self.shared_cnn = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # H/2 x W/2
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # 最终输出固定大小
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, cnn_out_dim),
                nn.ReLU()
        )
        self.var_embed = nn.Embedding(meteo_num, cnn_out_dim)

        #用于变量融合的transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_dim, nhead=num_heads, dim_feedforward=128, dropout=trans_dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_numlayer)
                # === 添加 CCM 模块 ===
        self.use_ccm = True
        self.num_clusters = 4  # 可调
        if self.use_ccm:
            self.ccm = ChannelClusteringModule(
                d_model=cnn_out_dim,
                num_vars=meteo_num,
                num_clusters=self.num_clusters
            )
        self.global_context_proj = nn.Linear(4, cnn_out_dim)
        self.proj = nn.Linear(cnn_out_dim, fused_dim)
        

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, lstm_numlayer, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4,embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.lstm_numlayer, batch, self.h_dim).cuda(),
            torch.zeros(self.lstm_numlayer, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj,meteo_obs):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 4)
        - meteo_obs: [b, c, obslen, h, w]
        Output:
        - final_h: Tensor of shape (self.lstm_numlayer, batch, self.h_dim)
        """

        # meteo data embedding 
        meteo_obs = meteo_obs.permute(0,2,1,3,4) # [96,8,11,64,64]
        batch, obslen, meteo_num, h, w = meteo_obs.shape
        meteo_obs = meteo_obs.reshape(batch*obslen, meteo_num, h, w) # [96*8, 11, 64, 64]
        cnn_feats = []
        for i in range(meteo_num):
            meteo_obs_i = meteo_obs[:, i, :, :].unsqueeze(1) # [96*len, 1, 64, 64]
            feats_i = self.shared_cnn(meteo_obs_i) # [96*len, 64]
            cnn_feats.append(feats_i)
        cnn_feats = torch.stack(cnn_feats, dim=1) # [96*len, 11, 64]

        # meteo type embedding
        var_ids = torch.arange(self.meteo_num, device=meteo_obs.device) # [meteo_num]
        var_embed = self.var_embed(var_ids) # [meteo_num, 64]
        var_embed = var_embed.unsqueeze(0)  # [1, meteo_num, 64]     
        cnn_feats = cnn_feats + var_embed # [96*len, meteo_num, 64]

        # using transformer to fuse meteo data
        # fused = self.transformer_encoder(cnn_feats) # [96*len, 11, 64]
        # fused = fused.mean(dim=1) # [96*len, 64]
        # fused = self.proj(fused) # [96*len, 32]

        # === Transformer编码 ===
        fused_feats = self.transformer_encoder(cnn_feats)   # [B*len, V, D]
        B_len, V, D = fused_feats.shape

        # CCM
        if self.use_ccm:
            # query: 来自台风轨迹编码的上下文 (取均值或最后一层隐状态)
            query = self.global_context_proj(obs_traj.mean(dim=0))  # [B,D]
            fused_feats = fused_feats.view(batch, obslen, -1, fused_feats.shape[-1])
            fused_feats = fused_feats.mean(dim=1)                               # [B, V, D]
            fused_out, assign_prob, centers = self.ccm(fused_feats, query)
        else:
            fused_out = fused_feats.mean(dim=1)

        fused = self.proj(fused_out) #[96,32]



        # fused_meteo = fused.reshape(batch, obslen, -1).permute(1,0,2) # [8, 96, 32]
        fused_meteo = fused.unsqueeze(0).expand(obslen, batch, fused.size(-1))  # [obs_len, B, 32]

        # Encode observed Trajectory
        inputDim = obs_traj.size(2) #4 [8,96,4]
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, inputDim))
        obs_traj_embedding = obs_traj_embedding.reshape(-1, batch, self.embedding_dim)
        obs_traj_embedding = obs_traj_embedding + fused_meteo #[8,96,32]

        state_tuple = self.init_hidden(batch)
        output, final_h = self.encoder(obs_traj_embedding, state_tuple) #这儿的encoder是LSTM
        #final_h -> [hn,cn]
        return final_h[0], final_h[1], output

class DiscrimiEncoder(nn.Module):
    """DiscrimiEncoder is part of Discriminator"""
    def __init__(
        self, embedding_dim=32, h_dim=128, mlp_dim=128, lstm_numlayer=1,
        dropout=0.0, meteo_num=11
    ):
        super(DiscrimiEncoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.lstm_numlayer = lstm_numlayer
        self.meteo_num = meteo_num

        #为每个meteo变量定义一个独立的cnn，即图像编码
        self.encoder = nn.LSTM(embedding_dim, h_dim, lstm_numlayer, dropout=dropout)
        self.spatial_embedding = nn.Linear(4, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.lstm_numlayer, batch, self.h_dim).cuda(),
            torch.zeros(self.lstm_numlayer, batch, self.h_dim).cuda()
        )
    
    def forward(self, obs_traj, img_embed):
        """
        Inputs:
        - obs_traj: Tensor of shape (seq_len, batch, 4)
        - img_embed: [seq_len, batch, 32]
        Output:
        - final_h: Tensor of shape (self.lstm_numlayer, batch, self.h_dim)
        """
        batch = obs_traj.size(1)
        inputDim = obs_traj.size(2) #4
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, inputDim))
        obs_traj_embedding = obs_traj_embedding.reshape(
            -1, batch, self.embedding_dim
        )
        obs_traj_embedding = obs_traj_embedding+img_embed #[seq_len,b,32]
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        return state[0], state[1], output


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, pred_len, embedding_dim=32, h_dim=64, mlp_dim=1024, lstm_numlayer=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='None', #pool_net原默认值
        neighborhood_size=2.0, grid_size=8,embeddings_dim=128,
            h_dims=128,
    ):
        super(Decoder, self).__init__()

        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, lstm_numlayer, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                pass
            elif pooling_type == 'spool':
                pass

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 4)
    def init_hidden(self, batch):
        return (
            torch.zeros(self.lstm_numlayer, batch, self.h_dim).cuda(),
            torch.zeros(self.lstm_numlayer, batch, self.h_dim).cuda()
        )
    def forward(self, obs_traj, obs_traj_rel, last_pos, last_pos_rel, state_tuple,
                seq_start_end,decoder_img,last_img):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 4) obs的最后一个轨迹点信息
        - last_pos_rel: Tensor of shape (batch, 4)
        - state_tuple: (hh, ch) each tensor of shape (lstm_numlayer, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - decoder_img [4,batch,32]
        - last_img [batch,32]
        Output:
        - pred_traj: tensor of shape (self.pred_len, batch, 2)

        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)               # linear (4,32) ; [96,4]->[96,32]
        decoder_input = decoder_input.reshape(-1, batch, self.embedding_dim)  # [1,96,32]
        
        # add img_information
        last_img = last_img.unsqueeze(0) #[1,96,32]
        decoder_input = decoder_input+last_img

        # obs_traj_rel_new = obs_traj_rel.clone()
        # obs_date_mask_new = obs_date_mask.clone()

        for i_step in range(self.pred_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple) # output [1,96,64]
            rel_pos = self.hidden2pos(output.reshape(-1, self.h_dim)) # nn.Linear(h_dim, 4) ; [96,64]->[96,4]
            curr_pos = rel_pos + last_pos


            rel_pos = rel_pos.unsqueeze(0)
            embedding_input = rel_pos
            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.reshape(-1, batch, self.embedding_dim)

            # add img_information
            decoder_img_one = decoder_img[i_step].unsqueeze(0)
            decoder_input = decoder_input+decoder_img_one

            pred_traj_fake_rel.append(rel_pos.reshape(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        # 预测的路径差值--与第八个的pos相加即能路径
        return pred_traj_fake_rel, state_tuple[0]



class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, meteo_num=12, embedding_dim=32, encoder_h_dim=64,
        decoder_h_dim=64, mlp_dim=128, lstm_numlayer=1, noise_dim=(16, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=False, dropout=0.0, bottleneck_dim=16, 
        activation='relu', batch_norm=0, neighborhood_size=2.0, grid_size=8,num_gs=6,num_sample=6,use_attention=True,
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim # (16, )
        self.lstm_numlayer = lstm_numlayer # 1
        self.noise_type = noise_type # 'gaussian'
        self.noise_mix_type = noise_mix_type # 'ped'
        self.pooling_type = pooling_type 
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.num_gs = num_gs
        self.num_sample = num_sample
        self.use_attention = use_attention

        self.Unet = Unet3D(meteo_num,meteo_num)
        # self.predrnn = Net(32, 1, h_w=[50, 50], n_GPU=1)
        self.var_embed = nn.Embedding(meteo_num, 32)
        self.img_embedding = nn.Linear(64*64,32)
        self.meteo_embedding = nn.Linear(64*64, 32)
        self.env_net = Env_net(meteo_num)
        self.inner_fuse_meteo = Env_net(meteo_num)
        self.feature2dech_env = nn.Linear(96,64)
        self.feature2dech = nn.Linear(96, 64)


        self.encoder = GeneratorEncoder(
            embedding_dim=embedding_dim,#32
            h_dim=encoder_h_dim,        #64
            mlp_dim=mlp_dim,            #128
            lstm_numlayer=lstm_numlayer,      #1
            dropout=dropout,             #0.0
            meteo_num=meteo_num
        )

        self.self_fuse_meteo = GeneratorEncoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            lstm_numlayer=lstm_numlayer,
            dropout=dropout,
            meteo_num=meteo_num
        )

        self.gs = nn.ModuleList()
        # [11-5,6and10,7-9]
        for i in range(num_gs):
            self.gs.append(Decoder(
                pred_len,
                embedding_dim=embedding_dim,
                h_dim=decoder_h_dim,
                mlp_dim=mlp_dim,
                lstm_numlayer=lstm_numlayer,
                pool_every_timestep=pool_every_timestep,
                dropout=dropout,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                pooling_type=pooling_type,
                grid_size=grid_size,
                neighborhood_size=neighborhood_size,
                embeddings_dim=embedding_dim,
                h_dims=encoder_h_dim,
            ))
        self.net_chooser = nn.Sequential(
            nn.Linear(encoder_h_dim, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, num_gs),
        )

        if pooling_type == 'pool_net':
            pass
        elif pooling_type == 'spool':
            pass

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        if use_attention:
            # attention pooling over meteo vars
            self.attn_mlp = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def get_samples(self, enc_h, num_samples=6): #num_samples表示每次选择多少个生成器
        """Returns generator indexes of shape (batch size, num samples)"""

        # net_chooser: sequential linear+relu+linear
        net_chooser_out = self.net_chooser(enc_h) # [batch_size, num_gs]

        dist = Categorical(logits=net_chooser_out)
        sampled_gen_idxs = dist.sample((num_samples,)).transpose(0, 1) #返回一个随机的类别索引 [batch_size,num_samples]
        return net_chooser_out, sampled_gen_idxs.detach().cpu().numpy()

    def get_topK_samples(self, enc_h, num_samples=6): #num_samples表示每次选择多少个生成器
        """直接选取概率最大的 num_samples 个生成器索引，shape: [batch_size, num_samples]
        # 训练阶段，采样增强
        _, sampled_gen_idxs = model.get_samples(enc_h, num_samples=6)

        # 测试阶段，稳定选择 top-k
        _, topk_gen_idxs = model.get_topk_samples(enc_h, num_samples=3)
        """
        net_chooser_out = self.net_chooser(enc_h)  # [batch_size, num_gs]
        
        # 取 top-k 对应的索引（不采样）
        topk_values, topk_indices = torch.topk(net_chooser_out, k=num_samples, dim=1)
        return net_chooser_out, topk_indices.detach().cpu().numpy()
    

    def mix_noise(self,final_encoder_h,seq_start_end,batch,user_noise=None):
        mlp_decoder_context_input = final_encoder_h.view(
            -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        # decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_h = decoder_h.view(-1, batch, self.encoder_h_dim)

        decoder_c = torch.zeros(
            self.lstm_numlayer, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        return state_tuple

    def forward(self, obs_traj, obs_traj_rel, seq_start_end,meteo_obs,inner_data,
                num_samples=1,all_g_out=False,predrnn_img=None,user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 4)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 4)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        - meteo_obs: (b,c,obs_len,h,w) [96,11,8,64,64]
        - num_samples: net_chooser=1; discriminator_step=1; generator_step=6

        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        obs_len = obs_traj_rel.size(0)
        meteo_num = meteo_obs.size(1) # 2

        # for netchooser
        # self_fuse_meteo -> GeneratorEncoder
        fused_h, _, _ = self.self_fuse_meteo(obs_traj_rel, meteo_obs)        # [num_layer,b,64]
        fused_im, _, _ = self.inner_fuse_meteo(inner_data,meteo_obs[:,:,-1]) # inner_data和meteo_obs都取最后一个时间步 [b,32]
        dec_h_evn = self.feature2dech_env(torch.cat([fused_h.squeeze(),fused_im],dim=1)).unsqueeze(0) #[b,96] -> [b,64]

        # for generator Unet预测的12个时间步的图像信息 
        predout_unet = self.Unet(meteo_obs)          # (b,c,obs_len,h,w) [96,11,8,64,64] -> [96,11,11,64,64]
        first_imgs = meteo_obs[:,:,0].unsqueeze(2)   # [96,11,1,64,64]
        all_img = torch.cat([first_imgs,predout_unet],dim=2) #[96,11,12,64,64]
        

        # encoding Unet预测的图像，方便后续送入decoder中，这里还加了一个类型嵌入
        var_ids = torch.arange(meteo_num, device=meteo_obs.device) # [meteo_num]
        var_embed = self.var_embed(var_ids)                 # [meteo_num, 32]
        var_embed = var_embed.reshape(1,meteo_num,1,-1)     # [1, meteo_num,1,32]

        img_input = all_img.reshape(batch, meteo_num, 12,-1)   # [b,11,12,4096]
        img_embed_input = self.img_embedding(img_input)     # [b,c,12,32]
        img_embed_input = img_embed_input + var_embed       # [b,11,12,32]
        
        if self.use_attention:
            attn_score = self.attn_mlp(img_embed_input)     # [b,11,12,1]
            attn_weights = torch.softmax(attn_score, dim=1) # softmax over V [b,11,12,1]
            img_embed_input = torch.sum(attn_weights * img_embed_input, dim=1) # [b,12,32]
            img_embed_input = img_embed_input.permute(1,0,2)                   # [12,b,32]
        else:
            img_embed_input = img_embed_input.mean(dim=1).squeeze().permute(1,0,2) # [12,b,32]


        '''和上面一样的步骤'''
        # Encode sequence
        # Generator
        encoder_img = all_img[:,:,:obs_len]                 # [b,c,8,64,64] 获取obs_len的图像信息
        final_encoder_h,_,_ = self.encoder(obs_traj_rel, encoder_img)
        dec_h = self.feature2dech(torch.cat([final_encoder_h.squeeze(), fused_im], dim=1)).unsqueeze(0)
        

        image_out = all_img #[96,11,12,64,64] 经过Unet的图像信息，也算是预测的图像信息

        if all_g_out: # (obs_len, batch, 4) net_chooser_step=True 其余都是false
            last_pos = obs_traj[-1]
            last_pos_rel = obs_traj_rel[-1]
            decoder_img = img_embed_input[obs_len:] #后4个时间步的image信息
            last_img = img_embed_input[obs_len - 1] #前8个时间步的最后一个时间步的image信息
            preds_rel = []
            with torch.no_grad():
                '''这种方式的噪声混合能够增强模型的鲁棒性，
                并使其能够生成多样化的轨迹预测，尤其是在生成对抗网络（GAN）或者多生成器模型中。
                state_tuple = (decoder_h, decoder_c)作为decoder的输入'''
                state_tuple = self.mix_noise(dec_h, seq_start_end, batch) #这里总感觉是出错了，应该是dec_h_evn 而不是dec_h
                for g_i , g in enumerate(self.gs): # 遍历所有的生成器 这里有6个生成器,都是decoder
                    pred_traj_fake_rel, final_decoder_h = g(
                        obs_traj,
                        obs_traj_rel,
                        last_pos,
                        last_pos_rel,
                        state_tuple,
                        seq_start_end,
                        decoder_img,
                        last_img
                    )
                    preds_rel.append(pred_traj_fake_rel.reshape(self.pred_len, 1, batch, 4))

            # [prelen,g_num,batch,4] [4,6,96,4]
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1) 
            net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h_evn.squeeze(), num_samples) #num_samples=1
            # 这里只是会调用get_samples函数,但不做生成器选择

        else: # num_samples: discriminator_step=1; generator_step=6
            with torch.no_grad(): 
                '''#这里是无梯度的计算，因为这里在训练生成器，因此netchooser无需计算梯度，和之前恰恰相反
                '''
                net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h.squeeze(),num_samples)
            # Predict Trajectory
            preds_rel = []
            for sample_i in range(num_samples): #对于generator_step=6 表示取样6次
                pred_traj_fake_rel_reverse = torch.ones((self.pred_len, batch, 4), requires_grad=True).cuda() # 为每个样本初始化预测轨迹
                gs_index = sampled_gen_idxs[:,sample_i] #所有样本的第i次取样的生成器索引
                # for g_i in range(np.unique(gs_index).shape[0]): #第一次取样有多少个不同的生成器索引
                for g_i in np.unique(gs_index): # 这里的g_i是生成器的索引
                    # sampled_gen_idxs   [b,num_samples]
                    now_data_index = (gs_index == g_i) #布尔数组,所有使用生成器gi的样本
                    if np.sum(now_data_index) < 1:
                        continue
                    last_pos = obs_traj[-1, now_data_index] #[obs_len,batch,4]
                    last_pos_rel = obs_traj_rel[-1, now_data_index]
                    decoder_img = img_embed_input[obs_len:, now_data_index]
                    last_img = img_embed_input[obs_len - 1, now_data_index]

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    state_tuple = self.mix_noise(dec_h[:,now_data_index], seq_start_end[now_data_index], np.sum(now_data_index))
                    # the method of sampling is not same as the MG-GAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    decoder = self.gs[g_i]
                    decoder_out = decoder(
                        obs_traj[:,now_data_index],
                        obs_traj_rel[:,now_data_index],
                        last_pos,
                        last_pos_rel,
                        state_tuple,
                        seq_start_end[now_data_index],
                        decoder_img,
                        last_img
                    )
                    pred_traj_fake_rel, final_decoder_h = decoder_out
                    pred_traj_fake_rel_reverse[:, now_data_index] = pred_traj_fake_rel
                preds_rel.append(pred_traj_fake_rel_reverse.reshape(self.pred_len, 1, batch, 4))
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)
            # [prelen,num_samples,batch,4]

        return pred_traj_fake_rel_nums,image_out,net_chooser_out, sampled_gen_idxs


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, meteo_num=11, embedding_dim=32, h_dim=128, mlp_dim=128,
        lstm_numlayer=1, activation='relu', batch_norm=False, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim # 128
        self.d_type = d_type

        self.img_embedding = nn.Linear(meteo_num * 64 * 64, 32)
        self.encoder = DiscrimiEncoder(
            embedding_dim=embedding_dim,# 32
            h_dim=h_dim,                # 128
            mlp_dim=mlp_dim,            # 128
            lstm_numlayer=lstm_numlayer,# 1
            dropout=dropout             # 0.0
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        info_classifier_dims = [h_dim, mlp_dim, 2]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            pass

    def forward(self, traj, traj_rel, seq_start_end,img):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 4)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 4)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - img [b,c,len,h,w]
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        b,c,len,_,_ = img.shape
        img_embed_input = img.reshape(b,len,-1) # [b,len,c*64*64]
        img_embed = self.img_embedding(img_embed_input).permute(1,0,2) #[b,len,32]-> [len,b,32]
        final_h, _, _ = self.encoder(traj_rel,img_embed) # [num_layer, b, h_dim]

        # output = final_encoder['output']
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()# [b,h_dim]
        else:
            pass
        scores = self.real_classifier(classifier_input)

        return scores,classifier_input
