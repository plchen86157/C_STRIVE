# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Runs through nuScenes dataset and runs various evaluations on the traffic model.

By default, computes the same losses/errors as during training.
"""

import os
import time
from tqdm import tqdm
import torch
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from losses.traffic_model import (
    TrafficModelLoss,
    compute_disp_err,
    compute_coll_rate_env,
)
from nuscenes.nuscenes import NuScenes
# from datasets.collision_dataset import CollisionDataset

from datasets.collision_dataset_from_csv import CollisionDataset

from datasets.map_env import NuScenesMapEnv
from losses.traffic_model import compute_coll_rate_veh
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, load_state
from utils.config import get_parser, add_base_args
from icecream import ic

import pandas as pd
from nuscenes.map_expansion.map_api import NuScenesMap
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Polygon as pg
import matplotlib.patches as mpatches
from PIL import Image
import math
import csv
ic.configureOutput(prefix=f"Debug | ", includeContext=True)
# ic.disable()
from datasets.utils import (
    MeanStdNormalizer,
    read_adv_scenes,
    NUSC_NORM_STATS,
    NUSC_VAL_SPLIT_200,
    NUSC_VAL_SPLIT_400,
)
VEH_COLL_THRESH = 0.02
vehicle_width = 2
vehicle_length = 4.7

def parse_cfg():
    """
    Parse given config file into a config object.

    Returns: config object and config dict
    """
    parser = get_parser("Test motion model")
    parser = add_base_args(parser)

    # additional data args

    parser.add_argument(
        "--whole_csv_path",
        type=str,
        default="./nuscenes_csv_result_on_STRIVE/",
    )
    parser.add_argument('--forward_csv_path', default='nuscenes_csv_result_on_STRIVE_forward/')
    parser.add_argument('--map_path', type=str, default='./data/nuscenes/trainval/')
    parser.add_argument('--animation_path', type=str, default='animation_on_STRIVE/')
    parser.add_argument('--folder_name', type=str, default='-output 4')


    parser.add_argument(
        "--test_on_val",
        dest="test_on_val",
        action="store_true",
        help="If given, uses the validation dataset rather than test set for evaluation.",
    )
    parser.set_defaults(test_on_val=False)
    parser.add_argument(
        "--shuffle_test",
        dest="shuffle_test",
        action="store_true",
        help="If given, shuffles test dataset.",
    )
    parser.set_defaults(shuffle_test=False)

    #
    # test options
    #
    # reconstruct (use posterior)
    parser.add_argument(
        "--test_recon_viz_multi",
        dest="test_recon_viz_multi",
        action="store_true",
        help="Save all-agent visualization for reconstructing all test sequences.",
    )
    parser.set_defaults(test_recon_viz_multi=False)
    parser.add_argument(
        "--test_recon_coll_rate",
        dest="test_recon_coll_rate",
        action="store_true",
        help="Computes collision rate of reconstructed test trajectories",
    )
    parser.set_defaults(test_recon_coll_rate=False)
    # sample (use prior)
    parser.add_argument(
        "--test_sample_viz_multi",
        dest="test_sample_viz_multi",
        action="store_true",
        help="Save all-agent visualization for samplin all test sequences.",
    )
    parser.set_defaults(test_sample_viz_multi=False)
    parser.add_argument(
        "--test_sample_viz_rollout",
        dest="test_sample_viz_rollout",
        action="store_true",
        help="Create videos of multiple sampled futures individually.",
    )
    parser.set_defaults(test_sample_viz_rollout=False)

    parser.add_argument(
        "--test_sample_disp_err",
        dest="test_sample_disp_err",
        action="store_true",
        help="Computes min displacement errors (ADE, FDE, and angle-based version) based on multiple samples.",
    )
    parser.set_defaults(test_sample_disp_err=False)
    parser.add_argument(
        "--test_sample_coll_rate",
        dest="test_sample_coll_rate",
        action="store_true",
        help="Computes collision rate of N random samples.",
    )
    parser.set_defaults(test_sample_coll_rate=False)


    parser.add_argument(
        "--test_sample_num", type=int, default=3, help="Number of future traj to sample"
    )
    parser.add_argument(
        "--test_sample_future_len",
        type=int,
        default=None,
        help="If not None, samples this many steps into the future rather than future_len",
    )
    parser.add_argument(
        "--collision_dir",
        type=str,
        help="Collision Dataset Path",
    )
    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict

def angle_vectors(v1, v2):
    """ Returns angle between two vectors.  """
    # 若是車輛靜止不動 ego_vec為[0 0]
    if v1[0] < 0.0001 and v1[1] < 0.0001:
        v1_u = [1.0, 0.1]
    else:
        v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # 因為np.arccos給的範圍只有0~pi (180度)，但若cos值為-0.5，不論是120度或是240度，都會回傳120度，因此考慮三四象限的情況，強制轉180度到一二象限(因車輛和行人面積的對稱性，故可這麼做)
    #if v1_u[1] < 0:
    #    v1_u = v1_u * (-1)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if math.isnan(angle):
        return 0.0
    else:
        return angle
    
def is_intersection(x, y, town_name):
    nusc_map = NuScenesMap(dataroot='./NuScenes/', map_name=town_name)
    rstk = nusc_map.record_on_point(x, y, "road_segment")
    if rstk == "":
        return False
    rs = nusc_map.get("road_segment", rstk)
    return rs["is_intersection"]

def compute_v(traj: torch.Tensor) -> np.ndarray:
    """
    Compute velocity from trajectory
    """
    traj = traj.cpu().numpy()
    # ic(traj.shape)
    traj_v = np.linalg.norm((traj[:, 1:, :2] - traj[:, :-1, :2]), axis=2) / 0.2
    traj_v = np.concatenate((traj_v[:, :1], traj_v), axis=1)
    # ic(traj_v.shape)
    return traj_v


def compute_h(traj: torch.Tensor) -> np.ndarray:
    """
    Compute heading angle from cosh and sinh
    """
    traj = traj.cpu().numpy()
    cosh = traj[:, :, 2]
    sinh = traj[:, :, 3]
    traj_h = np.empty(cosh.shape)
    for i in range(traj.shape[0]):
        for j in range(traj.shape[1]):
            if cosh[i, j] == -1:
                traj_h[i, j] = np.pi
            else:
                temp = sinh[i, j] / (1 + cosh[i, j])
                traj_h[i, j] = 2 * np.arctan(temp)
    traj_h = np.rad2deg(traj_h)
    return traj_h


def run_one_epoch(
    cfg,
    data_loader,
    model,
    map_env,
    loss_fn,
    device,
    out_path,
    test_recon_viz_multi=False,
    test_recon_coll_rate=False,
    test_sample_viz_multi=False,
    test_sample_viz_rollout=False,
    test_sample_disp_err=False,
    test_sample_coll_rate=False,
    test_sample_num=3,  # how many futures to sample from the prior
    test_sample_future_len=None,
    use_challenge_splits=False,
):
    """
    Run through test dataset and perform various desired evaluations.
    """
    pbar = tqdm.tqdm(data_loader)

    if test_recon_viz_multi:
        recon_multi_agent_out_path = os.path.join(out_path, "viz_recon_multi")
        mkdir(recon_multi_agent_out_path)
    if test_sample_viz_multi:
        sample_multi_agent_out_path = os.path.join(out_path, "viz_sample_multi")
        mkdir(sample_multi_agent_out_path)
    if test_sample_viz_rollout:
        sample_rollout_out_path = os.path.join(out_path, "viz_sample_rollout")
        mkdir(sample_rollout_out_path)

    metrics = {}  # regular error metrics, want to compute mean
    freq_metrics = {}  # frequency metrics where we sum occurences of something.
    data_idx = 0
    categories = ['car', 'truck']
    ninfo = NUSC_NORM_STATS[tuple(sorted(categories))]
    norm_mean = [
        ninfo["lscale"][0],
        ninfo["lscale"][0],
        ninfo["h"][0],
        ninfo["h"][0],
        ninfo["s"][0],
        ninfo["hdot"][0],
    ]
    norm_std = [
        ninfo["lscale"][1],
        ninfo["lscale"][1],
        ninfo["h"][1],
        ninfo["h"][1],
        ninfo["s"][1],
        ninfo["hdot"][1],
    ]
    normalizer = MeanStdNormalizer(
        torch.Tensor(norm_mean), torch.Tensor(norm_std)
    )
    for i, data in enumerate(pbar):
        # print("i:", i)
        scene_graph, map_idx, scene_name = data
        # ic.enable()
        # ic(scene_name)
        # ic(scene_graph)
        # ic(map_idx)
        # ic.disable()

        scene_name = scene_name[0]
        traj = scene_graph.past[:, :, :4]
        traj = traj.to(device)  # (NA, 20, 4)
        # ic(traj.shape)
        for j in range(4):
            scene_graph = scene_graph.to(device)
            coll_type = scene_graph.col_t
            map_idx = map_idx.to(device)
            coll_type = coll_type.to(device)
            # ic.enable()
            # ic(scene_graph)
            if j != 0:
                # future_v = (
                #     torch.from_numpy(future_v)
                #     .to(device)
                #     .unsqueeze(2)
                #     .type(torch.float32)
                # )
                # future_hdot = (
                #     torch.from_numpy(future_hdot)
                #     .to(device)
                #     .unsqueeze(2)
                #     .type(torch.float32)
                # )

                # ic.enable()
                # ic(j)
                # ic(traj[:, -8:, :].shape, future_v.shape, future_hdot.shape)

                past_traj = traj[:, -8:, :]
                f_v = compute_v(past_traj)
                f_h = compute_h(past_traj)
                f_v = (
                    torch.from_numpy(f_v)
                    .to(device)
                    .unsqueeze(2)
                    .type(torch.float32)
                )
                f_h = (
                    torch.from_numpy(f_h)
                    .to(device)
                    .unsqueeze(2)
                    .type(torch.float32)
                )
                # scene_graph.past = torch.cat(
                #     [traj[:, -8:, :], future_v[:, -8:, :], future_hdot[:, -8:, :]],
                #     dim=2,
                # )
                # ic(j, past_traj.shape)
                # ic(j, f_v.shape)
                # ic(j, f_h.shape)
                scene_graph.past = torch.cat(
                    [past_traj, f_v, f_h],
                    dim=2,
                )

                # ic(scene_graph)
                # ic.disable()
            # predict
            # ic(j, scene_graph.past.shape)
            pred = model(scene_graph, map_idx, map_env, use_post_mean=True)
            # concate future state to traj
            future_state = pred["future_pred"]  # (NA, 12, 4)
            traj = torch.cat([traj, future_state], dim=1)
            # compute future velocity and heading rate for next iteration
            # ic(j, traj.shape, traj)
            # ic(j, future_state.shape)
            # ic(scene_graph.past[0])

            #########################################
            # scene_graph.past = normalizer.unnormalize(scene_graph.past)
            #########################################

            # ic(scene_graph.past[0])
            # ic(pred["future_pred"][0])


            # future_v = compute_v(future_state)  # (NA, 12)
            # future_hdot = compute_h(future_state)  # (NA, 12)
            # ic(future_v.shape)
            # ic(future_hdot.shape)
            # ic.disable()

        #     # metrics to save
        #     batch_freq_metrics = {}
        # continue
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        # print(scene_graph)
        # print(map_idx)
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)

        # uses mean of posterior to compute recon errors
        # pred = model(scene_graph, map_idx, map_env, use_post_mean=True)

        # future_pred = pred["future_pred"]  # (NA, FT, 4)
        agent_ids = scene_graph.agent_ids

        # ic(agent_ids)
        # ic(traj.shape, traj)

        # print(os.path.join(cfg.collision_dir + 'test/' + scene_name + '.csv'))
        first_traj_df = pd.read_csv(os.path.join(cfg.collision_dir + 'test/' + scene_name + '.csv'), index_col=False)
        # EGO
        ego_list = []
        vehicle_list = []
        for track_id, remain_df in first_traj_df.groupby("TRACK_ID"):
            if track_id == 'ego':
                ego_list.append(remain_df)
        # print(ego_list[0])
                
        #########################################
        traj = normalizer.unnormalize(traj)
        #########################################

        pred_v = compute_v(traj)
        pred_h = compute_h(traj)
        pred_v = (
            torch.from_numpy(pred_v)
            .to(device)
            .unsqueeze(2)
            .type(torch.float32)
        )
        pred_h = (
            torch.from_numpy(pred_h)
            .to(device)
            .unsqueeze(2)
            .type(torch.float32)
        )
        pred_xy = traj[:, :, :2]
        all_state = torch.cat(
                [pred_xy, pred_v, pred_h],
                dim=2,
            )
        all_state = all_state.cpu().numpy()
        # ic(all_state.shape, all_state) # NA, 24, 4
        pred_with_ids = list(zip(agent_ids[0], all_state))

        df_columns = ["TRACK_ID", "TIMESTAMP", "V", "X", "Y", "YAW"]
        df = pd.DataFrame(columns=df_columns)
        for agent_id, pred in pred_with_ids:
            if agent_id != 'ego':
                # print(f"Agent ID: {agent_id}, Predicted Trajectory: {pred.shape}") # 24, 4
                timesteps = range(pred.shape[0])  # 從 0 開始的 timestamp
                
                # 將 agent_id、timesteps 和預測的軌跡值組成 DataFrame
                agent_df = pd.DataFrame({
                    "TRACK_ID": [agent_id] * pred.shape[0],
                    "TIMESTAMP": timesteps,
                    "V": pred[:, 2],  # 預測的速度
                    "X": pred[:, 0],  # 預測的 X 座標
                    "Y": pred[:, 1],  # 預測的 Y 座標
                    "YAW": pred[:, 3],  # 預測的航向角
                })
                vehicle_list.append(agent_df)
                # 將每個 agent 的 DataFrame 追加到主 DataFrame
                df = pd.concat([df, agent_df], ignore_index=True)
        combined_list = ego_list + vehicle_list
        combined_df = pd.concat(combined_list, ignore_index=True)
        result_list = []
        variant = scene_name.split('_')[7]
        scenario_length = int(variant.split('-')[-1]) + 1
        for track_id, track_df in combined_df.groupby('TRACK_ID'):
            if len(track_df) > scenario_length:
                trimmed_df = track_df.head(scenario_length)
            else:
                trimmed_df = track_df
            result_list.append(trimmed_df)
        trimmed_data = pd.concat(result_list, ignore_index=True)
        
        # combined_df.columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y', 'YAW']
        correct_timestamps = trimmed_data[trimmed_data['TRACK_ID'] == 'ego']['TIMESTAMP'].values
        grouped_df = trimmed_data[trimmed_data['TRACK_ID'] != 'ego'].groupby('TRACK_ID')
        
        # print(combined_df.TRACK_ID.values)
        for track_id, group in grouped_df:
            # print(track_id, group.index, correct_timestamps[:len(group)])
            trimmed_data.loc[group.index, 'TIMESTAMP'] = correct_timestamps[:len(group)]
        # combined_df = combined_df[['TRACK_ID', 'TIMESTAMP', 'X', 'Y', 'YAW']]
        # combined_df.insert(2, 'V', 0)
        # combined_df['TRACK_ID'] = combined_df['TRACK_ID'].apply(lambda x: str(int(float(x))) if x != 'ego' else x)
        trimmed_data.fillna(0, inplace=True)
        trimmed_data.to_csv(cfg.whole_csv_path + scene_name + '.csv', index=False)

def calculate_yaw_diff(row, next_row):
    """
    計算當前行和下一行的位移向量角度差
    row: 當前行
    next_row: 下一行
    """
    real_pred_x = row['X']
    real_pred_y = row['Y']
    
    real_pred_x_next = next_row['X']
    real_pred_y_next = next_row['Y']
    
    # 計算當前到下一點的位移向量
    other_vec = [ - (real_pred_y_next - real_pred_y), real_pred_x_next - real_pred_x]
    
    # 基準向量，假設為朝向 X 軸正方向
    base_vec = [1, 0]
    
    # 計算向量角度（返回的是弧度）
    real_other_angle = (np.rad2deg(angle_vectors(other_vec, base_vec)) + 90.0) * np.pi / 180
    
    # 如果靜止不動，使用當前的 YAW 值
    if np.abs(real_pred_x_next - real_pred_x) < 0.01 and np.abs(real_pred_y_next - real_pred_y) < 0.01:
        real_other_angle = (row['YAW'] + 90.0) * np.pi / 180
    
    return real_other_angle

# interpolation & forward
def post_process_for_inference_and_metric(args):
    """ input: original inference csv """
    all_data_num = 0

    lane_change_num = 0
    junction_crossing_num = 0
    LTAP_num = 0
    opposite_direction_num = 0
    rear_end_num = 0

    lane_change_all = 0
    junction_crossing_all = 0
    LTAP_all = 0
    opposite_direction_all = 0
    rear_end_all = 0

    lane_change_yaw_distance = 0
    junction_crossing_yaw_distance = 0
    LTAP_yaw_distance = 0
    opposite_direction_yaw_distance = 0
    rear_end_yaw_distance = 0

    col_lane_change_num = 0.0000001
    col_junction_crossing_num = 0.0000001
    col_LTAP_num = 0.0000001
    col_opposite_direction_num = 0.0000001
    col_rear_end_num = 0.0000001
    col_other_num = 0

    sim_lane_change_num = 0
    sim_junction_crossing_num = 0
    sim_LTAP_num = 0
    sim_opposite_direction_num = 0
    sim_rear_end_num = 0

    # 5 10 20 30 degree 
    degree_list = [5, 10, 20, 30]
    sim_lane_change_np = np.zeros(4)
    sim_junction_crossing_np = np.zeros(4)
    sim_LTAP_np = np.zeros(4)
    sim_opposite_direction_np = np.zeros(4)
    sim_rear_end_np = np.zeros(4)
    sim_all_scene_np = np.zeros(4)
    
    ####################################################
    interpolation_every_frame = True
    frame_multiple = 5
    split = 'test'
    ####################################################
    
    # for scenario_name in sorted(os.listdir(args.data_dest_path)):
    for scenario_name in tqdm(os.listdir(args.whole_csv_path)):
        # if scenario_name != 'data_hcis_v4.7_trainval_boston-seaport_HO_scene-0222_5-0-1.0-20_177.58_5b19f5b9423b4911b68cbae9ed21d9bb.csv':
        #     continue
        all_data_num += 1
        attacker_id = scenario_name.split('_')[-1].split('.')[0]
        
        # if all_data_num > 11:
        #     break
        print(scenario_name)
        
        traj_df = pd.read_csv(os.path.join(args.whole_csv_path + scenario_name))

         

        frame_num = len(set(traj_df.TIMESTAMP.values))
        # print(frame_num, traj_df)
        further_list = []
        further_df = traj_df
        inter_df = pd.DataFrame(columns=['TRACK_ID', 'TIMESTAMP', 'V', 'X', 'Y', 'YAW'])
        if interpolation_every_frame:
            for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                # E.g. 20 frames * 5 times => (20-1)*5+1 = 96 frames
                # print(remain_df)
                # for origin_f_index in range(1, frame_num + 1):
                for origin_f_index in range(1, frame_num):
                    # print(origin_f_index, remain_df)
                    # print(frame_num, origin_f_ttcindex-1, remain_df.iloc[origin_f_index-1, 3])
                    # last frame in original df => further seems to be add last frame
                    # print(origin_f_index, frame_num)
                    frame_multiple_for_loop = frame_multiple
                    if origin_f_index == frame_num:
                        frame_multiple_for_loop = 1
                        dis_x, dis_y = 0, 0
                    else:
                        dis_x = (remain_df.iloc[origin_f_index, 3] - remain_df.iloc[origin_f_index-1, 3]) / frame_multiple
                        dis_y = (remain_df.iloc[origin_f_index, 4] - remain_df.iloc[origin_f_index-1, 4]) / frame_multiple
                    # for those padding zero vehicle
                    if dis_x > 10 or dis_y > 10:
                        dis_x, dis_y = 0, 0 
                    # if track_id == 'f5df5ef1e5624a029ce64dd462556de5':
                        # print(dis_x, dis_y)
                    
                    
                    for fps_f_index in range(frame_multiple_for_loop):
                        # print(remain_df.iloc[0]['TIMESTAMP'])
                        # t = {'TRACK_ID':[track_id], 'TIMESTAMP':[remain_df.iloc[origin_f_index-1, 1] + fps_f_index * 500000 / frame_multiple],
                        #         'V':[remain_df.iloc[origin_f_index-1, 2]], 'X':[remain_df.iloc[origin_f_index-1, 3] + fps_f_index * dis_x],
                        #         'Y':[remain_df.iloc[origin_f_index-1, 4] + fps_f_index * dis_y], 'YAW':[remain_df.iloc[origin_f_index-1, 5]]}
                        t = {'TRACK_ID':[track_id], 'TIMESTAMP':[remain_df.iloc[origin_f_index-1, 1] + fps_f_index * 500000 / frame_multiple],
                                'V':[remain_df.iloc[origin_f_index-1, 2]], 'X':[remain_df.iloc[origin_f_index-1, 3] + fps_f_index * dis_x],
                                'Y':[remain_df.iloc[origin_f_index-1, 4] + fps_f_index * dis_y], 'YAW':[remain_df.iloc[origin_f_index, 5]]}
                        df_insert = pd.DataFrame(t)
                        inter_df = pd.concat([inter_df, df_insert], ignore_index=True)
            ### Interpolation for moving foward with higher FPS###
            for track_id, remain_df in inter_df.groupby("TRACK_ID"):
                more_frames = 4 * frame_multiple
                # trajectory may be a curve, so it should rely on last frame and the previous frame
                dis_x = (remain_df.iloc[(frame_num-1)*frame_multiple-1, 3] - remain_df.iloc[(frame_num-1)*frame_multiple-2, 3])
                dis_y = (remain_df.iloc[(frame_num-1)*frame_multiple-1, 4] - remain_df.iloc[(frame_num-1)*frame_multiple-2, 4])
                all_x, all_y, all_v, all_yaw = [], [], [], []
                for f_index in range(more_frames):
                    all_v.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 2])
                    all_x.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 3] + dis_x * (f_index + 1))
                    all_y.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 4] + dis_y * (f_index + 1))
                    all_yaw.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 5])
                for further_t in range(more_frames):
                    b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * 500000 / frame_multiple], 'TRACK_ID': [track_id],
                        # 'V': [all_v[further_t]ttc], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
                        'V': [all_v[further_t]], 'X': [all_x[further_t]], 'Y': [all_y[further_t]],
                        'YAW': [all_yaw[further_t]]}
                    df_insert = pd.DataFrame(b)
                    inter_df = pd.concat([inter_df, df_insert], ignore_index=True)
            ### Interpolation for moving foward with higher FPS###
        
        # new_attacker_id = str(int(attacker_id[:5], 16))
        # print(new_attacker_id, traj_df.TRACK_ID.values)
        further_df = inter_df

        if not os.path.isdir(args.forward_csv_path):
            os.makedirs(args.forward_csv_path)
        further_df.to_csv(args.forward_csv_path + scenario_name[:-4] + '_further.csv', index=False)
        
        collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = cal_cr_and_similarity(further_df, attacker_id)

        pred_collision_rate = 0
        attacker_right_collision_rate = 0
        real_yaw_dist_average = 0
        condition = scenario_name.split('_')[5]
        if condition == 'LTAP':
            LTAP_all += 1
            ideal_yaw_offset = 90
        elif condition == 'JC':
            junction_crossing_all += 1
            ideal_yaw_offset = 90
        elif condition == 'HO':
            opposite_direction_all += 1
            ideal_yaw_offset = 180
        elif condition == 'RE':
            rear_end_all += 1
            ideal_yaw_offset = 0
        elif condition == 'LC':
            lane_change_all += 1
            # ideal_yaw_offset = 15
            ideal_yaw_offset = 20
        gt_record_yaw_distance = float(scenario_name.split('_')[8])

        if collision_flag:
            pred_collision_rate += 1
            
            while record_yaw_distance < 0:
                record_yaw_distance = (record_yaw_distance + 360.0)
            record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance

            # metric only calculate on GT collision
            if attacker_right_flag:
                attacker_right_collision_rate += 1
                
                # yaw_distance = abs(ideal_yaw_offset - record_yaw_distance)
                yaw_distance = abs(gt_record_yaw_distance - record_yaw_distance)
                
                real_yaw_dist_average += yaw_distance
                
                ##### Check JC, LTAP in the intersection #####
                check_intersection = False
                if check_intersection:
                    if condition == 'JC' or condition == 'LTAP':
                        LTAP_intersection_flag = 0
                        JC_intersection_flag = 0
                        intersection_threshold = 20
                        ego_list = []
                        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                            if str(track_id) == 'ego':
                                ego_list.append(remain_df.reset_index())
                        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                            if track_id == attacker_id:
                                for i in range(frame_num):
                                    x = remain_df.iloc[i, 3]
                                    y = remain_df.iloc[i, 4]
                                    yaw = remain_df.iloc[i, 5]
                                    # print(abs(yaw - ego_list[0].loc[i, 'YAW']), is_intersection(x, y, town_name=scenario.split('_')[4]))
                                    if is_intersection(x, y, town_name=scenario_name.split('_')[4]):
                                        continue
                                    diff_yaw = abs(yaw - ego_list[0].loc[i, 'YAW'])
                                    if abs(diff_yaw - 180) < intersection_threshold:
                                        LTAP_intersection_flag = 1
                                    elif abs(diff_yaw - 90) < intersection_threshold:
                                        JC_intersection_flag = 1
                else:
                    LTAP_intersection_flag = 1
                    JC_intersection_flag = 1
                                

                if condition == 'LTAP':
                    LTAP_yaw_distance += yaw_distance
                    col_LTAP_num += 1
                    if yaw_distance < 30:
                        sim_LTAP_num += 1
                    for degree_index in range(len(degree_list)):
                        if yaw_distance < degree_list[degree_index] and LTAP_intersection_flag:
                            sim_LTAP_np[degree_index] += 1
                elif condition == 'JC':
                    junction_crossing_yaw_distance += yaw_distance
                    col_junction_crossing_num += 1
                    if yaw_distance < 30:
                        sim_junction_crossing_num += 1
                    for degree_index in range(len(degree_list)):
                        if yaw_distance < degree_list[degree_index] and JC_intersection_flag:
                            sim_junction_crossing_np[degree_index] += 1
                elif condition == 'HO':
                    opposite_direction_yaw_distance += yaw_distance
                    col_opposite_direction_num += 1
                    if yaw_distance < 30:
                        sim_opposite_direction_num += 1
                    for degree_index in range(len(degree_list)):
                        if yaw_distance < degree_list[degree_index]:
                            sim_opposite_direction_np[degree_index] += 1
                elif condition == 'RE':
                    rear_end_yaw_distance += yaw_distance
                    col_rear_end_num += 1
                    if yaw_distance < 30:
                        sim_rear_end_num += 1
                    for degree_index in range(len(degree_list)):
                        if yaw_distance < degree_list[degree_index]:
                            sim_rear_end_np[degree_index] += 1
                elif condition == 'LC':
                    lane_change_yaw_distance += yaw_distance
                    col_lane_change_num += 1
                    if yaw_distance < 30:
                        sim_lane_change_num += 1
                    for degree_index in range(len(degree_list)):
                        if yaw_distance < degree_list[degree_index]:
                            sim_lane_change_np[degree_index] += 1
            else:
                col_other_num += 1
    # type_name_list = ["LC", "HO", "RE", "JC", "LTAP", "All", "-"]
    
    # all_right_col_num = col_LTAP_num + col_junction_crossing_num + col_lane_change_num + col_opposite_direction_num + col_rear_end_num
    
    # for degree_i in range(len(degree_list)):
    #     sim_all_scene_np[degree_i] = round((sim_lane_change_np[degree_i] + sim_opposite_direction_np[degree_i] + sim_rear_end_np[degree_i] + \
    #                                         sim_junction_crossing_np[degree_i] + sim_LTAP_np[degree_i]) / all_right_col_num, 2)
    #     sim_lane_change_np[degree_i] = round(sim_lane_change_np[degree_i] / col_lane_change_num, 2)
    #     sim_opposite_direction_np[degree_i] = round(sim_opposite_direction_np[degree_i] / col_opposite_direction_num, 2)
    #     sim_rear_end_np[degree_i] = round(sim_rear_end_np[degree_i] / col_rear_end_num, 2)
    #     sim_junction_crossing_np[degree_i] = round(sim_junction_crossing_np[degree_i] / col_junction_crossing_num, 2)
    #     sim_LTAP_np[degree_i] = round(sim_LTAP_np[degree_i] / col_LTAP_num, 2)        
    # type_sim_list = [sim_lane_change_np, sim_opposite_direction_np, sim_rear_end_np, sim_junction_crossing_np, sim_LTAP_np, sim_all_scene_np]
    # # print(lane_change_all, rear_end_all, opposite_direction_all, LTAP_all, junction_crossing_all)
    # type_cr_list = [str(round(col_lane_change_num / lane_change_all, 2)), str(round(col_opposite_direction_num / opposite_direction_all, 2)),
    #                 str(round(col_rear_end_num / rear_end_all, 2)), str(round(col_junction_crossing_num / junction_crossing_all, 2)),
    #                     str(round(col_LTAP_num / LTAP_all, 2)), str(round((col_lane_change_num + col_opposite_direction_num + col_rear_end_num + col_junction_crossing_num + col_LTAP_num) / all_data_num, 2))]
    # type_yaw_list = [str(round(lane_change_yaw_distance / col_lane_change_num, 2)), str(round(opposite_direction_yaw_distance / col_opposite_direction_num, 2)),
    #                 str(round(rear_end_yaw_distance / col_rear_end_num, 2)), str(round(junction_crossing_yaw_distance / col_junction_crossing_num, 2)),
    #                     str(round(LTAP_yaw_distance / col_LTAP_num, 2)), str(round((lane_change_yaw_distance + opposite_direction_yaw_distance + rear_end_yaw_distance + junction_crossing_yaw_distance + LTAP_yaw_distance) / all_right_col_num, 2))]
    # csv_file = split + '_cr_sim.csv'
    # if not os.path.exists(csv_file):
    #     with open(csv_file, 'a+') as f:
    #         writer = csv.writer(f)
    #         # writer.writerow(['Time', 'LR', 'Adam_weight_decay', 'Ego iou>0.5', 'FDE', 'Ego yaw', 'TP iou>0.5', 'TP DE', 'TP yaw', 'TP ID', 'CR', 'Sim'])
    #         writer.writerow(['Time', 'Type', 'CR', 'Sim(5)', 'Sim(10)', 'Sim(20)', 'Sim(30)', 'Yaw dist']) # RCNN version
    #         f.close()
    # with open(csv_file, 'a+') as f:
    #     writer = csv.writer(f)
    #     # writer.writerow([save_folder.split('/')[-1], self.lr, self.weight_decay, ego_iou_50 / all_data_num, sum_fde / all_data_num, all_yaw_distance / all_data_num, 
    #     #                  attacker_iou_50 / all_data_num, sum_attacker_de / all_data_num, ideal_yaw_dist_average / all_data_num,
    #     #                  all_tp_cls_acc / all_data_num, cr, Average_Similarity])

    #     for type_index in range(len(type_name_list)):
    #         if type_index == len(type_name_list) - 1:
    #             writer.writerow(['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'])
    #         else:
    #             writer.writerow([args.folder_name, type_name_list[type_index],
    #                             type_cr_list[type_index], type_sim_list[type_index][0], type_sim_list[type_index][1], type_sim_list[type_index][2], type_sim_list[type_index][3], type_yaw_list[type_index]])  # RCNN version
    #     f.close()

def cal_cr_and_similarity(traj_df, attacker_id_gt):
    collision_flag = 0
    right_attacker_flag = 0
    real_yaw_distance = -999
    record_yaw_distance = -999
    vehicle_list = []
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        vehicle_list.append(remain_df)
    ego_list = []
    attacker_list = []
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if str(track_id) == 'ego':
            ego_list.append(remain_df.reset_index())
        elif str(track_id) == attacker_id_gt:
            attacker_list.append(remain_df.reset_index())
    # print(traj_df)

    scenario_length = len(vehicle_list[0])
    for t in range(1, scenario_length+1):
        ego_x = ego_list[0].loc[t - 1, 'X']
        # ego_x_next = ego_list[0].loc[t, 'X']
        ego_y = ego_list[0].loc[t - 1, 'Y']
        # ego_y_next = ego_list[0].loc[t, 'Y']
        # ego_vec = [ego_y_next - ego_y,
        #                     ego_x_next - ego_x]
        # ego_angle = np.rad2deg(angle_vectors(ego_vec, [1, 0])) * np.pi / 180
        # real_ego_angle = ego_list[0].loc[t - 1, 'YAW']
        real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
        real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
        ego_rec = [ego_x, ego_y, vehicle_width, vehicle_length, real_ego_angle]
        x_1 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_2 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_3 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        x_4 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        y_1 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_2 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_3 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        y_4 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
        ego_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
        
        plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='pink', alpha=0.5)
        # plt.plot([x_1, x_2, x_4, x_3, x_1], [
        #                             y_1, y_2, y_4, y_3, y_1], '-',  color='lime', markersize=3)
        attacker_x = attacker_list[0].loc[t-1, 'X']
        attacker_y = attacker_list[0].loc[t-1, 'Y']
        # attacker_x_next = attacker_list[0].loc[t, 'X']
        # attacker_y_next = attacker_list[0].loc[t, 'Y']
        real_attacker_angle = attacker_list[0].loc[t - 1, 'YAW'] + 360.0 if attacker_list[0].loc[t - 1, 'YAW'] < 0 else attacker_list[0].loc[t - 1, 'YAW']
        real_attacker_angle = (real_attacker_angle + 90.0) * np.pi / 180
        # ego_rec = [attacker_x_next, attacker_y_next, self.vehicle_width, self.vehicle_length, real_attacker_angle]
        ego_rec = [attacker_x, attacker_y, vehicle_width, vehicle_length, real_attacker_angle]
        x_1 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_2 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_3 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        x_4 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        y_1 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_2 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_3 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        y_4 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        attacker_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
        for n in range(len(vehicle_list)):
            vl = vehicle_list[n].to_numpy()
            # vl : frame, id, x, y
            # => id, frame, v, x, y, yaw(arc)
            now_id = vl[0][0]
            if str(now_id) == 'ego':
                continue
            real_pred_x = vl[t - 1][3]
            # real_pred_x_next = vl[t][3]
            real_pred_y = vl[t - 1][4]
            # real_pred_y_next = vl[t][4]
            # other_vec = [real_pred_y_next - real_pred_y,
                                    # real_pred_x_next - real_pred_x]
            # other_angle = np.rad2deg(angle_vectors(other_vec, [1, 0])) * np.pi / 180
            real_other_angle = (vl[t - 1][5] + 90.0) * np.pi / 180
            # other_angle = vl[past_len][4]
            # ego_angle = ego_list[0][4][int(filename_t) + past_len]
            #print(ego_x, ego_y, real_pred_x, real_pred_y)
            ego_rec = [real_pred_x, real_pred_y, vehicle_width, vehicle_length, real_other_angle]
            # ego_rec = [real_pred_x_next, real_pred_y_next, self.vehicle_width
            #                             , self.vehicle_length, other_angle]
            
            x_1 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_2 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_3 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            x_4 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            y_1 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_2 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_3 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            y_4 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            other_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
            other_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
            # plt.plot([x_1, x_2, x_4, x_3, x_1], [
            #                         y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
            if str(now_id) != str(attacker_id_gt):
                attacker_other_iou = attacker_polygon.intersection(other_polygon).area / attacker_polygon.union(other_polygon).area
                
                if attacker_other_iou > VEH_COLL_THRESH:
                    print("attacker early hit")
                    #print(t, now_id, "other:", attacker_id_gt, real_pred_x, real_pred_y, real_other_angle, "GT attacker:", ego_x, ego_y, real_ego_angle)

                    collision_flag = 2
                    # print(now_id, attacker_other_iou, real_pred_x_next, real_pred_y_next)
            cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
            # print(t, now_id, ego_polygon.intersection(other_polygon).area, cur_iou, "GT:", attacker_id_gt, "flag:", collision_flag)
            
            if cur_iou > VEH_COLL_THRESH:
                # print(attacker_id_gt, "COLLIDE!", now_id)
                collision_flag = 1
                if str(now_id) == str(attacker_id_gt):
                    # plt.close()
                    # fig,ax = plt.subplots()
                    # ax.add_patch(ego_pg)
                    # ax.add_patch(other_pg)
                    # ax.set_xlim([1821,1835])
                    # ax.set_ylim([2529,2544])
                    #plt.show()

                    #print("COLLIDE! GT!!!!!!!! ", cur_iou)
                    
                    right_attacker_flag = 1
                    # Must collide with GT attacker
                    
                    # real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
                    real_yaw_distance = None
                    record_yaw_distance = (real_ego_angle - real_other_angle) * 180 / np.pi
                    #print(record_yaw_distance)
                else:
                    plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='green', alpha=1)
                    #plt.plot([x_1, x_2, x_4, x_3, x_1], [
                    #                 y_1, y_2, y_4, y_3, y_1], '-',  color='violet', markersize=3)

                
            if collision_flag:
                break
        if collision_flag:
            break
    return collision_flag, real_yaw_distance, right_attacker_flag, record_yaw_distance
        
# def cal_cr_and_similarity_cal_yaw(traj_df, attacker_id_gt):
#     collision_flag = 0
#     right_attacker_flag = 0
#     real_yaw_distance = -999
#     record_yaw_distance = -999
#     vehicle_list = []
#     for track_id, remain_df in traj_df.groupby('TRACK_ID'):
#         vehicle_list.append(remain_df)
#     ego_list = []
#     attacker_list = []
#     for track_id, remain_df in traj_df.groupby('TRACK_ID'):
#         if str(track_id) == 'ego':
#             ego_list.append(remain_df.reset_index())
#         elif str(track_id) == attacker_id_gt:
#             attacker_list.append(remain_df.reset_index())
#     # print(traj_df)

#     scenario_length = len(vehicle_list[0])
#     for t in range(1, scenario_length+1):
#         ego_x = ego_list[0].loc[t - 1, 'X']
#         # ego_x_next = ego_list[0].loc[t, 'X']
#         ego_y = ego_list[0].loc[t - 1, 'Y']
#         # ego_y_next = ego_list[0].loc[t, 'Y']
#         # ego_vec = [ego_y_next - ego_y,
#         #                     ego_x_next - ego_x]
#         # ego_angle = np.rad2deg(angle_vectors(ego_vec, [1, 0])) * np.pi / 180
#         # real_ego_angle = ego_list[0].loc[t - 1, 'YAW']
#         real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
#         real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
#         ego_rec = [ego_x, ego_y, vehicle_width, vehicle_length, real_ego_angle]
#         x_1 = float(np.cos(
#             ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
#         x_2 = float(np.cos(
#             ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
#         x_3 = float(np.cos(
#             ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
#         x_4 = float(np.cos(
#             ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
#         y_1 = float(np.sin(
#             ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
#         y_2 = float(np.sin(
#             ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
#         y_3 = float(np.sin(
#             ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
#         y_4 = float(np.sin(
#             ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
#         ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
#         ego_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
        
#         plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='pink', alpha=0.5)
#         # plt.plot([x_1, x_2, x_4, x_3, x_1], [
#         #                             y_1, y_2, y_4, y_3, y_1], '-',  color='lime', markersize=3)
#         attacker_x = attacker_list[0].loc[t-1, 'X']
#         attacker_y = attacker_list[0].loc[t-1, 'Y']


#         if t < scenario_length:
#             attacker_x_next = attacker_list[0].loc[t, 'X']
#             attacker_y_next = attacker_list[0].loc[t, 'Y']
#             att_vec = [attacker_y_next - attacker_y,
#                             attacker_x_next - attacker_x]
#         real_attacker_angle = (np.rad2deg(angle_vectors(att_vec, [1, 0])) + 90.0) * np.pi / 180
#         # real_attacker_angle = attacker_list[0].loc[t - 1, 'YAW'] + 360.0 if attacker_list[0].loc[t - 1, 'YAW'] < 0 else attacker_list[0].loc[t - 1, 'YAW']
#         # real_attacker_angle = (real_attacker_angle + 90.0) * np.pi / 180
        
        
#         # ego_rec = [attacker_x_next, attacker_y_next, self.vehicle_width, self.vehicle_length, real_attacker_angle]
#         ego_rec = [attacker_x, attacker_y, vehicle_width, vehicle_length, real_attacker_angle]
#         x_1 = float(np.cos(
#             ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
#         x_2 = float(np.cos(
#             ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
#         x_3 = float(np.cos(
#             ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
#         x_4 = float(np.cos(
#             ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
#         y_1 = float(np.sin(
#             ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
#         y_2 = float(np.sin(
#             ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
#         y_3 = float(np.sin(
#             ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
#         y_4 = float(np.sin(
#             ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
#         attacker_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
#         for n in range(len(vehicle_list)):
#             vl = vehicle_list[n].to_numpy()
#             # vl : frame, id, x, y
#             # => id, frame, v, x, y, yaw(arc)
#             now_id = vl[0][0]
#             if str(now_id) == 'ego':
#                 continue
#             real_pred_x = vl[t - 1][3]
#             # real_pred_x_next = vl[t][3]
#             real_pred_y = vl[t - 1][4]
#             # real_pred_y_next = vl[t][4]
#             # other_vec = [real_pred_y_next - real_pred_y,
#                                     # real_pred_x_next - real_pred_x]
#             # other_angle = np.rad2deg(angle_vectors(other_vec, [1, 0])) * np.pi / 180
            
            
#             if t < scenario_length:
#                 real_pred_x_next = vl[t][3]
#                 real_pred_y_next = vl[t][4]
#                 other_vec = [real_pred_y_next - real_pred_y,
#                                 real_pred_x_next - real_pred_x]
#             real_other_angle = (np.rad2deg(angle_vectors(other_vec, [1, 0])) + 90.0) * np.pi / 180
#             # if not moving (parking car)
#             if np.abs(real_pred_x_next - real_pred_x) < 0.01 and np.abs(real_pred_y_next - real_pred_y) < 0.01:
#                 real_other_angle = (vl[t - 1][5] + 90.0) * np.pi / 180

            
            
#             # real_other_angle = (vl[t - 1][5] + 90.0) * np.pi / 180
#             # other_angle = vl[past_len][4]
#             # ego_angle = ego_list[0][4][int(filename_t) + past_len]
#             #print(ego_x, ego_y, real_pred_x, real_pred_y)
#             ego_rec = [real_pred_x, real_pred_y, vehicle_width, vehicle_length, real_other_angle]
#             # ego_rec = [real_pred_x_next, real_pred_y_next, self.vehicle_width
#             #                             , self.vehicle_length, other_angle]
            
#             x_1 = float(np.cos(
#                 ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
#             x_2 = float(np.cos(
#                 ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
#             x_3 = float(np.cos(
#                 ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
#             x_4 = float(np.cos(
#                 ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
#             y_1 = float(np.sin(
#                 ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
#             y_2 = float(np.sin(
#                 ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
#             y_3 = float(np.sin(
#                 ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
#             y_4 = float(np.sin(
#                 ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
#             other_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
#             other_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
#             # plt.plot([x_1, x_2, x_4, x_3, x_1], [
#             #                         y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
#             if str(now_id) != str(attacker_id_gt):
#                 attacker_other_iou = attacker_polygon.intersection(other_polygon).area / attacker_polygon.union(other_polygon).area
                
#                 if attacker_other_iou > VEH_COLL_THRESH:
#                     print("attacker early hit")
#                     #print(t, now_id, "other:", attacker_id_gt, real_pred_x, real_pred_y, real_other_angle, "GT attacker:", ego_x, ego_y, real_ego_angle)

#                     collision_flag = 2
#                     # print(now_id, attacker_other_iou, real_pred_x_next, real_pred_y_next)
#             cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
#             # print(t, now_id, ego_polygon.intersection(other_polygon).area, cur_iou, "GT:", attacker_id_gt, "flag:", collision_flag)
            
#             if cur_iou > VEH_COLL_THRESH:
#                 # print(attacker_id_gt, "COLLIDE!", now_id)
#                 collision_flag = 1
#                 if str(now_id) == str(attacker_id_gt):
#                     # plt.close()
#                     # fig,ax = plt.subplots()
#                     # ax.add_patch(ego_pg)
#                     # ax.add_patch(other_pg)
#                     # ax.set_xlim([1821,1835])
#                     # ax.set_ylim([2529,2544])
#                     #plt.show()

#                     #print("COLLIDE! GT!!!!!!!! ", cur_iou)
                    
#                     right_attacker_flag = 1
#                     # Must collide with GT attacker
                    
#                     # real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
#                     real_yaw_distance = None
#                     record_yaw_distance = (real_ego_angle - real_other_angle) * 180 / np.pi
#                     #print(record_yaw_distance)
#                 else:
#                     plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='green', alpha=1)
#                     #plt.plot([x_1, x_2, x_4, x_3, x_1], [
#                     #                 y_1, y_2, y_4, y_3, y_1], '-',  color='violet', markersize=3)

                
#             if collision_flag:
#                 break
#         if collision_flag:
#             break
#     return collision_flag, real_yaw_distance, right_attacker_flag, record_yaw_distance

def plot_animation(args):
    past_len = 8
    interpolation_frame = 5
    folder = args.forward_csv_path
    sav_folder = args.animation_path
    map_path = args.map_path
    for scenario_name in sorted(os.listdir(folder)):
        town_name = scenario_name.split('_')[4]
        split_name = scenario_name.split('_')
        # if scenario_name != 'data_hcis_v4.6_trainval_boston-seaport_LTAP_scene-0594_2-2-0.8333333333333334-18_97.87_831b2f71e061472ea179506eb8c49914_foward.csv':
        #     continue
        # # # TTC normalized has changed # # #
        # if scenario_name != 'data_hcis_v4.6_trainval_boston-seaport_LTAP_scene-0594_2-2-1.0-18_97.87_831b2f71e061472ea179506eb8c49914.csv':
        #     continue

        # if split_name[6].split('-')[1] != '0660':
        #    continue
        print(scenario_name)
        dir_name = split_name[5] + '_' + split_name[6] + '_' + split_name[7] + '_' + split_name[8] + '_' + split_name[9]
        sav_path = sav_folder + dir_name +  '/'
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)

        traj_df = pd.read_csv(os.path.join(
            folder + scenario_name))
        ego_list = []
        risky_vehicle_list = []
        angle_list = []
        flag = 0            
        vehicle_list = []
        fill_dict = {}
        collision_flag = 0
        right_attacker_flag = 0
        real_yaw_distance = -999
        record_yaw_distance = -999
        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
            vehicle_list.append(remain_df)
            
            
            # fig, ax = plt.subplots()
            # ax.set_xlabel("x axis(m)")
            # ax.set_ylabel("y axis(m)")

        
        d = dict()
        d['scenario_id'] = scenario_name
        split_name = scenario_name.split('_')
        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
        
        # lane_feature = np.load(topo_folder + initial_name + '.npy', allow_pickle=True)
        for n in range(len(vehicle_list)):
            vl = vehicle_list[n].to_numpy()
            now_id = vl[0][0]
            data_length = vl.shape[0]
            if now_id == "ego":
                forever_present_x = vl[-1][3]
                forever_present_y = vl[-1][4]
        attacker_id = scenario_name.split('_')[9].split('.')[0]

        ##############
        # attacker_id = str(int(attacker_id[:5], 16))
        ##############
        
        for track_id, remain_df in traj_df.groupby('TRACK_ID'):
            fill_dict[track_id] = []
            if str(track_id) == 'ego':
                ego_list.append(remain_df.reset_index())
        scenario_length = len(vehicle_list[0])

        for t in range(1, scenario_length + 1):
            nusc_map = NuScenesMap(dataroot=map_path, map_name=town_name)
            fig, ax = nusc_map.render_layers(["drivable_area"])
            print(initial_name, t)
            # plt.figure(dpi=500)
            # plt.xlabel("x axis(m)")
            # plt.ylabel("y axis(m)")
            black_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='darkgray', edgecolor='black', label='Agents')
            black_legend = ax.legend(handles=[black_patch], loc='upper left', bbox_to_anchor=(0.02, 0.98))
            for text in black_legend.get_texts():
                text.set_fontsize(20)
            ax.add_artist(black_legend)
            ego_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='red', edgecolor='black', label='Ego')
            ego_legend = ax.legend(handles=[ego_patch], loc='upper left', bbox_to_anchor=(0.02, 0.95))
            for text in ego_legend.get_texts():
                text.set_fontsize(20)
            ax.add_artist(ego_legend)
            att_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='blue',edgecolor='black', label='Attacker')
            att_legend = ax.legend(handles=[att_patch], loc='upper left', bbox_to_anchor=(0.02, 0.92))
            for text in att_legend.get_texts():
                text.set_fontsize(20)
            
            # ego, = ax.plot(
            #     [0, 0], label="detected collision", color='red', linestyle='--')
            # first_legend = ax.legend(
            #     handles=[ego], loc='lower left', bbox_to_anchor=(0.6, 0.3))
            # ax.add_artist(first_legend)
            # black_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
            #                                     edgecolor='black', label='agents')
            # black_legend = ax.legend(
            #     handles=[black_patch], loc='lower left', bbox_to_anchor=(0.6, 0.2))
            # ax.add_artist(black_legend)
            # darkgreen_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
            #                                         edgecolor='lime', label='ego')
            # darkgreen_legend = ax.legend(
            #     handles=[darkgreen_patch], loc='lower left', bbox_to_anchor=(0.6, 0.1))
            # ax.add_artist(darkgreen_legend)
            # purple_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
            #                                     edgecolor='violet', label='risk')
            # ax.legend(
            #     handles=[purple_patch], loc='lower left', bbox_to_anchor=(0.6, 0))
        
            
            # ego, = ax.plot([0, 0], [0, 0], '--o',
            #                 color='blue', markersize=1)
            # agent, = ax.plot([0, 0], [0, 0], '--o',
            #                 color='red', markersize=1)
            # vehicle, = ax.plot([0, 0], [0, 0], '--o',
            #                     color='green', markersize=1)
            # ax.legend([ego, agent, vehicle], [
            #         "ego", "attacker", "vehicle"])

            # for features in lane_feature:
            #     xs, ys = np.vstack((features[0][:, :2], features[0][-1, 3:5]))[
            #         :, 0], np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 1]
            #     plt.plot(xs, ys, '-.', color='lightgray')
            #     x_s, y_s = np.vstack((features[1][:, :2], features[1][-1, 3:5]))[
            #         :, 0], np.vstack((features[1][:, :2], features[1][-1, 3:5]))[:, 1]
            #     plt.plot(x_s, y_s, '-.', color='lightgray')
            
            # ego_fill_bewteen_np = np.zeros((len(ego_list[0]),2))
            
            
            
            ego_x = ego_list[0].loc[t - 1, 'X']
            # ego_x_next = ego_list[0].loc[t, 'X']
            ego_y = ego_list[0].loc[t - 1, 'Y']
            # ego_y_next = ego_list[0].loc[t, 'Y']
            # ego_vec = [ego_y_next - ego_y,ego_x_next - ego_x]
            # ego_angle = np.rad2deg(angle_vectors(ego_vec, [1, 0])) * np.pi / 180
            real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
            real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
            #### real_ego_angle
            # ego_rec = [ego_x_next, ego_y_next, vehicle_width, vehicle_length, ego_angle]
            ego_rec = [ego_x, ego_y, vehicle_width, vehicle_length, real_ego_angle]
            
            x_1 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_2 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_3 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            x_4 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            y_1 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_2 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_3 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            y_4 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            ego_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
            ego_polygon = Polygon(ego_coords)
            # ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
            # ego_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
            if t <= past_len * interpolation_frame:
                ego_color = 'lightcoral'
            else:
                ego_color = 'red'

            ########### trajectory ###########
            # (x_1, y_1) ---- (x_2, y_2)
            #     |                |
            #     |                |
            # (x_3, y_3) ---- (x_4, y_4)
            # gradient = np.linspace(0, 1, 100)
            # colors_gradient = plt.cm.viridis(gradient)
            # plt.fill_between([ego_x_next], [ego_y, ego_y_next], color=colors_gradient, label='Fill Between Gradient')
            # plt.fill_between([ego_y_next], [ego_x, ego_x_next], color=colors_gradient, label='Fill Between Gradient')
            ########### trajectory ###########

            # ego_y_list = [841, 842, 843]
            # left_x_list = [1702, 1703, 1704]
            # right_x_list = [1706, 1707, 1708]
            # plt.fill_betweenx(ego_y, left_x, right_x, color='gray', alpha=0.5)
            # plt.fill_between(ego_y_list, left_x_list, right_x_list, color='blue', alpha=0.5)
            
            ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
            ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=ego_color, alpha=1)
            fill_dict['ego'].append(np.array([[x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]]))
            
            for n in range(len(vehicle_list)):
                vl = vehicle_list[n].to_numpy()
                # vl : frame, id, x, y
                # => id, frame, v, x, y, yaw(arc)
                now_id = vl[0][0]
                if str(now_id) == 'ego':
                    continue
                real_pred_x = vl[t - 1][3]
                # real_pred_x_next = vl[t][3]
                real_pred_y = vl[t - 1][4]
                # real_pred_y_next = vl[t][4]
                # other_vec = [real_pred_y_next - real_pred_y,real_pred_x_next - real_pred_x]
                # other_angle = np.rad2deg(angle_vectors(other_vec, [1, 0])) * np.pi / 180
                real_other_angle = (vl[t - 1][5] + 90.0)
                # 90 or 180 => tend to wrong
                # check_angle = real_other_angle + 360.0 if real_other_angle < 0 else real_other_angle
                # if abs(abs(check_angle) - 90.0) or abs(abs(check_angle) - 270.0) < 10:
                #     real_other_angle = 90
                # elif abs(abs(check_angle) - 180.0) or abs(abs(check_angle) - 360.0) < 10:
                #     real_other_angle = 0                
                real_other_angle = real_other_angle * np.pi / 180
                # other_angle = vl[past_len][4]
                # ego_angle = ego_list[0][4][int(filename_t) + past_len]
                #print(ego_x, ego_y, real_pred_x, real_pred_y)
                # ego_rec = [real_pred_x_next, real_pred_y_next, vehicle_width, vehicle_length, other_angle]
                ego_rec = [real_pred_x, real_pred_y, vehicle_width, vehicle_length, real_other_angle]
                x_1 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_2 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_3 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                x_4 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                y_1 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_2 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_3 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                y_4 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                other_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
                other_polygon = Polygon(other_coords)
                # other_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
                # other_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
                
                ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                fill_dict[now_id].append(np.array([[x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]]))
                # print(now_id, attacker_id)
                # if now_id == '29c91f1d2509452f9a07386af12f06ec':
                #     print(t, now_id, "GT:", attacker_id, real_pred_x, real_pred_y, real_other_angle, "EGO:", ego_x, ego_y, real_ego_angle)

                if now_id == attacker_id:
                    # print("attacker id")
                    # print(ego_x, ego_y, real_pred_x, real_pred_y, attacker_id)
                    if t <= past_len * interpolation_frame:
                        attacker_color = 'cyan'
                    else:
                        attacker_color = 'blue'
                    ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=attacker_color, alpha=0.5)
                else:
                    ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='darkgray', alpha=0.5)
                
                cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
                #print(t, now_id, ego_polygon.intersection(other_polygon).area, "GT:", attacker_id_gt)
                
                if cur_iou > VEH_COLL_THRESH:
                    print(attacker_id, "COLLIDE!", now_id)
                    ax.plot([real_pred_x, ego_x], [
                            real_pred_y, ego_y], '-.', color='red', markersize=1)
                    collision_flag = 1
                    if str(now_id) == str(attacker_id):
                        # plt.close()
                        # fig,ax = plt.subplots()
                        # ax.add_patch(ego_pg)
                        # ax.add_patch(other_pg)
                 
                        right_attacker_flag = 1
                        # real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
                        record_yaw_distance = real_ego_angle - real_other_angle
                        #print(record_yaw_distance)
            
            # Plot Past Trajectory
            if t > 1:
                for n in range(len(vehicle_list)):
                    vl = vehicle_list[n].to_numpy()
                    now_id = vl[0][0]
                    if str(now_id) == attacker_id:
                        if t <= past_len * interpolation_frame:
                            fill_color = 'cyan'
                        else:
                            fill_color = 'blue'
                    elif now_id == 'ego':
                        if t <= past_len * interpolation_frame:
                            fill_color = 'lightcoral'
                        else:
                            fill_color = 'red'
                    else:
                        fill_color = 'darkgray'
                    for fill_t in range(t):
                        alpha_value = 0 if 1 - (t - fill_t - 1) * 0.2 <= 0 else 1 - (t - fill_t - 1) * 0.2
                        # print(alpha_value, fill_dict[now_id][fill_t])
                        ax.fill(fill_dict[now_id][fill_t][0], fill_dict[now_id][fill_t][1], '-',  color=fill_color, alpha=alpha_value)
                        # ax.fill_between(fill_dict[now_id][fill_t][0], fill_dict[now_id][fill_t][1], '-',  color=fill_color, alpha=alpha_value)
                        # if concat
                        if t > 0 and (fill_t + 1) != t:
                            # print(fill_dict)
                            # exit()
                            front_point_left_x = fill_dict[now_id][fill_t][0][0]
                            front_point_left_y = fill_dict[now_id][fill_t][1][0]
                            front_point_right_x = fill_dict[now_id][fill_t][0][1]
                            front_point_right_y = fill_dict[now_id][fill_t][1][1]
                            back_point_left_x_next_t = fill_dict[now_id][fill_t + 1][0][2]
                            back_point_left_y_next_t = fill_dict[now_id][fill_t + 1][1][2]
                            back_point_right_x_next_t = fill_dict[now_id][fill_t + 1][0][3]
                            back_point_right_y_next_t = fill_dict[now_id][fill_t + 1][1][3]
                            # concat_x = [front_point_left_x, front_point_right_x, back_point_right_x_next_t, back_point_left_x_next_t]
                            # concat_y = [front_point_left_y, front_point_right_y, back_point_right_y_next_t, back_point_left_y_next_t]
                            concat_x = [front_point_left_x, front_point_right_x, back_point_left_x_next_t, back_point_right_x_next_t]
                            concat_y = [front_point_left_y, front_point_right_y, back_point_left_y_next_t, back_point_right_y_next_t]
                            # print(now_id, concat_x, concat_y)
                            not_detected_vehicle = abs(front_point_left_x) < 10 and abs(front_point_left_y) < 10 and abs(front_point_right_x) < 10 and abs(front_point_right_y)  < 10
                            # print(not abs(front_point_left_x) < 10 and abs(front_point_left_y) < 10 and abs(front_point_right_x) < 10 and abs(front_point_right_y)  < 10)
                            if not not_detected_vehicle:
                                ax.fill(concat_x, concat_y, '-',  color=fill_color, alpha=alpha_value)

                    
                
            ax.set_xlim(forever_present_x - 50,
                    forever_present_x + 50)
            ax.set_ylim(forever_present_y - 50,
                    forever_present_y + 50)
            # ax.set_xlim(forever_present_x - 100,
            #         forever_present_x + 100)
            # ax.set_ylim(forever_present_y - 100,
            #         forever_present_y + 100)
            # ax.set_clip_box([[forever_present_x - 50, forever_present_y - 50], [forever_present_x + 50, forever_present_y]])
            
            if right_attacker_flag:
                right_attacker_flag_str = "Collide with attacker!"
            else:
                right_attacker_flag_str = "Safe!"
            title = right_attacker_flag_str
            
            ax.set_title(title, fontsize=20)
            print(title)
            # sav_path = sav_folder + dir_name + '_' + right_attacker_flag_str +  '/'
            # if not os.path.exists(sav_path):
            #                     os.makedirs(sav_path)
            fig.savefig(sav_path + str(t) + '.png')
            # plt.close(fig)
            plt.cla()
            
            #     if collision_flag:
            #         break
            # if collision_flag:
            #     break
            
            
        images = []
        for filename in sorted(os.listdir(sav_path)):
            #images.append(imageio.imread(filename))
            if filename.split('.')[-1] == 'gif':
                continue
            front_num = filename.split('.')[0]
            if int(front_num) >= 10:
                continue
            images.append(Image.open(sav_path + filename))
        for filename in sorted(os.listdir(sav_path)):
            #images.append(imageio.imread(filename))
            if filename.split('.')[-1] == 'gif':
                continue
            front_num = filename.split('.')[0]
            if int(front_num) < 10:
                continue
            images.append(Image.open(sav_path + filename))
        images[0].save(
            sav_path + scenario_name + '.gif', 
            save_all=True, 
            append_images=images[1:], 
            optimize=True,
            loop=0,
            duration=100,
        )
        

        


def main():
    cfg, cfg_dict = parse_cfg()

    make_csv_file = False
    if make_csv_file:
        # create output directory and logging
        cfg.out = os.path.join(cfg.out, time.strftime("%Y_%m_%d_%H_%M_%S"))
        mkdir(cfg.out)
        log_path = os.path.join(cfg.out, "test_log.txt")
        Logger.init(log_path)
        # save arguments used
        Logger.log("Args: " + str(cfg_dict))

        # device setup
        device = get_device()
        Logger.log("Using device %s..." % (str(device)))

        # load dataset
        test_dataset = map_env = None
        # first create map environment
        data_path = os.path.join(cfg.data_dir, cfg.data_version)
        map_env = NuScenesMapEnv(
            data_path,
            bounds=cfg.map_obs_bounds,
            L=cfg.map_obs_size_pix,
            W=cfg.map_obs_size_pix,
            layers=cfg.map_layers,
            device=device,
        )
        # print("col_dir:", cfg.collision_dir)
        test_dataset = CollisionDataset(
            data_path,
            cfg.collision_dir,
            map_env,
            version=cfg.data_version,
            # split="test" if not cfg.test_on_val else "val",
            split="test",
            categories=cfg.agent_types,
            npast=cfg.past_len,
            nfuture=cfg.future_len,
            reduce_cats=cfg.reduce_cats,
        )
        # create loaders
        test_loader = GraphDataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_test,
            num_workers=cfg.num_workers,
            pin_memory=False,
            worker_init_fn=lambda _: np.random.seed(),
        )  # get around numpy RNG seed bug
        # print(cfg.past_len, 
        #     cfg.future_len,
        #     cfg.map_obs_size_pix,
        #     len(test_dataset.categories),)
        # create model
        model = TrafficModel(
            cfg.past_len,
            cfg.future_len,
            cfg.map_obs_size_pix,
            len(test_dataset.categories),
            map_feat_size=cfg.map_feat_size,
            past_feat_size=cfg.past_feat_size,
            future_feat_size=cfg.future_feat_size,
            latent_size=cfg.latent_size,
            output_bicycle=cfg.model_output_bicycle,
            conv_channel_in=map_env.num_layers,
            conv_kernel_list=cfg.conv_kernel_list,
            conv_stride_list=cfg.conv_stride_list,
            conv_filter_list=cfg.conv_filter_list,
        ).to(device)
        # print(model)
        loss_weights = {
            "recon": 1.0,
            "kl": 1.0,
            "coll_veh_prior": 0.0,
            "coll_env_prior": 0.0,
        }
        loss_fn = TrafficModelLoss(loss_weights).to(device)

        # load model weights
        if cfg.ckpt is not None:
            ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
            Logger.log("Loaded checkpoint from epoch %d..." % (ckpt_epoch))
        else:
            throw_err("Must pass in model weights to evaluate a trained model!")

        Logger.log("Num model params: %d" % (count_params(model)))

        # so can unnormalize as needed
        model.set_normalizer(test_dataset.get_state_normalizer())
        model.set_att_normalizer(test_dataset.get_att_normalizer())
        if cfg.model_output_bicycle:
            from datasets.utils import NUSC_BIKE_PARAMS

            model.set_bicycle_params(NUSC_BIKE_PARAMS)

        # run evaluations on test data
        model.eval()
        with torch.no_grad():
            start_t = time.time()
            run_one_epoch(
                cfg,
                test_loader,
                model,
                map_env,
                loss_fn,
                device,
                cfg.out,
                test_recon_viz_multi=cfg.test_recon_viz_multi,
                test_recon_coll_rate=cfg.test_recon_coll_rate,
                test_sample_viz_multi=cfg.test_sample_viz_multi,
                test_sample_viz_rollout=cfg.test_sample_viz_rollout,
                test_sample_disp_err=cfg.test_sample_disp_err,
                test_sample_coll_rate=cfg.test_sample_coll_rate,
                test_sample_num=cfg.test_sample_num,
                test_sample_future_len=cfg.test_sample_future_len,
                use_challenge_splits=cfg.use_challenge_splits,
            )
            Logger.log("Test time: %f s" % (time.time() - start_t))
    

    post_process_for_inference_and_metric(cfg)
    plot_animation(cfg)


if __name__ == "__main__":
    main()
    
