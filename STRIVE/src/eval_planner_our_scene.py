# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, time, glob
import json
import csv
import tqdm
import torch
import numpy as np

# import matplotlib as mpl

# mpl.use("Agg")

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger
from utils.scenario_gen import log_metric, log_freq_stat, print_metrics
from utils.config import get_parser, add_base_args

from planners.planner import PlannerConfig
from planners.hardcode_goalcond_nusc import DEF_CONFIG, HardcodeNuscPlanner
from icecream import ic
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

def parse_cfg():
    """
    Parse given config file into a config object.

    Returns: config object and config dict
    """
    parser = get_parser("Planner evaluation")
    parser = add_base_args(parser)

    # dataset options
    parser.add_argument(
        "--skip_regular",
        dest="skip_regular",
        action="store_true",
        help="If given, only evaluates on given external scenarios and not regular nuScenes scenarios.",
    )
    parser.set_defaults(skip_regular=False)
    parser.add_argument(
        "--filter_regular",
        dest="filter_regular",
        action="store_true",
        help="If given, regular scenarios are filtered by the indices of the given adversarial scenarios. Note, for this to work as intended, the seq_interval and data split must be the same as used during scenario generation.",
    )
    parser.set_defaults(filter_regular=False)

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["test", "val", "train"],
        help="Which split of the dataset to find scnarios in",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=400,
        help="The size of the validation set used to split the trainval version of the dataset.",
    )
    parser.add_argument(
        "--seq_interval",
        type=int,
        default=10,
        help="skips ahead this many steps from start of current sequence to get the next sequence.",
    )
    parser.add_argument(
        "--shuffle", dest="shuffle", action="store_true", help="Shuffle data"
    )
    parser.set_defaults(shuffle=False)
    parser.add_argument(
        "--random_val",
        dest="random_val",
        action="store_true",
        help="Randomize val split data",
    )
    parser.set_defaults(random_val=False)

    # additional scenarios to evaluate on (e.g. adversarial scenarios)
    parser.add_argument(
        "--scenario_dir",
        type=str,
        default=None,
        help="Directory to load additional evaluation scenarios from.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Data source.",
    )

    # planner
    parser.add_argument(
        "--eval_replay_planner",
        dest="eval_replay_planner",
        action="store_true",
        help="Evaluates stats for GT ego trajectory (replay planner) rather than the rule-based planner.",
    )
    parser.set_defaults(eval_replay_planner=False)

    # rule-based planner configuration
    parser.add_argument("--planner_dt", type=float, default=0.2)
    parser.add_argument("--planner_preddt", type=float, default=0.2)
    parser.add_argument("--planner_nsteps", type=int, default=25)
    parser.add_argument("--planner_xydistmax", type=float, default=2.0)
    parser.add_argument("--planner_cdistang", type=float, default=20.0)
    parser.add_argument("--planner_smax", type=float, default=15.0)
    parser.add_argument("--planner_accmax", type=float, default=3.0)
    parser.add_argument(
        "--planner_predsfacs", type=float, nargs="+", default=[0.5, 1.0]
    )
    parser.add_argument("--planner_predafacs", type=float, nargs="+", default=[0.5])
    parser.add_argument("--planner_interacdist", type=float, default=70.0)
    parser.add_argument("--planner_planaccfacs", type=float, nargs="+", default=[1.0])
    parser.add_argument("--planner_plannspeeds", type=int, default=5)
    parser.add_argument("--planner_col_plim", type=float, default=0.1)
    parser.add_argument("--planner_score_wmin", type=float, default=0.7)
    parser.add_argument("--planner_score_wfac", type=float, default=0.05)

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict


def read_adv_scenes(scene_path):
    scene_flist = sorted(glob.glob(os.path.join(scene_path, "*.json")))
    scene_list = []
    for scene_fpath in scene_flist:
        scene_name = scene_fpath.split("/")[-1][:-5]
        print("Loading %s..." % (scene_name))
        jdict = None
        with open(scene_fpath, "r") as f:
            jdict = json.load(f)
        if jdict is None:
            print("Failed to load! Skipping")
            continue

        cur_scene = {
            "name": scene_name,
            "map": jdict["map"],
            "attack_t": jdict["attack_t"],
        }
        cur_scene["past"] = torch.tensor(jdict["past"])
        cur_scene["init_state"] = torch.tensor(jdict["past"])[:, -1, :]
        cur_scene["veh_att"] = torch.tensor(jdict["lw"])
        cur_scene["adv_fut"] = torch.tensor(jdict["fut_adv"])[1:]
        cur_scene["plan_fut"] = torch.tensor(jdict["fut_adv"])[0]

        scene_list.append(cur_scene)

        ic(np.array(jdict["past"]).shape)
        ic(np.array(jdict["fut_adv"]).shape)
        # ic(cur_scene)

    return scene_list

def read_adv_scenes_from_csv(scene_path):
    scene_flist = sorted(glob.glob(os.path.join(scene_path, "*.csv")))
    scene_list = []
    for scene_fpath in scene_flist:
        scene_name = scene_fpath.split("/")[-1][:-11]
        print("Loading %s..." % (scene_name))
        jdict = None
        
        traj_df = pd.read_csv(scene_fpath, index_col=False)

        ############ padding nearest frame ############
        traj_df.loc[traj_df['X'] == 0, 'X'] = None
        traj_df.loc[traj_df['Y'] == 0, 'Y'] = None
        traj_df[['X', 'Y']] = traj_df.groupby('TRACK_ID')[['X', 'Y']].apply(lambda group: group.ffill().bfill())
        ############ padding nearest frame ############

        ############ move forward no interpolation ############
        need_forward_no_interp = True
        more_frames = 4
        frame_num = len(set(traj_df.TIMESTAMP.values))
        if need_forward_no_interp:
            # forward_df = pd.DataFrame(columns=['TRACK_ID', 'TIMESTAMP', 'V', 'X', 'Y', 'YAW'])
            forward_df = traj_df
            for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                dis_x = (remain_df.iloc[frame_num-1, 3] - remain_df.iloc[frame_num-2, 3])
                dis_y = (remain_df.iloc[frame_num-1, 4] - remain_df.iloc[frame_num-2, 4])
                if dis_x > 10 or dis_y > 10:
                    dis_x, dis_y = 0, 0 
                for fps_f_index in range(1, more_frames+1):
                    t = {'TRACK_ID':[track_id], 'TIMESTAMP':[remain_df.iloc[frame_num-1, 1] + fps_f_index * 500000],
                        'V':[remain_df.iloc[frame_num-1, 2]], 'X':[remain_df.iloc[frame_num-1, 3] + fps_f_index * dis_x],
                        'Y':[remain_df.iloc[frame_num-1, 4] + fps_f_index * dis_y], 'YAW':[remain_df.iloc[frame_num-1, 5]]}
                    df_insert = pd.DataFrame(t)
                    forward_df = pd.concat([forward_df, df_insert], ignore_index=True)
        traj_df = forward_df
        ############ move forward no interpolation ############

        whole_past_traj = []
        whole_future_traj = []
        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
            # 計算 x, y, hcos, hsin, s, hdot
            x = remain_df['X'].values
            y = remain_df['Y'].values
            # hcos = np.cos(remain_df['YAW'].values)
            # hsin = np.sin(remain_df['YAW'].values)
            hcos = np.cos(np.radians(remain_df['YAW'].values))
            hsin = np.sin(np.radians(remain_df['YAW'].values))
            s = remain_df['V'].values

            # 計算 hdot: 朝向角的角速度 (YAW rate)
            timestamps = remain_df['TIMESTAMP'].values
            yaw = remain_df['YAW'].values
            # 計算 YAW 的時間差和角度差
            delta_time = np.diff(timestamps, prepend=timestamps[0])  # 前置第一個值防止數量不匹配
            delta_yaw = np.diff(yaw, prepend=yaw[0])
            hdot = delta_yaw / delta_time

            # 堆疊成 [timestamp, 6] 格式
            past_traj_data = np.column_stack((x, y, hcos, hsin, s, hdot))
            future_traj_data = np.column_stack((x, y, hcos, hsin))

            # 將過去和未來軌跡分開
            past_traj = past_traj_data[:8]
            future_traj = future_traj_data[8:]

            # 根據 track_id 來加入到 ego 或其他物體的清單
            if str(track_id) == 'ego':
                # whole_past_traj.append(past_traj)
                # whole_future_traj.append(future_traj)
                whole_past_traj.insert(0, past_traj)
                whole_future_traj.insert(0, future_traj)
            else:
                whole_past_traj.append(past_traj)
                whole_future_traj.append(future_traj)
        # for track_id, remain_df in traj_df.groupby("TRACK_ID"):
        #     if str(track_id) == 'ego':
        #         # print(remain_df['X'].values, remain_df['X'].values[:8], remain_df['X'].values[8:])
        #         ego_past_traj = np.column_stack((remain_df['X'].values[:8], remain_df['Y'].values[:8]))
        #         whole_past_traj.append(ego_past_traj)
        #         ego_future_traj = np.column_stack((remain_df['X'].values[8:], remain_df['Y'].values[8:]))
        #         whole_future_traj.append(ego_future_traj)
        # for track_id, remain_df in traj_df.groupby("TRACK_ID"):
        #     if str(track_id) != 'ego':
        #         other_past_traj = np.column_stack((remain_df['X'].values[:8], remain_df['Y'].values[:8]))
        #         whole_past_traj.append(other_past_traj)
        #         other_future_traj = np.column_stack((remain_df['X'].values[8:], remain_df['Y'].values[8:]))
        #         whole_future_traj.append(other_future_traj)
        single_lw = [4.084000110626221, 1.7300000190734863]
        scene_length = int(scene_name.split('_')[7].split('-')[-1])
        vehicle_num = len(whole_past_traj)
        whole_lw = np.tile(single_lw, (vehicle_num, 1))

        ###############################################
        # if scene_length == 8:
        #     continue
        ###############################################

        cur_scene = {
            "name": scene_name,
            "map": scene_name.split('_')[4],
            "attack_t": scene_length,
        }

        cur_scene["past"] = torch.tensor(np.array(whole_past_traj))
        cur_scene["init_state"] = torch.tensor(np.array(whole_past_traj))[:, -1, :]
        cur_scene["veh_att"] = torch.tensor(whole_lw)
        cur_scene["adv_fut"] = torch.tensor(np.array(whole_future_traj))[1:]
        cur_scene["plan_fut"] = torch.tensor(np.array(whole_future_traj))[0]

        scene_list.append(cur_scene)

        # ic(np.array(jdict["past"]).shape) # NA, 8, 6
        # ic(np.array(jdict["fut_adv"]).shape) # NA, 4, 4
        # ic(cur_scene)
        # ic(cur_scene["init_state"].shape)
        # ic(cur_scene["veh_att"].shape)


    return scene_list

def compute_velocity_yaw(traj):
    x, y = traj[:, 0], traj[:, 1]
    hcos, hsin = traj[:, 2], traj[:, 3]
    
    v = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / 0.1
    v = np.append(v, v[-1]) 

    yaw = np.arctan2(hsin, hcos)
    yaw = np.degrees(yaw)
    yaw = (yaw + 180) % 360 - 180
    
    return x, y, v, yaw

def traj_to_df(planner_traj, non_ego_traj, scene_name, source):
    T = planner_traj.shape[0]
    timestamps = np.arange(T) * 0.1
    x, y, v, yaw = compute_velocity_yaw(planner_traj)
    ego_df = pd.DataFrame({
        "TRACK_ID": "ego",
        "Timestamp": timestamps,
        "X": x,
        "Y": y,
        "V": v,
        "YAW": yaw
    })

    other_df_list = []
    for i, traj in enumerate(non_ego_traj):
        x, y, v, yaw = compute_velocity_yaw(traj)
        df = pd.DataFrame({
            "TRACK_ID": f"vehicle_{i+1}",
            "Timestamp": timestamps,
            "X": x,
            "Y": y,
            "V": v,
            "YAW": yaw
        })
        other_df_list.append(df)

    final_df = pd.concat([ego_df] + other_df_list, ignore_index=True)

    csv_dir = './rule-based_planner_on_forward_interpolation_' + source + '/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    final_df.to_csv(csv_dir + scene_name + ".csv", index=False)

def compute_metrics(
    adv_scene,
    all_scenario_type_dict,
    col_scenario_type_dict,
    planner_traj,
    non_ego_traj,
    veh_att,
    dt,
    metrics,
    freq_metrics_cnt,
    freq_metrics_total,
    prefix,
    log_no_prefix_copy=True,
    ego_idx=0,
):
    """
    :param planner_traj: (T x 4)
    :param non_eg_traj: (NA-1 x T x 4)
    :param veh_att: (NA, 2)
    """
    from losses.adv_gen_nusc import interp_traj, check_single_veh_coll

    ################
    scenario_type = adv_scene["name"].split('_')[5]
    all_scenario_type_dict[scenario_type] += 1
    print(col_scenario_type_dict)
    print(all_scenario_type_dict)
    ################



    NA = veh_att.size(0)
    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[ego_idx] = True

    # interpolate to get better approx of coll and acceleration
    #####################################
    interp_scale = 3
    # interp_scale = 1
    #####################################
    interp_dt = dt / float(interp_scale)
    planner_traj_interp = interp_traj(
        planner_traj.unsqueeze(0), scale_factor=interp_scale
    )[0]
    non_ego_traj_interp = interp_traj(non_ego_traj, scale_factor=interp_scale)

    # ic(planner_traj.shape, planner_traj)
    # ic(planner_traj_interp.shape)
    # ic(non_ego_traj.shape, non_ego_traj)
    # ic(non_ego_traj_interp.shape)
    # ego_xy = planner_traj_interp[:,:2].numpy()
    # other_xy = non_ego_traj_interp[:, :,:2].numpy()
    # ic(ego_xy.shape)
    # ic(other_xy.shape)
    # # plt.plot(ego_xy, color='red')
    # ego_x = ego_xy[:, 0]
    # ego_y = ego_xy[:, 1]
    # plt.scatter(ego_x, ego_y, color='r')
    # plt.plot(ego_x, ego_y, marker='o', linestyle='-', color='r', label='Trajectory')
    # for other_xy_each in other_xy:
    #     other_x = other_xy_each[:, 0]
    #     other_y = other_xy_each[:, 1]
    #     plt.plot(other_x, other_y, marker='o', linestyle='-', color='black', label='Trajectory')
    #     # plt.plot(other_xy_each, color='black')
    # plt.show()



    #
    # collisions
    #
    planner_coll_all, planner_coll_time = check_single_veh_coll(
        planner_traj_interp, veh_att[ego_idx], non_ego_traj_interp, veh_att[~ego_mask]
    )
    did_collide = np.sum(planner_coll_all) > 0
    coll_time = np.amin(planner_coll_time)
    coll_agt = np.argmin(planner_coll_time)
    coll_time_sec = (coll_time + 1) * interp_dt  # convert to seconds
    freq_metrics_cnt, freq_metrics_total = log_freq_stat(
        freq_metrics_cnt, freq_metrics_total, prefix + "_coll", int(did_collide), 1
    )
    if log_no_prefix_copy:
        freq_metrics_cnt, freq_metrics_total = log_freq_stat(
            freq_metrics_cnt, freq_metrics_total, "total_coll", int(did_collide), 1
        )

    cur_seq_metrics = {"did_collide": int(did_collide)}

    # collision index in original low-res data
    coll_idx = (
        int((coll_time * interp_dt) / dt) if did_collide else planner_traj.size(0) - 1
    )

    # relative speed at collision if it did collide
    if did_collide:
        if coll_idx > 0:
            atk_vel = (
                non_ego_traj[coll_agt, coll_idx, :2]
                - non_ego_traj[coll_agt, coll_idx - 1, :2]
            ) / dt
            plan_vel = (
                planner_traj[coll_idx, :2] - planner_traj[coll_idx - 1, :2]
            ) / dt
        else:
            atk_vel = (
                non_ego_traj[coll_agt, coll_idx + 1, :2]
                - non_ego_traj[coll_agt, coll_idx, :2]
            ) / dt
            plan_vel = (
                planner_traj[coll_idx + 1, :2] - planner_traj[coll_idx, :2]
            ) / dt
        rel_vel = plan_vel - atk_vel
        atk_s = torch.norm(atk_vel).item()
        plan_s = torch.norm(plan_vel).item()
        coll_rel_s = torch.norm(rel_vel).item()
        metrics = log_metric(metrics, prefix + "_coll_vel", np.array([coll_rel_s]))
        if log_no_prefix_copy:
            metrics = log_metric(metrics, "total_coll_vel", np.array([coll_rel_s]))

        cur_seq_metrics["coll_vel"] = coll_rel_s

        ##########################
        col_scenario_type_dict[scenario_type] += 1
        ##########################

    #
    # comfort (accelerations)
    #
    pre_crash_pos = planner_traj[: coll_idx + 1, :2]
    pre_crash_head = planner_traj[: coll_idx + 1, 2:]
    if pre_crash_pos.size(0) > 2:
        planner_vel = (pre_crash_pos[1:] - pre_crash_pos[:-1]) / dt
        planner_s = torch.norm(planner_vel, dim=-1)
        unit_head = pre_crash_head / torch.norm(pre_crash_head, dim=-1, keepdim=True)
        planner_vel = planner_s.unsqueeze(1) * unit_head[:-1]
        plan_fwd_accel = torch.abs((planner_s[1:] - planner_s[:-1]) / dt)
        planner_accel = (planner_vel[1:] - planner_vel[:-1]) / dt
        lat_dir = torch.cat([-unit_head[:-2, 1:2], unit_head[:-2, 0:1]], dim=1)
        plan_lat_accel = torch.sum(
            planner_accel * lat_dir, dim=-1
        )  # project accel onto lateral dir
        plan_lat_accel = torch.abs(plan_lat_accel)

        planner_accel = torch.norm(planner_accel, dim=-1)
        planner_accel = planner_accel.numpy()
        plan_fwd_accel = plan_fwd_accel.numpy()
        plan_lat_accel = plan_lat_accel.numpy()

        max_fwd_accel = np.amax(plan_fwd_accel)
        max_lat_accel = np.amax(plan_lat_accel)

        # mean accel over all pre-crash frame
        metrics = log_metric(metrics, prefix + "_accel", planner_accel)
        if log_no_prefix_copy:
            metrics = log_metric(metrics, "total_accel", planner_accel)

        cur_seq_metrics["mean_accel"] = np.mean(planner_accel)

        # mean accel over all pre-crash frame
        metrics = log_metric(metrics, prefix + "_accel_fwd", plan_fwd_accel)
        if log_no_prefix_copy:
            metrics = log_metric(metrics, "total_accel_fwd", plan_fwd_accel)

        cur_seq_metrics["mean_accel_fwd"] = np.mean(plan_fwd_accel)

        # mean accel over all pre-crash frame
        metrics = log_metric(metrics, prefix + "_accel_lat", plan_lat_accel)
        if log_no_prefix_copy:
            metrics = log_metric(metrics, "total_accel_lat", plan_lat_accel)

        cur_seq_metrics["mean_accel_lat"] = np.mean(plan_lat_accel)

    return metrics, freq_metrics_cnt, freq_metrics_total, cur_seq_metrics, all_scenario_type_dict, col_scenario_type_dict


def run_planner_eval(
    plan_cfg,
    source,
    eval_loader,
    map_env,
    dt,
    device,
    out_path,
    state_norm,
    att_norm,
    scenario_dir=None,
    skip_regular=False,
    eval_replay_planner=False,
    filter_regular=False,
):
    """
    eval_loader is assumed to be using batch size 1.
    """

    mkdir(out_path)
    viz_nusc_out_path = viz_adv_out_path = None

    planner = HardcodeNuscPlanner(map_env, plan_cfg)

    metrics = {}
    freq_metrics_cnt = {}
    freq_metrics_total = {}

    all_scenario_type_dict = {'HO': 0, 'LTAP': 0, 'RE': 0, 'JC': 0, 'LC': 0}
    col_scenario_type_dict = {'HO': 0, 'LTAP': 0, 'RE': 0, 'JC': 0, 'LC': 0}    


    tot_seq_metrics_list = []
    tot_name_list = []

    # read in and run through adversarial dataset
    name_list = []
    seq_metrics_list = []
    adv_scene_list = None
    if scenario_dir is not None:
        print("Reading in adversarial scenarios...")
        # list of scenes: each a dict with map (str), init_state (N, 6), veh_att (N, 2), adv_fut (N-1, FT, 4)
        ic(scenario_dir)
        # adv_scene_list = read_adv_scenes(scenario_dir)
        adv_scene_list = read_adv_scenes_from_csv(scenario_dir)
        # ic(adv_scene_list)
        pbar_adv = tqdm.tqdm(adv_scene_list)

        ################## example #######################
        # example_adv_scene_list =  read_adv_scenes("./out/adv_gen_rule_based_out_1728564449/scenario_results/adv_sol_success")
        # pbar_adv = tqdm.tqdm(example_adv_scene_list)

        # for i, adv_scene in enumerate(pbar_adv):
        #     name_list.append("adv_" + adv_scene["name"])
        #     NA = adv_scene["init_state"].size(0)
        #     start_t = time.time()
        #     # reset planner
        #     init_state = adv_scene["init_state"]
        #     veh_att = adv_scene["veh_att"]
        #     map_idx = map_env.map_list.index(adv_scene["map"])
        #     map_idx = torch.tensor([map_idx], dtype=int)
        #     planner.reset(init_state, veh_att, torch.zeros((NA)), 1, map_idx)
        #     # rollout
        #     non_ego_traj = adv_scene["adv_fut"].numpy()  # (NA-1, FT, 4)
        #     FT = non_ego_traj.shape[1]
        #     plan_t = np.linspace(dt, dt * FT, FT)
        #     non_ego_ptr = torch.tensor([0, NA - 1])
        #     if eval_replay_planner:
        #         planner_traj = adv_scene["plan_fut"]
        #     else:
        #         planner_traj = planner.rollout(
        #             non_ego_traj, plan_t, non_ego_ptr.numpy(), plan_t, control_all=False
        #         )[
        #             0
        #         ]  # (FT, 4)
        ################## example #######################
                
        for i, adv_scene in enumerate(pbar_adv):
            name_list.append("adv_" + adv_scene["name"])
            NA = adv_scene["init_state"].size(0)

            start_t = time.time()
            # reset planner
            init_state = adv_scene["init_state"]
            veh_att = adv_scene["veh_att"]
            map_idx = map_env.map_list.index(adv_scene["map"])
            map_idx = torch.tensor([map_idx], dtype=int)
            planner.reset(init_state, veh_att, torch.zeros((NA)), 1, map_idx)
            # rollout
            non_ego_traj = adv_scene["adv_fut"].numpy()  # (NA-1, FT, 4)
            FT = non_ego_traj.shape[1]
            plan_t = np.linspace(dt, dt * FT, FT)
            non_ego_ptr = torch.tensor([0, NA - 1])

            ic(adv_scene["name"])
            ic(non_ego_traj.shape)
            ic(dt, plan_t)
            # ic(eval_replay_planner) # False

            if eval_replay_planner:
                planner_traj = adv_scene["plan_fut"]
            else:
                planner_traj = planner.rollout(
                    non_ego_traj, plan_t, non_ego_ptr.numpy(), plan_t, control_all=False
                )[
                    0
                ]  # (FT, 4)

            # compute & save metrics
            (
                metrics,
                freq_metrics_cnt,
                freq_metrics_total,
                cur_seq_metrics, all_scenario_type_dict, col_scenario_type_dict
            ) = compute_metrics(
                adv_scene,
                all_scenario_type_dict,
                col_scenario_type_dict,
                planner_traj,
                adv_scene["adv_fut"],
                veh_att,
                dt,
                metrics,
                freq_metrics_cnt,
                freq_metrics_total,
                "adv",
                log_no_prefix_copy=True
            )
            missing_metrics = [
                "mean_accel",
                "mean_accel_fwd",
                "mean_accel_lat",
                "coll_vel",
            ]
            for mm in missing_metrics:
                if mm not in cur_seq_metrics:
                    cur_seq_metrics[mm] = np.nan
            seq_metrics_list.append(cur_seq_metrics)

            traj_to_df(planner_traj, adv_scene["adv_fut"], adv_scene["name"], source)

            # print_metrics(metrics, freq_metrics_cnt, freq_metrics_total)
            

    tot_name_list += name_list
    tot_seq_metrics_list += seq_metrics_list

    """
    if not skip_regular:
        filter_scene_dict = None
        if filter_regular:
            assert adv_scene_list is not None  # must have scenarios to filter
            # create dict of the scenarios to actually run
            filter_scene_dict = {
                int(advscene["name"].split("_")[1]): True for advscene in adv_scene_list
            }

        # now go through the regular non-adversarial scenarios
        name_list = []
        seq_metrics_list = []
        nusc_ego_traj = []
        pbar_data = tqdm.tqdm(eval_loader)
        for i, data in enumerate(pbar_data):
            if filter_regular and i not in filter_scene_dict:
                continue

            scene_graph, map_idx = data
            NA = scene_graph.future_gt.size(0)

            if scene_graph.past_gt.size(0) == 1:
                print("Only ego in scene, skipping...")
                continue

            name_list.append("regular_seq_%05d" % (i))

            # ego always at 0
            cur_ego_idx = 0
            start_t = time.time()
            # reset planner
            init_state = state_norm.unnormalize(scene_graph.past_gt[:, -1, :])
            veh_att = att_norm.unnormalize(scene_graph.lw)
            planner.reset(
                init_state, veh_att, scene_graph.batch, 1, map_idx, ego_idx=cur_ego_idx
            )
            # rollout
            ego_mask = torch.zeros((NA), dtype=torch.bool)
            ego_mask[cur_ego_idx] = True
            non_ego_traj = state_norm.unnormalize(
                scene_graph.future_gt[~ego_mask][:, :, :4]
            ).numpy()  # (NA-1, FT, 4)
            FT = non_ego_traj.shape[1]
            plan_t = np.linspace(dt, dt * FT, FT)
            non_ego_ptr = scene_graph.ptr - torch.arange(2)

            if not eval_replay_planner:
                planner_traj = planner.rollout(
                    non_ego_traj, plan_t, non_ego_ptr.numpy(), plan_t, control_all=False
                )[
                    0
                ]  # (FT, 4)
            else:
                assert cur_ego_idx == 0
                planner_traj = state_norm.unnormalize(
                    scene_graph.future_gt[0, :, :4]
                )  # only works with single ego_idx

            # compute & save metrics
            (
                metrics,
                freq_metrics_cnt,
                freq_metrics_total,
                cur_seq_metrics, all_scenario_type_dict, col_scenario_type_dict
            ) = compute_metrics(
                adv_scene,
                all_scenario_type_dict,
                col_scenario_type_dict,
                planner_traj,
                torch.from_numpy(non_ego_traj),
                veh_att,
                dt,
                metrics,
                freq_metrics_cnt,
                freq_metrics_total,
                "regular",
                log_no_prefix_copy=True,
                ego_idx=cur_ego_idx,
            )
            # print_metrics(metrics, freq_metrics_cnt, freq_metrics_total)

            missing_metrics = [
                "mean_accel",
                "mean_accel_fwd",
                "mean_accel_lat",
                "coll_vel",
            ]
            for mm in missing_metrics:
                if mm not in cur_seq_metrics:
                    cur_seq_metrics[mm] = np.nan
            seq_metrics_list.append(cur_seq_metrics)

            traj_to_df(planner_traj, non_ego_traj, adv_scene["name"])

        tot_name_list += name_list
        tot_seq_metrics_list += seq_metrics_list

        # and finally for total
        csv_path_out = os.path.join(out_path, "all_eval_results.csv")
        with open(csv_path_out, "w") as f:
            csvwrite = csv.writer(f)
            header_row = ["scene"]
            met_names = sorted(list(tot_seq_metrics_list[0].keys()))
            header_row += met_names
            csvwrite.writerow(header_row)
            for sidx, scene_name in enumerate(tot_name_list):
                currow = [scene_name]
                currow += [tot_seq_metrics_list[sidx][k] for k in met_names]
                csvwrite.writerow(currow)
    """
    Logger.log("Final ================")
    print_metrics(metrics, freq_metrics_cnt, freq_metrics_total)

    all_col_num = sum(col_scenario_type_dict.values())
    all_num = sum(all_scenario_type_dict.values())
    col_scenario_type_dict['all'] = all_col_num
    all_scenario_type_dict['all'] = all_num
    print("all:", all_scenario_type_dict, all_num)
    for variant_key in all_scenario_type_dict:
        col_scenario_type_dict[variant_key] /= all_scenario_type_dict[variant_key]
    col_df = pd.DataFrame(col_scenario_type_dict, index=['collision rate'])
    col_df.to_csv('collision_rate_on_rule-based_planner_' + source + '.csv', mode='a', header=False)
    print("col:", col_scenario_type_dict, all_col_num)


def main():
    cfg, cfg_dict = parse_cfg()

    # create output directory and logging
    cfg.out = (
        cfg.out + "_" + str(int(time.time())) + "_" + str(np.random.randint(0, 10000))
    )
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, "eval_planner_log.txt")
    Logger.init(log_path)
    # save arguments used
    Logger.log("Args: " + str(cfg_dict))

    # device setup
    device = torch.device("cpu")
    Logger.log("Using device %s..." % (str(device)))

    # load dataset
    eval_dataset = map_env = None
    # first create map environment
    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(
        data_path,
        bounds=[-17.0, -38.5, 60.0, 38.5],
        L=256,
        W=256,
        layers=["drivable_area", "carpark_area", "road_divider", "lane_divider"],
        device=device,
        load_lanegraph=True,
        lanegraph_res_meters=1.0,
    )
    eval_dataset = NuScenesDataset(
        data_path,
        map_env,
        version=cfg.data_version,
        split=cfg.split,
        categories=["car", "truck"],
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        seq_interval=cfg.seq_interval,
        randomize_val=cfg.random_val,
        val_size=cfg.val_size,
    )
    # create loaders
    eval_loader = GraphDataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=False,
        worker_init_fn=lambda _: np.random.seed(),
    )  # get around numpy RNG seed bug

    # create planner config
    # the config passed in through args
    plan_cfg_dict = {}
    for k in DEF_CONFIG:
        plan_cfg_dict[k] = cfg_dict["planner_" + k]
    print("Custom planner config:")
    print(plan_cfg_dict)
    
    plan_cfg = PlannerConfig(**plan_cfg_dict)

    # save custom config
    with open(os.path.join(cfg.out, "plan_cfg.json"), "w") as fplan:
        json.dump(plan_cfg_dict, fplan)

    run_planner_eval(
        plan_cfg,
        cfg.source,
        eval_loader,
        map_env,
        eval_dataset.dt,
        # 0.1,
        device,
        cfg.out,
        eval_dataset.get_state_normalizer(),
        eval_dataset.get_att_normalizer(),
        scenario_dir=cfg.scenario_dir,
        skip_regular=cfg.skip_regular,
        eval_replay_planner=cfg.eval_replay_planner,
        filter_regular=cfg.filter_regular,
    )


if __name__ == "__main__":
    main()
