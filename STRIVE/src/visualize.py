import argparse
import pickle
from generation.Drawer import Drawer
from generation.NuscData import NuscData
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import glob, os
from multiprocessing import Process, Semaphore
from tqdm import trange
import random
from icecream import ic


def parse_cfg():
    parser = argparse.ArgumentParser(
        prog="python3 src/visualize.py",
        description="Visualize generated dataset from given pickle file",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        default=None,
        help="Dataset folder",
    )
    parser.add_argument(
        "-v",
        "--version",
        default="trainval",
        choices=["mini", "trainval", "middle"],
        help="Data version",
    )
    parser.add_argument(
        "-e",
        "--encode",
        action="store_true",
        help="Encode with H.264",
    )
    args = parser.parse_args()
    print(args)
    return args


def show(path: str, out_dir, nuscs, encode):
    out = os.path.join(out_dir, path.replace("/", "_")[12:-7])
    print(f"Loading: {path}", flush=True)
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    found = False
    for nuscData, nuscMap in nuscs:
        if nuscData.scene["name"] == dataset.scene["name"]:
            found = True
            plt = Drawer(nuscMap)
            plt.plot_dataset(dataset, out)
            plt.export_mp4(out, encode)
            plt.close()
            print(f"Saved: {path}", flush=True)
    if not found:
        assert False, f"Scene {dataset.scene['name']} not found"


def sem_show(sem: Semaphore, path: str, out_dir, nuscs, encode):
    sem.acquire()
    if not os.path.exists(f"{out_dir}/{path.replace('/', '_')[12:-7]}.mp4"):
        show(path, out_dir, nuscs, encode)
    else:
        print(f"Skipped: {path}", flush=True)
    sem.release()


def main():
    args = parse_cfg()
    out_dir = os.path.join(args.dir, args.version, "viz")
    maps = dict()
    nuscs = list()
    nusc_obj = NuScenes(
        version=f"v1.0-{'mini' if args.version == 'mini' else 'trainval'}",
        dataroot=f"data/nuscenes/{'mini' if args.version == 'mini' else 'trainval'}",
        verbose=True,
    )
    for i in trange(len(nusc_obj.scene)):
        nuscData = NuscData(nusc_obj, i)
        mapName = nuscData.get_map()
        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot=f"data/nuscenes/{'mini' if args.version == 'mini' else 'trainval'}",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]
        nuscs.append((nuscData, nuscMap))
    pickles = glob.glob(
        f"./{os.path.join(args.dir,args.version)}/**/*.pickle", recursive=True
    )
    # pickles = ['./data/hcis/v4.1/trainval/boston-seaport/HO/scene-0164/5-0-29_83515afd82064e64b301d56843858d7e.pickle']
    # pickles = [p for p in pickles if p.find("scene-0757") != -1]
    # ic(pickles)
    random.shuffle(pickles)
    params = list()
    for path in pickles:
        out = os.path.join(out_dir, path.replace("/", "_")[12:-7])
        if not os.path.exists(f"{out}.mp4"):
            params.append((path, out_dir, nuscs, args.encode))
    print(f"Total: {len(params)}")
    params = random.sample(params, min(len(params), 400))
    print(f"Sampled: {len(params)}")
    sem = Semaphore(1)
    plist = list()
    for param in params:
        p = Process(target=sem_show, args=(sem, *param))
        p.start()
        plist.append(p)
    for p in plist:
        p.join()
    print("Done")


if __name__ == "__main__":
    main()
