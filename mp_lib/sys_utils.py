import os
import pickle
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import lmdb
import os.path as op
import cv2 as cv

import shutil
from loguru import logger
import os


def copy(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)


def copy_repo(src_files, dst_folder, filter_keywords):
    src_files = [
        f for f in src_files if not any(keyword in f for keyword in filter_keywords)
    ]
    dst_files = [op.join(dst_folder, op.basename(f)) for f in src_files]
    for src_f, dst_f in zip(src_files, dst_files):
        logger.info(f"FROM: {src_f}\nTO:{dst_f}")
        copy(src_f, dst_f)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def mkdir_p(exp_path):
    os.makedirs(exp_path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path, obj):
    with (open(path, "wb")) as f:
        pickle.dump(obj, f)


def count_files(path):
    """
    Non-recursively count number of files in a folder.
    """
    files = glob(path)
    return len(files)


def fetch_lmdb_reader(db_path):
    env = lmdb.open(
        db_path,
        subdir=op.isdir(db_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn = env.begin(write=False)
    return txn


def read_lmdb_image(txn, fname):
    image_bin = txn.get(fname.encode("ascii"))
    if image_bin is None:
        return image_bin
    image = np.fromstring(image_bin, dtype=np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image


def package_lmdb(lmdb_name, map_size, fnames, keys, write_frequency=5000):
    """
    Package image files into a lmdb database.
    fnames are the paths to each file and also the key to fetch the images.
    lmdb_name is the name of the lmdb database file
    map_size: recommended to set to len(fnames)*num_types_per_image*10
    keys: the key of each image in dict
    """
    assert len(fnames) == len(keys)
    print("Start packaging lmdb database: {}".format(lmdb_name))
    db = lmdb.open(lmdb_name, map_size=map_size)
    txn = db.begin(write=True)
    print("Begin loop")
    for idx, (fname, key) in tqdm(enumerate(zip(fnames, keys)), total=len(fnames)):
        img = cv.imread(fname)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        status, encoded_image = cv.imencode(".png", img, [cv.IMWRITE_JPEG_QUALITY, 100])
        assert status
        txn.put(key.encode("ascii"), encoded_image.tostring())

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()

    db.sync()
    db.close()
    print("Done")
