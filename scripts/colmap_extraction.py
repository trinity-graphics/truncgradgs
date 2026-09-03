import argparse
import os
import shutil
import sqlite3
import sys

import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id),
        )
        return cursor.lastrowid


def camTodatabase(txtfile, database_path):
    import os

    camModelDict = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "FULL_OPENCV": 5,
        "SIMPLE_RADIAL_FISHEYE": 6,
        "RADIAL_FISHEYE": 7,
        "OPENCV_FISHEYE": 8,
        "FOV": 9,
        "THIN_PRISM_FISHEYE": 10,
    }

    if os.path.exists(database_path) == False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList = list()
    modelList = list()
    widthList = list()
    heightList = list()
    paramsList = list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0, len(lines), 1):
            if lines[i][0] != "#":
                strLists = lines[i].split()
                cameraId = int(strLists[0])
                cameraModel = camModelDict[strLists[1]]  # SelectCameraModel
                width = int(strLists[2])
                height = int(strLists[3])
                paramstr = np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(
                    cameraModel, width, height, params, cameraId
                )

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0, len(idList), 1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert (
            model == modelList[i] and width == widthList[i] and height == heightList[i]
        )
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # TODO: refine it.
    parser.add_argument("path", type=str, help="path to our colmap synthetic dataset.")
    parser.add_argument("--sparse-only", action="store_true", help="only extract sparse point cloud.")
    args = parser.parse_args()

    # path must end with / to make sure image path is relative
    if args.path[-1] != "/":
        args.path += "/"

    colmap_workspace = os.path.join(args.path, "tmp")
    db_path = os.path.join(colmap_workspace, "database.db")

    os.makedirs(os.path.join(colmap_workspace, "created", "sparse"), exist_ok=True)
    os.symlink(
        os.path.join(args.path, "sparse/0/cameras.txt"),
        os.path.join(colmap_workspace, "created/sparse/cameras.txt"),
    )
    os.symlink(
        os.path.join(args.path, "sparse/0/images.txt"),
        os.path.join(colmap_workspace, "created/sparse/images.txt"),
    )
    with open(os.path.join(colmap_workspace, "created/sparse/points3D.txt"), "w") as f:
        f.write("")
    os.symlink(
        os.path.join(args.path, "images"), os.path.join(colmap_workspace, "images")
    )

    do_system(
        f"colmap feature_extractor \
                --database_path {db_path} \
                --image_path {os.path.join(colmap_workspace, 'images')}"
    )

    camTodatabase(os.path.join(colmap_workspace, "created/sparse/cameras.txt"), db_path)

    do_system(
        f"colmap exhaustive_matcher  \
                --database_path {db_path}"
    )

    os.makedirs(os.path.join(colmap_workspace, "triangulated", "sparse"), exist_ok=True)

    do_system(
        f"colmap point_triangulator   \
                --database_path {db_path} \
                --image_path {os.path.join(colmap_workspace, 'images')} \
                --input_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                --output_path  {os.path.join(colmap_workspace, 'triangulated/sparse')}"
    )


    if args.sparse_only:
        shutil.copyfile(
            os.path.join(colmap_workspace, "triangulated/sparse/points3D.bin"),
            os.path.join(args.path, "sparse/0/points3D_colmap.bin"),
        )
        shutil.rmtree(colmap_workspace)
        exit(0)

    do_system(
        f"colmap model_converter \
                --input_path  {os.path.join(colmap_workspace, 'triangulated/sparse')} \
                --output_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                --output_type TXT"
    )

    os.makedirs(os.path.join(colmap_workspace, "dense"), exist_ok=True)

    do_system(
        f"colmap image_undistorter  \
                --image_path  {os.path.join(colmap_workspace, 'images')} \
                --input_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                --output_path  {os.path.join(colmap_workspace, 'dense')}"
    )

    do_system(
        f"colmap patch_match_stereo   \
                --workspace_path   {os.path.join(colmap_workspace, 'dense')}"
    )

    do_system(
        f"colmap stereo_fusion    \
                --workspace_path {os.path.join(colmap_workspace, 'dense')} \
                --output_path {os.path.join(args.path, 'sparse/0/', 'points3D_colmap_dense.ply')}"
    )

    shutil.rmtree(colmap_workspace)
    os.remove(os.path.join(args.path, "sparse/0/", "points3D_colmap_dense.ply.vis"))
    shutil.copyfile(
        os.path.join(args.path, "sparse/0/points3D_colmap_dense.ply"),
        os.path.join(args.path, "points3D_colmap_dense.ply"),
    )

    print(
        f"[INFO] Initial point cloud is saved in {os.path.join(args.path, 'sparse/0/', 'points3D_colmap_dense.ply')}."
    )
