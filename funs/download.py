"""로컬 데이터셋 로딩 모듈."""
import os

import numpy as np
import pandas as pd
from scipy import io


def load_cwru(root: str, sample_rate: str = "12k") -> pd.DataFrame:
    """CWRU 데이터를 로컬 디렉토리에서 읽어 DataFrame으로 반환."""
    label_map = {
        ("N", "000"): 0, ("B", "007"): 1, ("B", "014"): 2, ("B", "021"): 3, ("B", "028"): 999,
        ("IR", "007"): 4, ("IR", "014"): 5, ("IR", "021"): 6, ("IR", "028"): 999,
        ("OR@03", "007"): 999, ("OR@03", "021"): 999,
        ("OR@06", "007"): 7, ("OR@06", "014"): 8, ("OR@06", "021"): 9,
        ("OR@12", "007"): 999, ("OR@12", "021"): 999,
    }
    load_to_rpm = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

    df = {"data": [], "fault_type": [], "crack_size": [], "rpm": [], "label": [], "is_anomaly": []}

    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".mat"):
            continue
        stem = fname[:-4]  # e.g. "IR_007_0"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        prefix, load_idx_str = parts
        try:
            load_idx = int(load_idx_str)
        except ValueError:
            continue

        # prefix: "N_000", "B_007", "IR_007", "OR@06_007", etc.
        sub = prefix.split("_", 1)
        if len(sub) != 2:
            continue
        fault_type, crack_size = sub[0], sub[1]
        key = (fault_type, crack_size)
        if key not in label_map:
            continue
        if load_idx not in load_to_rpm:
            continue

        mat_path = os.path.join(root, fname)
        try:
            mat = io.loadmat(mat_path)
        except Exception:
            print(f"[WARNING] {fname} 로드 실패 — 스킵")
            continue

        body = None
        for elem in mat.keys():
            if "DE" in elem:
                body = mat[elem]
                break

        if body is None:
            print(f"[WARNING] 'DE' 채널 없음: {fname} — 스킵")
            continue

        body = np.ravel(body, order="F")
        df["fault_type"].append(fault_type)
        df["crack_size"].append(crack_size)
        df["rpm"].append(load_to_rpm[load_idx])
        df["label"].append(label_map[key])
        df["is_anomaly"].append(0 if fault_type == "N" else 1)
        df["data"].append(body)

    return pd.DataFrame(df)


def load_paderborn(root: str) -> pd.DataFrame:
    """PU 데이터를 로컬 디렉토리에서 읽어 DataFrame으로 반환."""
    bearing_fault = {
        "K001": "N", "K002": "N", "K003": "N", "K004": "N", "K005": "N", "K006": "N",
        "KI04": "IR", "KI14": "IR", "KI16": "IR", "KI17": "IR", "KI18": "IR", "KI21": "IR",
        "KA04": "OR", "KA15": "OR", "KA16": "OR", "KA22": "OR", "KA30": "OR",
    }
    label_map = {"N": 0, "IR": 1, "OR": 2}
    domain_statistics = {
        "N15_M07_F10": (1500, 0.7, 1000),
        "N09_M07_F10": (900,  0.7, 1000),
    }

    df = {
        "data": [], "fault_type": [], "sampling_rate": [], "rpm": [],
        "load_torque(Nm)": [], "radial_force(N)": [], "label": [], "is_anomaly": [],
    }

    for bearing_name in sorted(os.listdir(root)):
        bearing_dir = os.path.join(root, bearing_name)
        if not os.path.isdir(bearing_dir):
            continue
        if bearing_name not in bearing_fault:
            continue

        fault_type = bearing_fault[bearing_name]
        is_anomaly = 0 if fault_type == "N" else 1

        for mat_fname in sorted(os.listdir(bearing_dir)):
            if not mat_fname.endswith(".mat"):
                continue
            stem = mat_fname[:-4]  # e.g. "N15_M07_F10_K001_1"

            domain = None
            for d in domain_statistics:
                if stem.startswith(d):
                    domain = d
                    break
            if domain is None:
                continue

            mat_path = os.path.join(bearing_dir, mat_fname)
            try:
                mat = io.loadmat(mat_path)
                y = mat[stem]["Y"][0][0][0]
            except Exception:
                print(f"[WARNING] {mat_fname} 로드 실패 — 스킵")
                continue

            body = None
            for i in range(len(y)):
                if y[i]["Name"] == "vibration_1":
                    body = y[i]["Data"].ravel()
                    break

            if body is None:
                continue

            rpm, torque, force = domain_statistics[domain]
            df["data"].append(body)
            df["fault_type"].append(fault_type)
            df["sampling_rate"].append(64)
            df["rpm"].append(rpm)
            df["load_torque(Nm)"].append(torque)
            df["radial_force(N)"].append(force)
            df["label"].append(label_map[fault_type])
            df["is_anomaly"].append(is_anomaly)

    return pd.DataFrame(df)
