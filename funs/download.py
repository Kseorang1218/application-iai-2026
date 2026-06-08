"""데이터셋 다운로드 및 DataFrame 변환 모듈."""
import os

import numpy as np
import pandas as pd
from scipy import io

def download_cwru(root: str, sample_rate: str = "12k") -> pd.DataFrame:
    """
    이상 탐지 및 결함 진단용 CWRU (Case Western Reserve University) 데이터셋을 다운로드

    Reference: https://github.com/junior209lsj/FaultDiagnosisOptimizerBenchmark

    Parameters
    ----------
    root : str
        데이터 파일을 저장할 루트 디렉토리
    sample_rate : str
        샘플링 레이트. "12k" 또는 "48k"

    Returns
    ----------
    pd.DataFrame
        CWRU 데이터셋의 데이터 세그먼트를 포함하는 DataFrame
    """
    BASE_URL = "https://engineering.case.edu/sites/default/files/"

    if sample_rate == "48k":
        filenames = {
            # 정상
            "97.mat": "N_000_0", "98.mat": "N_000_1", "99.mat": "N_000_2", "100.mat": "N_000_3",
            # 내륜 결함
            "109.mat": "IR_007_0", "110.mat": "IR_007_1", "111.mat": "IR_007_2", "112.mat": "IR_007_3",
            "174.mat": "IR_014_0", "175.mat": "IR_014_1", "176.mat": "IR_014_2", "177.mat": "IR_014_3",
            "213.mat": "IR_021_0", "214.mat": "IR_021_1", "215.mat": "IR_021_2", "217.mat": "IR_021_3",
            # 외륜 결함 @06
            "135.mat": "OR@06_007_0", "136.mat": "OR@06_007_1", "137.mat": "OR@06_007_2", "138.mat": "OR@06_007_3",
            "201.mat": "OR@06_014_0", "202.mat": "OR@06_014_1", "203.mat": "OR@06_014_2", "204.mat": "OR@06_014_3",
            "238.mat": "OR@06_021_0", "239.mat": "OR@06_021_1", "240.mat": "OR@06_021_2", "241.mat": "OR@06_021_3",
            # 볼 결함
            "122.mat": "B_007_0", "123.mat": "B_007_1", "124.mat": "B_007_2", "125.mat": "B_007_3",
            "189.mat": "B_014_0", "190.mat": "B_014_1", "191.mat": "B_014_2", "192.mat": "B_014_3",
            "226.mat": "B_021_0", "227.mat": "B_021_1", "228.mat": "B_021_2", "229.mat": "B_021_3",
        }
    else:
        filenames = {
            # 정상
            "97.mat": "N_000_0", "98.mat": "N_000_1", "99.mat": "N_000_2", "100.mat": "N_000_3",
            # 내륜 결함
            "105.mat": "IR_007_0", "106.mat": "IR_007_1", "107.mat": "IR_007_2", "108.mat": "IR_007_3",
            "169.mat": "IR_014_0", "170.mat": "IR_014_1", "171.mat": "IR_014_2", "172.mat": "IR_014_3",
            "209.mat": "IR_021_0", "210.mat": "IR_021_1", "211.mat": "IR_021_2", "212.mat": "IR_021_3",
            "3001.mat": "IR_028_0", "3002.mat": "IR_028_1", "3003.mat": "IR_028_2", "3004.mat": "IR_028_3",
            # 외륜 결함 @06
            "130.mat": "OR@06_007_0", "131.mat": "OR@06_007_1", "132.mat": "OR@06_007_2", "133.mat": "OR@06_007_3",
            "197.mat": "OR@06_014_0", "198.mat": "OR@06_014_1", "199.mat": "OR@06_014_2", "200.mat": "OR@06_014_3",
            "234.mat": "OR@06_021_0", "235.mat": "OR@06_021_1", "236.mat": "OR@06_021_2", "237.mat": "OR@06_021_3",
            # 외륜 결함 @03
            "144.mat": "OR@03_007_0", "145.mat": "OR@03_007_1", "146.mat": "OR@03_007_2", "147.mat": "OR@03_007_3",
            "246.mat": "OR@03_021_0", "247.mat": "OR@03_021_1", "248.mat": "OR@03_021_2", "249.mat": "OR@03_021_3",
            # 외륜 결함 @12
            "156.mat": "OR@12_007_0", "158.mat": "OR@12_007_1", "159.mat": "OR@12_007_2", "160.mat": "OR@12_007_3",
            "258.mat": "OR@12_021_0", "259.mat": "OR@12_021_1", "260.mat": "OR@12_021_2", "261.mat": "OR@12_021_3",
            # 볼 결함
            "118.mat": "B_007_0", "119.mat": "B_007_1", "120.mat": "B_007_2", "121.mat": "B_007_3",
            "185.mat": "B_014_0", "186.mat": "B_014_1", "187.mat": "B_014_2", "188.mat": "B_014_3",
            "222.mat": "B_021_0", "223.mat": "B_021_1", "224.mat": "B_021_2", "225.mat": "B_021_3",
            "3005.mat": "B_028_0", "3006.mat": "B_028_1", "3007.mat": "B_028_2", "3008.mat": "B_028_3",
        }

    # 999: 다중 클래스 분류에서 사용하지 않는 조합 (학습/평가 시 제외 대상)
    label_map = {
        ("N", "000"): 0, ("B", "007"): 1, ("B", "014"): 2, ("B", "021"): 3, ("B", "028"): 999,
        ("IR", "007"): 4, ("IR", "014"): 5, ("IR", "021"): 6, ("IR", "028"): 999,
        ("OR@03", "007"): 999, ("OR@03", "014"): 999, ("OR@03", "021"): 999, ("OR@03", "028"): 999,
        ("OR@06", "007"): 7, ("OR@06", "014"): 8, ("OR@06", "021"): 9, ("OR@06", "028"): 999,
        ("OR@12", "007"): 999, ("OR@12", "014"): 999, ("OR@12", "021"): 999, ("OR@12", "028"): 999,
    }

    if not os.path.isdir(root):
        os.makedirs(root)

    load_to_rpm = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

    # 데이터 프레임 구조에 is_anomaly 추가
    df = {
        "data": [],
        "fault_type": [],
        "crack_size": [],
        "rpm": [],
        "label": [],
        "is_anomaly": []
    }

    missing = [(k, v) for k, v in filenames.items()
               if not os.path.isfile(os.path.join(root, v + ".mat"))]
    if not missing:
        print(f"[CWRU] 모든 .mat 파일이 이미 존재함 — 다운로드 스킵 ({root})")
    else:
        print(f"[CWRU] {len(missing)}/{len(filenames)}개 파일 다운로드 필요")

    for key, value in filenames.items():
        filename = os.path.join(root, value + ".mat")
        if not os.path.isfile(filename):
            os.system(f"wget -O {filename} {BASE_URL + key}")

        try:
            data = io.loadmat(filename)
        except Exception:
            print(f"[CWRU] {value}.mat 손상됨 — 재다운로드 중...")
            os.remove(filename)
            os.system(f"wget -O {filename} {BASE_URL + key}")
            data = io.loadmat(filename)
        body = None
        for elem in data.keys():
            if "DE" in elem:
                body = data[elem]

        if body is None:
            print(f"[WARNING] 'DE' 채널을 찾을 수 없음: {value}.mat — 스킵")
            continue

        body = np.ravel(body, order="F")

        labels = value.split("_")

        # 다중 클래스 레이블 추출
        label = label_map[(labels[0], labels[1])]

        # 이상 탐지용 이진 레이블 할당 (정상 'N'은 0, 그 외 결함은 1)
        is_anomaly = 0 if labels[0] == "N" else 1

        df["fault_type"].append(labels[0])
        df["crack_size"].append(labels[1])
        df["rpm"].append(load_to_rpm[int(labels[2])])
        df["label"].append(label)
        df["is_anomaly"].append(is_anomaly)
        df["data"].append(body)

    data_frame = pd.DataFrame(df)

    return data_frame
