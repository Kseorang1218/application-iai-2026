"""데이터셋 다운로드 및 DataFrame 변환 모듈."""
import glob
import os
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import patoolib
import requests
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


def download_paderborn(root: str, sample: bool = False) -> pd.DataFrame:
    """
    Paderborn University 데이터셋 다운로드
    Reference: https://github.com/junior209lsj/FaultDiagnosisOptimizerBenchmark

    Parameters
    ----------
    root: str
        데이터 파일을 저장할 루트 디렉토리
    sample: bool
        True이면 각 베어링당 첫 번째 데이터 파일만 사용

    Returns
    ----------
    pd.DataFrame
        Paderborn University 데이터셋의 데이터 세그먼트를 포함하는 DataFrame
    """
    sampling_rate = 64  # kHz
    
    url = "https://groups.uni-paderborn.de/kat/BearingDataCenter"
    filenames = [
        ("K001", "N"),
        ("K002", "N"),
        ("K003", "N"),
        ("K004", "N"),
        ("K005", "N"),
        ("K006", "N"),
        ("KI04", "IR"),
        ("KI14", "IR"),
        ("KI16", "IR"),
        ("KI17", "IR"),
        ("KI18", "IR"),
        ("KI21", "IR"),
        ("KA04", "OR"),
        ("KA15", "OR"),
        ("KA16", "OR"),
        ("KA22", "OR"),
        ("KA30", "OR"),
    ]

    label_map = {"N": 0, "IR": 1, "OR": 2}

    domains = [
        "N15_M07_F10",
        "N09_M07_F10",
    ]

    domain_statistics = {
        "N15_M07_F10": (1500, 0.7, 1000),
        "N09_M07_F10": (900, 0.7, 1000),
    }

    if sample:
        sample_list = [1]
    else:
        sample_list = [x + 1 for x in range(20)]

    if not os.path.isdir(root):
        os.makedirs(root)

    # 데이터 프레임 구조에 is_anomaly 추가
    df = {
        "data": [],
        "fault_type": [],
        "sampling_rate": [],
        "rpm": [],
        "load_torque(Nm)": [],
        "radial_force(N)": [],
        "label": [],
        "is_anomaly": [],
    }

    missing = [f for f in filenames if not os.path.isdir(f"{root}/{f[0]}")]
    if not missing:
        print(f"[PU] 모든 베어링 폴더가 이미 존재함 — 다운로드 스킵 ({root})")
    else:
        print(f"[PU] {len(missing)}/{len(filenames)}개 베어링 다운로드/추출 필요")

    for filename in filenames:
        bearing_dir = f"{root}/{filename[0]}"
        rar_path = f"{root}/{filename[0]}.rar"
        if not os.path.isdir(bearing_dir):
            if not os.path.isfile(rar_path):
                print(f"  {filename[0]}.rar 다운로드 중 ...")
                os.system(f"wget -O {rar_path} {url}/{filename[0]}.rar")
            patoolib.extract_archive(rar_path, outdir=root, interactive=False)
            os.remove(rar_path)

        # 이상 탐지용 이진 레이블 생성 (N이면 0, 그 외 결함이면 1)
        is_anomaly = 0 if filename[1] == "N" else 1

        for domain in domains:
            for data_num in sample_list:
                data = io.loadmat(
                    f"{root}/{filename[0]}/{domain}_{filename[0]}_{data_num}.mat"
                )
                y = data[f"{domain}_{filename[0]}_{data_num}"]["Y"][0][0][0]

                body = None
                for i in range(len(y)):
                    flag = y[i]["Name"]
                    if flag == "vibration_1":
                        body = y[i]["Data"].ravel()
                        break
                
                # 진동 데이터를 찾지 못한 경우의 에러 방지
                if body is None:
                    continue

                df["data"].append(body)
                df["fault_type"].append(filename[1])
                df["sampling_rate"].append(sampling_rate)
                df["rpm"].append(domain_statistics[domain][0])
                df["load_torque(Nm)"].append(domain_statistics[domain][1])
                df["radial_force(N)"].append(domain_statistics[domain][2])
                df["label"].append(label_map[filename[1]])
                df["is_anomaly"].append(is_anomaly) # 이진 레이블 데이터 추가

    data_frame = pd.DataFrame(df)

    return data_frame


def download_uos(root: str, sample_rate: str = "16k") -> pd.DataFrame:
    """
    이상 탐지 및 결함 진단용 University of Seoul (UOS) 멀티 도메인 진동 데이터셋을 다운로드
    모든 베어링 타입(6204, N204, NJ204, 30204)을 다운로드하여 단일 DataFrame으로 병합

    Parameters
    ----------
    root : str
        데이터 파일을 저장할 루트 디렉토리
    sample_rate : str
        샘플링 레이트. "8k" 또는 "16k"

    Returns
    ----------
    pd.DataFrame
        UOS 데이터셋의 데이터 세그먼트를 포함하는 DataFrame
    """
    dataset_id_map = {
        "6204": "53vtnjy6c6",
        "N204": "7trwzz77xh",
        "NJ204": "7trwzz77xh",
        "30204": "2cygy6y4rk",
    }
    BASE_URL = "https://data.mendeley.com/public-api/zip/{}/download/1"
 
    sr_map = {"8k": "8", "16k": "16"}
    sr_value = sr_map.get(sample_rate, "16")
    sr_dir = f"SamplingRate_{sr_value}000"
 
    rotating_speeds = [600, 800, 1000, 1200, 1400, 1600]
    rotating_component_conditions = ["H"]  # 축 결함(L, U, M) 제외, 베어링 결함만 사용
 
    bearing_label_map = {"H": 0, "B": 1, "IR": 2, "OR": 3}
    rotating_label_map = {
        "H": 0, "L": 1, "U1": 2, "U2": 3, "U3": 4, "M1": 5, "M2": 6, "M3": 7,
    }
 
    if not os.path.isdir(root):
        os.makedirs(root)
 
    # 3개의 고유한 ZIP 파일 식별자를 순회하며 모든 서브셋 다운로드
    unique_dataset_ids = list(set(dataset_id_map.values()))
 
    # dataset_id → 베어링 타입 역매핑 (추출 완료 여부 확인용)
    id_to_models = {}
    for model, did in dataset_id_map.items():
        id_to_models.setdefault(did, []).append(model)

    pending_ids = []
    for dataset_id in unique_dataset_ids:
        sample_model = id_to_models[dataset_id][0]
        sample_pattern = os.path.join(root, sr_dir, "**", f"*_{sample_model}_*.mat")
        if not glob.glob(sample_pattern, recursive=True):
            pending_ids.append(dataset_id)

    if not pending_ids:
        print(f"[UOS] 모든 dataset_id가 이미 추출됨 — 다운로드 스킵 ({root})")
    else:
        print(f"[UOS] {len(pending_ids)}/{len(unique_dataset_ids)}개 dataset_id 다운로드/추출 필요")

    for dataset_id in unique_dataset_ids:
        zip_path = os.path.join(root, f"{dataset_id}.zip")

        if dataset_id not in pending_ids:
            continue

        if not os.path.isfile(zip_path):
            url = BASE_URL.format(dataset_id)
            print(f"  {url} 다운로드 중 ...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/91.0.4472.124 Safari/537.36'
            }
            try:
                resp = requests.get(url, headers=headers, stream=True, timeout=60)
                resp.raise_for_status()
                total_size = int(resp.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                sys.stdout.write(
                                    f"\r  진행률: {downloaded}/{total_size} bytes "
                                    f"({downloaded*100//total_size}%)")
                            else:
                                sys.stdout.write(f"\r  다운로드됨: {downloaded} bytes")
                            sys.stdout.flush()
                print()  # 진행률 출력 후 줄바꿈
            except requests.exceptions.RequestException as e:
                print(f"\n[ERROR] {dataset_id} 다운로드 실패: {e}")
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
                continue

        temp_extract_dir = os.path.join(root, f"temp_{dataset_id}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(temp_extract_dir)

            # 임시 디렉토리에서 대상 주파수 폴더를 찾아 병합
            for target_sr in ["SamplingRate_8000", "SamplingRate_16000"]:
                sr_folders = glob.glob(os.path.join(temp_extract_dir, "**", target_sr), recursive=True)
                for src_sr_dir in sr_folders:
                    dst_sr_dir = os.path.join(root, target_sr)
                    os.makedirs(dst_sr_dir, exist_ok=True)

                    for rpm_folder in os.listdir(src_sr_dir):
                        src_rpm_dir = os.path.join(src_sr_dir, rpm_folder)
                        dst_rpm_dir = os.path.join(dst_sr_dir, rpm_folder)

                        if os.path.isdir(src_rpm_dir):
                            os.makedirs(dst_rpm_dir, exist_ok=True)
                            for mat_file in os.listdir(src_rpm_dir):
                                src_file = os.path.join(src_rpm_dir, mat_file)
                                dst_file = os.path.join(dst_rpm_dir, mat_file)
                                if not os.path.exists(dst_file):
                                    shutil.move(src_file, dst_file)

            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            os.remove(zip_path)
        except zipfile.BadZipFile:
            print(f"[WARNING] 잘못된 zip 파일: {zip_path} — 스킵")
            continue
 
    df = {
        "data": [],
        "bearing_condition": [],
        "rotating_component_condition": [],
        "bearing_type": [],
        "sampling_rate_kHz": [],
        "rpm": [],
        "bearing_label": [],
        "rotating_label": [],
        "is_anomaly": [],
        "fault_type": [],
    }
 
    # 베어링 모델별 측정된 결함 종류 매핑
    # N204: healthy, ball, outer race / NJ204: inner race 전용 / 6204, 30204: 전체
    bearing_type_conditions = {
        "6204":  ["H", "B", "IR", "OR"],
        "N204":  ["H", "B", "OR"],
        "NJ204": ["IR"],
        "30204": ["H", "B", "IR", "OR"],
    }
 
    for speed in rotating_speeds:
        speed_dir = os.path.join(root, sr_dir, f"RotatingSpeed_{speed}")
 
        if not os.path.isdir(speed_dir):
            continue
 
        for rot_cond in rotating_component_conditions:
            for model_name, bearing_conditions in bearing_type_conditions.items():
                for bear_cond in bearing_conditions:
                    filename = f"{rot_cond}_{bear_cond}_{sr_value}_{model_name}_{speed}.mat"
                    filepath = os.path.join(speed_dir, filename)
 
                    if not os.path.isfile(filepath):
                        continue
 
                    try:
                        mat_data = io.loadmat(filepath)
                    except Exception:
                        continue
 
                    if "Data" not in mat_data:
                        continue
 
                    body = np.ravel(mat_data["Data"], order="F")
 
                    bearing_label = bearing_label_map[bear_cond]
                    rotating_label = rotating_label_map[rot_cond]
                    is_anomaly = 0 if (bear_cond == "H" and rot_cond == "H") else 1
 
                    df["data"].append(body)
                    df["bearing_condition"].append(bear_cond)
                    df["rotating_component_condition"].append(rot_cond)
                    df["bearing_type"].append(model_name)
                    df["sampling_rate_kHz"].append(int(sr_value))
                    df["rpm"].append(speed)
                    df["bearing_label"].append(bearing_label)
                    df["rotating_label"].append(rotating_label)
                    df["is_anomaly"].append(is_anomaly)
                    df["fault_type"].append(bear_cond)
 
    data_frame = pd.DataFrame(df)
    return data_frame

if __name__=="__main__":
    uos_df = download_uos("./dataset/uos")