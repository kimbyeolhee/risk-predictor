from .visual_dataset import VisualDataset

def load_dataset(data_type: str, patient_id_csv_path: str, video_csv_path: str, cfg: dict):
    if data_type == "visual":
        dataset = VisualDataset(patient_id_csv_path, video_csv_path, cfg)
    elif data_type == "text":
        pass
    elif data_type == "tabular":
        pass
    else:
        raise KeyError(f"Invalid data type: {data_type}")
    return dataset