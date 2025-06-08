import pandas as pd
import torch
import pickle
from model.FFT_model import FTTransformer

FEATURE_COLS = [
    'Integral_Total_HU', 'Kurtosis', 'Max_HU', 'Mean_HU', 'Median_HU', 'Min_HU',
    'Skewness', 'Sphere_Diameter', 'HU_STD', 'Total_HU', 'Surface Area',
    'Convex_Hull_Ratio', 'Sphericity', 'Major_Axis', 'Minor_Axis', 'Volume',
    'Eccentricity', 'Fourier_Very_Low', 'Fourier_Low', 'Fourier_Mid_Low',
    'Fourier_Mid_High', 'Fourier_High', 'HU_Histogram_1', 'HU_Histogram_2',
    'HU_Histogram_3', 'HU_Histogram_4', 'HU_Histogram_5'
]

def predict_with_model(xlsx_path, model_path, scaler_path, log_callback=None):
    try:
        if log_callback:
            log_callback("노스트라사무스 집중 중...")

        df = pd.read_excel(xlsx_path)

        model = FTTransformer(input_dim=len(FEATURE_COLS))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        X = df[FEATURE_COLS].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_tensor).numpy().squeeze()

        if probs.ndim == 0:
            df["Prediction"] = ["비정상" if probs > 0.5 else "정상"]
            df["Probability"] = [float(probs)]
        else:
            df["Prediction"] = ["비정상" if p > 0.5 else "정상" for p in probs]
            df["Probability"] = probs

        df.to_excel(xlsx_path, index=False)
        if log_callback:
            log_callback(f"돗자리 펴야겠네: {xlsx_path}")
        return True

    except Exception as e:
        if log_callback:
            log_callback(f"돌팔이었습니다...: {str(e)}")
        return False
