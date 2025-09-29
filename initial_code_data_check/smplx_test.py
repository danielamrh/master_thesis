import os
import smplx

model_path = os.environ.get("SMPLX_MODEL_DIR", "")
if model_path:
    print(f"Loading model from: {model_path}")
    try:
        smpl = smplx.create(model_path=model_path, model_type='smpl')
        print("SMPL model loaded successfully.")
    except Exception as e:
        print(f"Error loading SMPL model: {e}")
else:
    print("SMPLX_MODEL_DIR environment variable is not set.")
