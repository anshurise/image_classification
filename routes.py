from fastapi import APIRouter, UploadFile
from prediction import predict
import shutil
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/predict/")
async def upload_image(file: UploadFile):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict using the model
        result = predict(str(file_path), "models/cat_dog_model.h5")
        return {"file": file.filename, "result": result}

    except Exception as e:
        return {"error": str(e)}
