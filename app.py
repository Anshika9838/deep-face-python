import os
import tempfile
from typing import Any, Dict

from deepface.deepface.DeepFace import DeepFace
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="DeepFace Verify API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=500, detail="static/index.html not found")
    return FileResponse(index_path)


@app.post("/api/verify")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
) -> Dict[str, Any]:
    tmp1_path = None
    tmp2_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image1.filename or "image1")[1]) as tmp1:
            tmp1_path = tmp1.name
            tmp1.write(await image1.read())

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image2.filename or "image2")[1]) as tmp2:
            tmp2_path = tmp2.name
            tmp2.write(await image2.read())

        try:
            result = DeepFace.verify(img1_path=tmp1_path, img2_path=tmp2_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DeepFace verification failed: {e}")

        if not isinstance(result, dict):
            return {"verified": False, "raw": result}

        return result
    finally:
        for p in (tmp1_path, tmp2_path):
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass