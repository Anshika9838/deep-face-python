import importlib.util
import os
import sys
import tempfile
from typing import Any, Dict

try:
    from deepface import DeepFace  # type: ignore
except Exception:
    _vendored_deepface_path = os.path.join(
        os.path.dirname(__file__), "deepface", "deepface", "DeepFace.py"
    )
    if not os.path.isfile(_vendored_deepface_path):
        raise

    _spec = importlib.util.spec_from_file_location(
        "vendored_deepface_DeepFace", _vendored_deepface_path
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("Failed to load vendored DeepFace module")

    _module = importlib.util.module_from_spec(_spec)
    sys.modules["vendored_deepface_DeepFace"] = _module
    _spec.loader.exec_module(_module)
    DeepFace = _module.DeepFace

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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


@app.get("/compare")
def compare_page() -> FileResponse:
    return index()


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


@app.post("/api/find")
async def find_person(
    image: UploadFile = File(...),
    top_n: int = Query(5, ge=1, le=50),
    enforce_detection: bool = Query(False),
    detector_backend: str = Query("opencv"),
    refresh_db_on_error: bool = Query(True),
) -> Dict[str, Any]:
    db_path = os.path.join(os.path.dirname(__file__), "all_photos")
    if not os.path.isdir(db_path):
        raise HTTPException(status_code=400, detail="Database folder ./all_photos not found")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "image")[1]) as tmp:
            tmp_path = tmp.name
            tmp.write(await image.read())

        def _run_find():
            return DeepFace.find(
                img_path=tmp_path,
                db_path=db_path,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend,
            )

        try:
            dfs = _run_find()
        except Exception as e:
            msg = str(e)
            if (
                refresh_db_on_error
                and "does not match length of index" in msg
                and os.path.isdir(db_path)
            ):
                for name in os.listdir(db_path):
                    if name.startswith("ds_") and name.endswith(".pkl"):
                        try:
                            os.remove(os.path.join(db_path, name))
                        except Exception:
                            pass

                try:
                    dfs = _run_find()
                except Exception as e2:
                    raise HTTPException(status_code=400, detail=f"DeepFace find failed: {e2}")
            else:
                raise HTTPException(status_code=400, detail=f"DeepFace find failed: {e}")

        results = []
        if isinstance(dfs, list):
            for df in dfs:
                try:
                    records = df.head(top_n).to_dict(orient="records")
                except Exception:
                    records = []
                results.append(records)
        else:
            try:
                results = [dfs.head(top_n).to_dict(orient="records")]
            except Exception:
                results = [[]]

        return {
            "db_path": db_path,
            "num_faces": len(results),
            "top_n": top_n,
            "enforce_detection": enforce_detection,
            "detector_backend": detector_backend,
            "refresh_db_on_error": refresh_db_on_error,
            "results": results,
        }

    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass