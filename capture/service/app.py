from fastapi import FastAPI, UploadFile

from . import store

app = FastAPI(title="Capture Service")


@app.post("/upload/{scan_id}")
async def upload_chunk(scan_id: str, file: UploadFile, index: int = 0):
    store.save_chunk(scan_id, index, await file.read())
    return {"scan_id": scan_id, "index": index, "status": "received"}


@app.post("/meta/{scan_id}")
async def upload_meta(scan_id: str, meta: dict):
    store.save_meta(scan_id, meta)
    return {"scan_id": scan_id, "status": "meta_saved"}


@app.post("/complete/{scan_id}")
async def complete(scan_id: str):
    return {"scan_id": scan_id, "status": "complete", "frame_count": store.complete(scan_id)}


@app.get("/scan/{scan_id}")
async def scan_status(scan_id: str):
    return store.status(scan_id)
