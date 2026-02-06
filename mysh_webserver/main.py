import os
import uuid
from pathlib import Path
import io

from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd

from backend_celery import process_single_job
from mysh.logger import get_default_logger
from mysh_webserver.utils import get_job_status


logger = get_default_logger(__name__)
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/",response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={"title": "MYSH: Upload"}
    )


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Handle file upload and processing, always save as PNG"""

    if not file.content_type or not file.content_type.startswith('image/'):
        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "title": "mysh Image Segmentation",
                "error": "Please upload a valid image file"
            }
        )

    upload_id = str(uuid.uuid4())
    upload_filename = f'{upload_id}.png'
    upload_path = Path("uploads") / upload_filename

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGBA")

        image.save(upload_path, format="PNG")

        result = process_single_job.delay(upload_filename)
        job_id = result.id
        return RedirectResponse(url=f"/result/{job_id}", status_code=303)

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "title": "mysh Image Segmentation",
                "error": f"Processing failed: {str(e)}"
            }
        )


@app.get("/result/{job_id}", response_class=HTMLResponse)
async def show_result(request: Request, job_id: str):
    """Display processing results"""

    status = get_job_status(job_id)
    if status['status'] == 'NON_EXISTING':
        raise HTTPException(status_code=404, detail="Job not found")

    metrics_averages = None
    metrics_available = False
    metrics_download_url = None
    metrics_path = Path('jobs') / job_id / 'metrics.xlsx'
    if metrics_path.exists():
        try:
            df = pd.read_excel(metrics_path)
            cols = [c for c in df.columns if c not in ('sheath_id', 'confidence')]
            if cols:
                metrics_averages = df[cols].mean().to_dict()
                metrics_available = True
                metrics_download_url = f"/download_metrics/{job_id}"
        except Exception as e:
            logger.error(f"Error reading metrics for job {job_id}: {e}")
            metrics_averages = None
            metrics_available = False
            metrics_download_url = None
    else:
        logger.warning(f"No metrics file found for job {job_id}")

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "title": "Processing Results",
            "job_id": job_id,
            "job": None,
            "status": status['status'],
            "metrics_averages": metrics_averages,
            "metrics_available": metrics_available,
            "metrics_download_url": metrics_download_url
        }
    )


@app.get("/status/{job_id}")
async def get_job_status_api(job_id: str):
    return get_job_status(job_id)

@app.get("/orig/{job_id}")
async def serve_result_image(job_id: str):
    """Serve result image for display in web page"""
    result_path = Path('jobs') / job_id / 'input.png'
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path=result_path, media_type='image/png')

@app.get("/mask/{job_id}")
async def serve_result_image(job_id: str):
    """Serve result image for display in web page"""
    result_path = Path('jobs') / job_id / 'output.png'
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path=result_path, media_type='image/png')


@app.get("/download/{job_id}")
async def serve_result_image(job_id: str):
    """Serve result image for display in web page"""
    result_path = Path('jobs') / job_id / 'output.png'
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path=result_path, media_type='image/png', filename=f"{job_id}_mask.png")


@app.get("/download_metrics/{job_id}")
async def download_metrics(job_id: str):
    metrics_path = Path('jobs') / job_id / 'metrics.xlsx'
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    return FileResponse(path=metrics_path,
                        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        filename=f"{job_id}_metrics.xlsx")

