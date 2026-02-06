import dataclasses
import os
from typing import Literal

from backend_celery import celery

from mysh.logger import get_default_logger


logger = get_default_logger(__name__)




def get_job_status(job_id: str):
    """
    Check the status of a job based on its ID.
    """

    job_files_path = f'jobs/{job_id}'

    if not os.path.isdir(job_files_path):
        # Directory not yet created by the worker
        result = celery.AsyncResult(job_id)
        if result.state == 'PENDING':
            return {'status': 'PROCESSING'}
        return {'status': 'NON_EXISTING'}

    files = os.listdir(job_files_path)
    if [x for x in files if x.startswith('output')]:
        return {'status': 'SUCCESS'}

    result = celery.AsyncResult(job_id)
    if result.state == 'PROGRESS':
        return {
            'status': 'PROGRESS',
            'current': result.info.get('current', 0),
            'total': result.info.get('total', 100),
            'message': result.info.get('status', 'Processing...')
        }
    elif result.state == 'PENDING':
        return {'status': 'PROCESSING'}
    else:
        return {'status': 'ERROR'}


@dataclasses.dataclass
class JobResults:
    input_image_path: str
    mask_path: str
