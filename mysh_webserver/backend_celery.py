import os
import shutil

import cv2
from celery import Celery, Task

from mysh.metrics.myelin_sheath_metrics import generate_myelin_sheath_metrics_table
from mysh.models.myelin_sheath_detector import create_myelin_mask
from mysh.models.unet_pipeline.unet_myelin_sheath_detector import UnetMyelinSheathDetector

celery = Celery('tasks',
                broker=os.environ.get('CELERY_BROKER_URL', 'pyamqp://guest@localhost//'),
                backend='rpc://')



@celery.task(bind=True)
def process_single_job(self: Task, upload_filename):
    self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Loading image...'})

    input_path = f'uploads/{upload_filename}'
    job_dir = f'jobs/{self.request.id}'
    os.makedirs(job_dir)
    output_path = f'{job_dir}/input.png'
    shutil.copy(input_path, output_path)


    image = cv2.imread(output_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

    def progress_callback(current, total, status="Processing"):
        self.update_state(state='PROGRESS', meta={'current': current, 'total': total, 'status': status})

    detector = UnetMyelinSheathDetector()
    detected_sheaths = detector.detect_myelin_sheaths(image,
                                                      progress_callback=progress_callback,
                                                      inference_batch_size=32)
    myelin_mask_hat = create_myelin_mask(*image.shape[1:], detected_sheaths)

    output_path_mask = f'{job_dir}/output.png'
    cv2.imwrite(output_path_mask, myelin_mask_hat * 255)

    _ = generate_myelin_sheath_metrics_table(
        detected_sheaths,
        output_excel_path=f'{job_dir}/metrics.xlsx'
    )
    del detector
    return self.request.id
