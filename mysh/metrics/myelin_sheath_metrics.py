from typing import List, Optional

import pandas as pd

from mysh.myelin_sheath import MyelinSheath


def generate_myelin_sheath_metrics_table(detected_sheaths: List[MyelinSheath],
                                         output_excel_path: Optional[str] = None):
    """
    Aggregate metrics for a list of detected myelin sheaths.

    Parameters:
        detected_sheaths (List[MyelinSheath]): List of detected myelin sheaths.
        output_excel_path (Optional[str]): Path to save the aggregated metrics as an Excel file.
            If None, the metrics will not be saved to a file.
    Returns:
        dict: Dictionary containing aggregated metrics.
    """
    rows = []

    for idx, sheath in enumerate(detected_sheaths):
        g_ratio = sheath.g_ratio
        axon_metrics = sheath.axon_metrics
        myelin_metrics = sheath.myelin_metrics

        rows.append({
            "sheath_id": idx + 1,
            "confidence": sheath.confidence,
            "g_ratio": g_ratio,
            "axon_area": axon_metrics['area'],
            "axon_perimeter": axon_metrics['perimeter'],
            "axon_circularity": axon_metrics['circularity'],
            "myelin_area": myelin_metrics['area'],
            "myelin_perimeter": myelin_metrics['perimeter'],
            "myelin_circularity": myelin_metrics['circularity'],
            "mnf_area": sheath.mnf_area
        })


    df = pd.DataFrame(rows)
    if output_excel_path:
        df.to_excel(output_excel_path, index=False)
    return df
