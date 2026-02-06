# mysh-webserver

Internal web server for presenting the `mysh` ML model (myelin sheath segmentation). Built using FastAPI with a simple web-based interface for end users.

## Overview

This web server exposes a browser-accessible UI that allows users to:

- Upload an microscopy image
- Receive a UUID corresponding to their processing job
- Return later using the UUID to view results 
- Download results: semantic segmentation mask and Excel file with metrics calculated for each individual fibre

No authentication or direct API access is exposed to users.
