# `mysh`: MYelin SHeath Morphology

`mysh` (pronounced *mɨʃ*, in polish [mysz](https://pl.wikipedia.org/wiki/Mysz)) is 
an application for detection of morphology of myelin sheaths from microscopic pictures
of nerves.

## Running with Docker Compose

```bash
docker compose up --build
```

The web UI will be available at http://localhost:8000.
Upload a microscopic nerve image and the system will detect myelin sheaths,
produce a segmentation mask, and calculate per-sheath morphology metrics.

To use a different port:

```bash
PORT=3000 docker compose up --build
```
