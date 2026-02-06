from pathlib import Path

DEFAULT_MODEL_BASE_PATH = Path(__file__).parents[3] / 'models'
DEFAULT_AXON_MODEL_PATH = (DEFAULT_MODEL_BASE_PATH / 'axon_model_a84b3ae22c3f42708e39c3ca9395b61c.ckpt').absolute()
DEFAULT_MYELIN_MODEL_PATH = (DEFAULT_MODEL_BASE_PATH / 'myelin_model_bc43ef9036dd4622a7b38b8e20f623d4.ckpt').absolute()
