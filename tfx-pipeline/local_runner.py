import os
from absl import logging
from pipeline import configs
from pipeline import penguin_trainer
from tfx import v1 as tfx


# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.
OUTPUT_DIR = '.'

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to SQLite database in METADATA_PATH.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

PIPELINE_DEFINITION_FILE = configs.PIPELINE_NAME  + '_pipeline.json'


def run():
    """Define a local pipeline."""
    my_pipeline =  penguin_trainer._create_pipeline(
            pipeline_name = configs.PIPELINE_NAME,
            pipeline_root = PIPELINE_ROOT,
            data_root = configs.LOCAL_DATA_PATH,
            module_file = configs.LOCAL_TRAIN_MODULE_FILE,
            serving_model_dir = SERVING_MODEL_DIR,
            use_gpu = False)
    tfx.orchestration.LocalDagRunner().run(my_pipeline)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()