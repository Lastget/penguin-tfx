import os 
from pipeline import configs, penguin_trainer
from tfx import v1 as tfx
from absl import logging

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import logging


PIPELINE_DEFINITION_FILE = configs.PIPELINE_NAME  + '_pipeline.json'

def run():
    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
        output_filename=PIPELINE_DEFINITION_FILE)
    _ = runner.run(
        penguin_trainer._create_pipeline(
            pipeline_name = configs.PIPELINE_NAME,
            pipeline_root = configs.PIPELINE_ROOT,
            data_root = configs.DATA_ROOT,
            module_file = os.path.join(configs.MODULE_ROOT, 'penguine_trainer.py'),
            endpoint_name = configs.ENDPOINT_NAME,
            project_id = configs.GOOGLE_CLOUD_PROJECT,
            region = configs.GOOGLE_CLOUD_REGION,
            use_gpu = False))

def submit():
    aiplatform.init(project = configs.GOOGLE_CLOUD_PROJECT, location=configs.GOOGLE_CLOUD_REGION)

    job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                    display_name=configs.PIPELINE_NAME)
    job.submit()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
    submit()


