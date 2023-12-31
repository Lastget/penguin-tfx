import os 
from pipeline import configs, run_pipeline
from tfx import v1 as tfx
from absl import logging
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs


PIPELINE_DEFINITION_FILE = configs.PIPELINE_NAME  + '_pipeline.json'

# TFX produces two types of outputs, files and metadata.


def run():
    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
        output_filename=PIPELINE_DEFINITION_FILE)
    _ = runner.run(
        run_pipeline._create_pipeline(
            pipeline_name = configs.PIPELINE_NAME,
            pipeline_root = configs.GCP_PIPELINE_ROOT,
            data_root = configs.GCP_DATA_ROOT,
            training_module = os.path.join(configs.GCP_MODULE_ROOT, 'model.py'),
            vertex_job_spec = configs.VERTEX_JOB_SPEC,
            vertex_serving_spec = configs.VERTEX_SERVING_SPEC,
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

# trigger 7
