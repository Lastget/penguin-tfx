from pipeline import configs
from tfx import v1 as tfx

VERTEX_JOB_SPEC = configs.VERTEX_JOB_SPEC
VERTEX_SERVING_SPEC = configs.VERTEX_SERVING_SPEC

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, endpoint_name: str, project_id: str,
                     region: str, use_gpu: bool) -> tfx.dsl.Pipeline:
    
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    if use_gpu:
        VERTEX_JOB_SPEC['worker_pool_specs'][0]['machine_spec'].update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })

    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file = module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5),
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: VERTEX_JOB_SPEC,
            'use_gpu': use_gpu,
        })

    # Vertex AI provides pre-built containers with various configurations for serving.
    # See https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'

    if use_gpu:
        VERTEX_SERVING_SPEC.update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })
        serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'
    
    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs['model'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: VERTEX_SERVING_SPEC,
        })
    

    components = [
      example_gen,
      trainer,
      pusher,
    ]

    return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components)

    # trigger 21
        