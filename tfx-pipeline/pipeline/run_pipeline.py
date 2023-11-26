from tfx import v1 as tfx
from typing import Optional, Text, Dict
from tfx.components import CsvExampleGen, Trainer, Pusher
from ml_metadata.proto import metadata_store_pb2
from absl import logging



def _create_pipeline(pipeline_name: Text, 
                     pipeline_root: Text, 
                     data_root: Text,
                     training_module: Text,
                     serving_model_dir: Optional[Text] = None,
                     vertex_job_spec: Optional[Dict[Text,Text]] = None,
                     vertex_serving_spec: Optional[Dict[Text,Text]] = None,
                     region: Optional[Text] = 'us-central1', 
                     use_gpu: bool = False,
                     metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
                     ) -> tfx.dsl.Pipeline:
    
    example_gen = CsvExampleGen(input_base = data_root)
    logging.info("Pipeline example GEN done.")

    # Trainer 
    trainer_args = dict(
        module_file = training_module,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5),
        custom_config = {'use_gpu': use_gpu }
    )

    if use_gpu:
        vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })

    if vertex_job_spec is not None:
        trainer_args['custom_config']= {
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_job_spec,
            'use_gpu': use_gpu,
        }

        trainer = tfx.extensions.google_cloud_ai_platform.Trainer(**trainer_args)
        logging.info("Pipeline vertex Trainer done.")

    else:
        trainer = Trainer(**trainer_args)
        logging.info("Pipeline local Trainer done.")


    # Vertex AI provides pre-built containers with various configurations for serving.
    # See https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    pusher_args ={
      'model': trainer.outputs['model'],
    } 

    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'

    if use_gpu:
        vertex_serving_spec.update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })
        # gpu
        serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'

    if vertex_serving_spec is not None:
        pusher_args['custom_config'] = {
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: vertex_serving_spec,
        }    
        pusher = tfx.extensions.google_cloud_ai_platform.Pusher(**pusher_args)
        logging.info("Pipeline vertex Pusher done.")
    else:
        pusher_args['push_destination'] = tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(base_directory = serving_model_dir)
            )
        pusher = Pusher(**pusher_args)
        logging.info("Pipeline local Pusher done.")

    

    components = [
      example_gen,
      trainer,
      pusher,
    ]

    return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      metadata_connection_config = metadata_connection_config)

    # trigger 28
        