
from tfx import v1 as tfx

GOOGLE_CLOUD_PROJECT = 'master-host-403612'     
GOOGLE_CLOUD_REGION = 'us-central1'      
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-vertex-default'         

PIPELINE_NAME = 'penguin-vertex-training'


PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/pipeline_module/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' data.
DATA_ROOT = 'gs://{}/data/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Name of Vertex AI Endpoint.
ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME

# PIPELINE_IMAGE = 'us-central1-docker.pkg.dev/master-host-403612/cb-tfx/tfx-kfp:latest'

VERTEX_JOB_SPEC = {
    'project': GOOGLE_CLOUD_PROJECT,
    'worker_pool_specs': [{
        'machine_spec': {
            'machine_type': 'n1-standard-4',
        },
        'replica_count': 1,
        'container_spec': {
            # from public tfx image
            'image_uri': 'gcr.io/tfx-oss-public/tfx:1.14.0',
        },
    }],
}

# For Pusher 
VERTEX_SERVING_SPEC = {
    'project_id': GOOGLE_CLOUD_PROJECT,
    'endpoint_name': ENDPOINT_NAME,
    'machine_type': 'n1-standard-4',
}
