import os
PIPELINE_NAME = 'penguin-tfx'
GOOGLE_CLOUD_PROJECT = 'master-host-403612'     
GOOGLE_CLOUD_REGION = 'us-central1'      
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-vertex-default'         

## GCP 
# Paths for users' data.
GCP_DATA_ROOT = 'gs://{}/{}/data'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
GCP_TRAIN_MODULE_FILE = 'gs://{}/{}/pipeline-modules/model.py'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
GCP_MODULE_ROOT = 'gs://{}/{}/pipeline-modules'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
GCP_PIPELINE_ROOT = 'gs://{}/{}/pipeline-root'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
GCP_SERVING_DIR = 'gs://{}/{}/serving-model'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
# Name of Vertex AI Endpoint.
ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME

# LOCAL
LOCAL_TRAIN_MODULE_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'model.py'))
LOCAL_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'data'))


# PIPELINE_IMAGE = 'us-central1-docker.pkg.dev/master-host-403612/cb-tfx/tfx-kfp:latest'
VERTEX_JOB_SPEC = {
    'project': GOOGLE_CLOUD_PROJECT,
    'worker_pool_specs': [{
        'machine_spec': {
            'machine_type': 'e2-standard-4',
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
    'machine_type': 'e2-standard-4',
}

