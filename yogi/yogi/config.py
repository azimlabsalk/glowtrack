import os

yogi_dir_default = os.path.join(os.path.expanduser('~'), '.yogi')
yogi_dir = os.environ.get('YOGI_DIR', yogi_dir_default)

db_path_default = os.path.join(yogi_dir, 'yogi.db')
db_path = os.environ.get('YOGI_DB_PATH', db_path_default)

db_url_default = 'sqlite:///{}'.format(db_path)
db_url = os.environ.get('YOGI_DB_URL', db_url_default)

# Celery stuff
celery_broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://tesla.snl.salk.edu')
flower_tasks_url = os.environ.get('FLOWER_TASKS_URL', 'http://tesla.snl.salk.edu:5555/api/tasks')


# Model paths
models_dir = os.path.join(yogi_dir, 'models')

pretrained_model_dir = os.path.join(models_dir, 'pretrained')

pretrained_model_paths = {
    'resnet_v1_50': os.path.join(pretrained_model_dir, 'resnet_v1_50.ckpt'),
    'resnet_v1_101': os.path.join(pretrained_model_dir, 'resnet_v1_101.ckpt'),
}

pretrained_model_path = pretrained_model_paths['resnet_v1_101']

# Plot paths

plots_dir = os.path.join(yogi_dir, 'plots')
roc_dir = os.path.join(plots_dir, 'roc')
