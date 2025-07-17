import os

def normalize_name(name):
    return '_'.join(name.strip().split())

def allowed_file(filename, allowed={'png', 'jpg', 'jpeg'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

def get_dataset_folders(dataset_dir='dataset'):
    return [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
