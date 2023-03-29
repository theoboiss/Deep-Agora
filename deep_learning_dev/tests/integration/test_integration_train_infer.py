import os, sys, glob, logging, pytest, shutil


# Neutralise the loggers of the data.preparation package
logging.getLogger('deep_learning_lab.data_preparation').disabled = True


from deep_learning_lab import model
from deep_learning_lab.data_preparation import orchestration, patch


test_workdir = "tests/integration/"

orchestration.RAW_DATA_DIR = test_workdir
patch.RESULT_DIR = os.path.join(test_workdir, "results")

set_labels = ['TextLine']

labels_dir = '_'.join(set_labels)
labels_dir_path = os.path.join(patch.RESULT_DIR, '_'.join(set_labels))
dataset_path = os.path.join(labels_dir_path, "test_data")


@pytest.fixture(scope='session', autouse=True)
def setup_session():
    # Initializes assertions
    is_labels_dir = False
    
    # Tests assertions
    if os.path.isdir(labels_dir_path):
        is_labels_dir = True
        
    assert is_labels_dir, "WARNING: Please execute `test_integration_prep_files` first to patch data"
    
    # Clean the labels directory
    for root, dirs, files in os.walk(labels_dir_path, topdown= False):
        if root != os.path.join(labels_dir_path, "test_data"):
            for name in dirs:
                if name != "test_data":
                    dir_path = os.path.join(root, name)
                    shutil.rmtree(dir_path)
    
    # All the code above has been executed at the beginning of the tests
    
    yield
    
    # All the code bellow will be executed at the end of the tests
    
    pass
    


@pytest.fixture(scope='session')
def setup_train_env():
    """Sets up the test directory by testing if the required data has already been manually downloaded.
    
    Expected structure of the file system at the end:
    integration/           (=test_workdir)
    └── results/           (=patch.RESULT_DIR)
        └── TextLine/      (=labels_dir_path)
            └── test_data/ (=dataset_path)
                ├── images/
                │   ├── image.jpg
                |   └── ...
                ├── labels/
                |   ├── image.png
                |   └── ...
                └── classfile.json
    
    """
    raw_data_dir = os.path.join(test_workdir,"raw_page_dataset")
    raw_data = os.listdir(raw_data_dir)
    assert "M_Aigen_am_Inn_002-01_0000.jpg" in raw_data and "M_Aigen_am_Inn_002-01_0000.xml" in raw_data, \
    "WARNING: Please run `download_test_data.sh`and execute `test_integration_prep_files` first to patch data"
    
    # Initializes assertions
    is_images_dir = is_labels_dir = is_class_file = is_copied_image = is_mask = False
    original_image = os.path.basename(glob.glob(os.path.join(raw_data_dir, '*.jpg'))[0])
    
    # Tests assertions
    if os.path.isdir(os.path.join(dataset_path, "images")):
        is_images_dir = True
    if os.path.isdir(os.path.join(dataset_path, "labels")):
        is_labels_dir = True
    
    for path, currentDirectory, files in os.walk(test_workdir):
        for file in files:
            filepath = os.path.join(path, file)
            if filepath == os.path.join(path, "classfile.json"):
                is_class_file = True
            if filepath == os.path.join(labels_dir_path, "test_data", "images", original_image):
                is_copied_image = True
            if file == original_image[:original_image.find('.jpg')]+'.png':
                is_mask = True
    bools = (is_images_dir, is_labels_dir, is_class_file, is_copied_image, is_mask)
    assert all(bools), "WARNING: Please execute `test_integration_prep_files` first to patch data"


def test_training(setup_train_env):
    """Train a dummy model.
    
    Expected structure of the file system at the end:
    integration/           (=test_workdir)
    └── results/           (=patch.RESULT_DIR)
        └── TextLine/      (=labels_dir_path)
            ├── model/
            ├── tensorboard/
            └── test_data/ (=dataset_path)
                ├── images/
                |   ├── image.jpg
                |   └── ...
                ├── labels/
                |   ├── image.png
                |   └── ...
                ├── classfile.json
                ├── data.csv
                ├── test.csv
                ├── train.csv
                └── val.csv
    
    """
    trainer = model.Trainer(
        labels= set_labels,
        input_dir= "test_data",
        train_ratio= 0.5,
        val_ratio= 0.5,
        preselected_device= 1,
        working_dir= patch.RESULT_DIR
    )
    
    csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
    assert all(
        any(f+'.csv' in csv for csv in csv_files)
        for f in ('data', 'train', 'val')
    ), "Missing data files among 'data.csv', 'train.csv', 'val.csv'"
    
    trainer.train()
    
    model_dir = os.path.join(labels_dir_path, 'model')
    model_files = [name for name in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, name))]
    assert all("checkpoint" in n for n in model_files) and len(model_files), \
    "No model has been serialized or serialized files do not contain 'checkpoint'"
    
    tensor_logs = []
    log_dir = os.path.join(labels_dir_path, "tensorboard", "log")
    if os.path.isdir(log_dir):
        tensor_logs.extend(os.listdir(log_dir))
    assert all("events.out" in n for n in tensor_logs) and len(tensor_logs), "Tensorboard did not work"
    
    
@pytest.fixture(scope='session')
def setup_infer_env():
    """Sets up the test directory by testing if the required model has already been trained.
    
    Expected structure of the file system at the end:
    integration/           (=test_workdir)
    └── results/           (=patch.RESULT_DIR)
        └── TextLine/      (=labels_dir_path)
            ├── model/
            └── test_data/ (=dataset_path)
                └── images/
                    ├── image.jpg
                    └── ...
    
    """
    
    # Initializes assertions
    is_images_dir = False
    
    # Tests assertions
    images_path = os.path.join(dataset_path, "images")
    if os.path.isdir(images_path) and len(os.listdir(images_path)) == 2:
        is_images_dir = True
    
    assert is_images_dir, "WARNING: Please execute `test_integration_prep_files` first to patch data"

    model_dir = os.path.join(labels_dir_path, 'model')
    model_files = [name for name in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, name))]
    assert any("model_checkpoint" in m for m in model_files) and len(model_files), \
    f"No model has been serialized or serialized files. {model_dir} does not contain checkpoint"
    

def test_inference(setup_infer_env):
    """Make inference using a dummy predictor.
    
    Expected structure of the file system at the end:
    integration/           (=test_workdir)
    └── results/           (=patch.RESULT_DIR)
        └── TextLine/      (=labels_dir_path)
            ├── model/
            ├── predictions/
            |   ├── vignette.png
            |   └── ...
            └── test_data/ (=dataset_path)
                └── images/
                    ├── image.jpg
                    └── ...
    
    """
    
    # Makes dummy predictions
    predictor = model.Predictor(
        set_labels,
        input_dir= os.path.join("test_data", "images"),
        preselected_device= 1,
        working_dir= patch.RESULT_DIR
    )
    results = predictor.start(verbose= False)
    
    # Tests assertions
    preds_dir = os.path.join(labels_dir_path, "predictions")
    assert os.path.isdir(preds_dir), "Output of the predictor is absent"
    assert len(results) == len(os.listdir(os.path.join(dataset_path, "images"))), "Inference did not work"
