import os, glob, shutil, pytest

from deep_learning_lab import model
from deep_learning_lab.data_preparation import orchestration, patch


test_workdir = "tests/integration/" # path to the directory where integration tests will be run

orchestration.RAW_DATA_DIR = test_workdir # path to the directory containing the raw data used for testing
patch.RESULT_DIR = os.path.join(test_workdir, "results") # path to the directory where the results of the testing will be stored

set_labels = ['TextLine'] # list of strings containing the labels that will be used in the testing

labels_dir = '_'.join(set_labels) # name of the directory of test results specified by the set of labels
labels_dir_path = os.path.join(patch.RESULT_DIR, '_'.join(set_labels))# path to the directory of test results specified by the set of labels
dataset_path = os.path.join(labels_dir_path, "test_data") # path of the directory containing the test data used in the testing


@pytest.fixture(scope='session', autouse=True)
def setup_session():
    """Sets up the session fixture for the integration test.

    The function initializes and tests the assertions, and cleans up the labels directory.
    The assertions check if the directory path for the labels exists, and raise a warning if test_integration_prep_files
    is not executed prior to running the tests. The labels directory is then cleaned up using shutil.rmtree.

    The directory structure of the file system is expected to be like this:

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

    Returns:
        The setup function as a fixture for the integration test.

    Raises:
        AssertionError: If the directory path for the labels does not exist.
        
    """
    
    # Initialize assertions
    is_labels_dir = False
    
    # Test assertions
    if os.path.isdir(labels_dir_path):
        is_labels_dir = True
        
    assert is_labels_dir, "WARNING: Please execute `test_integration_prep_files` first to patch data"
    
    # Clean the labels directory
    for root, dirs, _ in os.walk(labels_dir_path, topdown= False):
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
    """Sets up the test environment by verifying if the required data has been manually downloaded.

    The function asserts the expected directory structure of the file system:
    
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
    
    Raises:
        AssertionError: If any of the assertions fail, the test environment setup is incomplete.
            - When the downloaded files are not present.
            - When the expected directory structure is not found.
        
    """

    # Check if the required data files exist
    raw_data_dir = os.path.join(test_workdir,"raw_page_dataset")
    raw_data = os.listdir(raw_data_dir)
    assert "M_Aigen_am_Inn_002-01_0000.jpg" in raw_data and "M_Aigen_am_Inn_002-01_0000.xml" in raw_data, \
    "WARNING: Please run `download_test_data.sh`and execute `test_integration_prep_files` first to patch data"
    
    # Initialize assertions
    is_images_dir = is_labels_dir = is_class_file = is_copied_image = is_mask = False
    original_image = os.path.basename(glob.glob(os.path.join(raw_data_dir, '*.jpg'))[0])
    
    # Check if images and labels directory exist
    if os.path.isdir(os.path.join(dataset_path, "images")):
        is_images_dir = True
    if os.path.isdir(os.path.join(dataset_path, "labels")):
        is_labels_dir = True
    
    # Walk through the directory and check if required files exist
    for path, _, files in os.walk(test_workdir):
        for file in files:
            filepath = os.path.join(path, file)
            if filepath == os.path.join(path, "classfile.json"):
                is_class_file = True
            if filepath == os.path.join(labels_dir_path, "test_data", "images", original_image):
                is_copied_image = True
            if file == original_image[:original_image.find('.jpg')]+'.png':
                is_mask = True

    # Check if all the required files exist
    bools = (is_images_dir, is_labels_dir, is_class_file, is_copied_image, is_mask)
    assert all(bools), "WARNING: Please execute `test_integration_prep_files` first to patch data"


def test_training(setup_train_env):
    """Test training of a dummy model.

    Asserts that a model is trained and serialized in the expected directory structure:

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

    Args:
        setup_train_env: A fixture that sets up a training environment.

    Raises:
        AssertionError: If the model is not trained and serialized in the expected directory structure.
    
    """
    # Instantiate a Trainer object with given parameters
    trainer = model.Trainer(
        labels= set_labels,
        input_dir= "test_data",
        train_ratio= 0.5,
        val_ratio= 0.5,
        preselected_device= -1,
        working_dir= patch.RESULT_DIR
    )
    
    # Check that all necessary data files are present
    csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
    assert all(
        any(f+'.csv' in csv for csv in csv_files)
        for f in ('data', 'train', 'val')
    ), "Missing data files among 'data.csv', 'train.csv', 'val.csv'"
    
    # Train the model
    trainer.train()
    
    # Check that model files have been serialized
    model_dir = os.path.join(labels_dir_path, 'model')
    model_files = [name for name in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, name))]
    assert any("model_checkpoint" in m for m in model_files) and len(model_files), \
    f"No model has been serialized or serialized files. {model_dir} does not contain checkpoint"
    
    # Check that tensorboard logs have been created
    tensor_logs = []
    log_dir = os.path.join(labels_dir_path, "tensorboard", "log")
    if os.path.isdir(log_dir):
        tensor_logs.extend(os.listdir(log_dir))
    assert all("events.out" in n for n in tensor_logs) and len(tensor_logs), "Tensorboard did not work"
    
    
@pytest.fixture(scope='session')
def setup_infer_env():
    """Sets up the test directory and checks whether the required model has already been trained.
    
    Directory structure at the end:
    integration/               (test_workdir)
    └── results/               (patch.RESULT_DIR)
        └── TextLine/          (labels_dir_path)
            ├── model/         (model_dir)
            └── test_data/     (dataset_path)
                └── images/
                    ├── image.jpg
                    └── ...
    
    Raises:
        AssertionError: If images directory does not exist or contains less than two images.
        AssertionError: If no model has been serialized or serialized files are not found.
    
    """
    
    # Initialize assertions
    is_images_dir = False
    
    # Check that images directory exist and contain at least 2 images
    images_path = os.path.join(dataset_path, "images")
    if os.path.isdir(images_path) and len(os.listdir(images_path)) >= 2:
        is_images_dir = True
    
    assert is_images_dir, "WARNING: Please execute `test_integration_prep_files` first to patch data"

    # Check that model files have been serialized
    model_dir = os.path.join(labels_dir_path, 'model')
    model_files = [name for name in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, name))]
    assert any("model_checkpoint" in m for m in model_files) and len(model_files), \
    f"No model has been serialized or serialized files. {model_dir} does not contain checkpoint"
    

def test_inference(setup_infer_env):
    """Run inference using a dummy predictor.

    The expected structure of the file system after running the test:
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
    
    Raises:
    AssertionError: If the output directory of the predictor or inference doesn't work

    Args:
    setup_infer_env: An instance of the test environment to run the inference

    """
    
    # Make dummy predictions
    predictor = model.Predictor(
        set_labels,
        input_dir= os.path.join("test_data", "images"),
        preselected_device= -1,
        working_dir= patch.RESULT_DIR
    )
    results = predictor.start(verbose= False)
    
    # Test assertions
    preds_dir = os.path.join(labels_dir_path, "predictions")
    assert os.path.isdir(preds_dir), "Output of the predictor is absent"
    assert len(results) == len(os.listdir(os.path.join(dataset_path, "images"))), "Inference did not work"
