import os, shutil, pytest

from deep_learning_lab.data_preparation import orchestration, patch


test_workdir = "tests/integration/" # path to the directory where integration tests will be run

orchestration.RAW_DATA_DIR = test_workdir # path to the directory containing the raw data used for testing
patch.RESULT_DIR = os.path.join(test_workdir, "results") # path to the directory where the results of the testing will be stored

set_labels = ['TextLine'] # list of strings containing the labels that will be used in the testing

result_dir = '_'.join(set_labels) # name of the directory where the test results will be stored
result_dir_path = os.path.join(patch.RESULT_DIR, '_'.join(set_labels), "test_data") # path to the directory where the test results will be stored
raw_data_dir = os.path.join(test_workdir,"raw_page_dataset") # path to the directory containing the raw data used for testing


@pytest.fixture(scope='session', autouse= True)
def setup_session():
    """Set up the integration test directory by checking if the required data has been downloaded manually.
    
    The expected file system structure is as follows:
    
    integration/            (=test_workdir/)
    └── raw_page_dataset/   (=raw_data_dir)
        ├── "image".jpg
        ├── ...
        ├── "image".xml
        └── ...
        
    Raises:
        AssertionError: If the required data is not present. Run `download_test_data.sh`.
    
    This fixture is responsible for cleaning up the result directory after the tests have run.

    """
    
    # Check if the required data files exist
    raw_data = os.listdir(raw_data_dir)
    assert "M_Aigen_am_Inn_002-01_0000.jpg" in raw_data and "M_Aigen_am_Inn_002-01_0000.xml" in raw_data, "Run `download_test_data.sh`"
    
    # Clean the result directory
    for root, dirs, _ in os.walk(patch.RESULT_DIR, topdown= False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)
    
    # All the code above has been executed at the beginning of the tests
    
    yield
    
    # All the code bellow will be executed at the end of the tests
    
    pass


def test_orch_filesystem():
    """Test whether the file system has been correctly altered by patching a minimal dataset.

    The expected file system structure is as follows:
    integration/            (=test_workdir/)
    ├── raw_page_dataset/   (=raw_data_dir)
    |   ├── "image".jpg
    |   ├── ...
    |   ├── "image".xml
    |   └── ...
    └── results/            (=patch.RESULT_DIR)
        └── TextLine/       (=result_dir_path/)
            ├── images/
            │   └── "image".jpg
            ├── labels/
            │   └── "image".png
            └── classfile.json

    Raises:
        AssertionError: If the file system has not been correctly altered.
    
    """
    
    # Patch the dataset
    orch = orchestration.Orchestrator(output_structure= {'dir_data': "test_data",
                                           'dir_images': "images",
                                           'dir_labels': "labels"})
    orch.ingestDatasets(datasets= [{'dir_data': raw_data_dir,
                                    'dir_images': "",
                                    'dir_annotations': ""}],
                        add_defaults= False)
    orch.ingestLabels(uniform_set_labels= set_labels, prompt= False)
    orch.validate(auto_yes= True, verbose= False)
    orch.preprocess()
    
    # Initialize assertions
    is_result_dir = is_images_dir =  is_labels_dir = is_class_file = is_copied_image = is_mask = False
    original_images = os.listdir(raw_data_dir)
    
    # Check that the file system has been correctly altered
    if os.path.isdir(result_dir_path):
        is_result_dir = True

    if os.path.isdir(os.path.join(result_dir_path, "images")):
        is_images_dir = True
    if os.path.isdir(os.path.join(result_dir_path, "labels")):
        is_labels_dir = True
    
    for path, _, files in os.walk(test_workdir):
        for file in files:
            filepath = os.path.join(path, file)
            if filepath == os.path.join(path, "classfile.json"):
                is_class_file = True
            if filepath in map(lambda image: os.path.join(path, image), original_images):
                is_copied_image = True
            if file in map(lambda image: image[:image.find('.jpg')]+'.png', original_images):
                is_mask = True

    # Assert that all checks pass
    bools = (is_result_dir, is_images_dir, is_labels_dir, is_class_file, is_copied_image, is_mask)
    assert all(bools), "File system has not correctly been altered"
