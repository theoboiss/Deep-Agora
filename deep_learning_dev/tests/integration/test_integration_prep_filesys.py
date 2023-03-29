import os, sys, glob, logging, pytest, shutil


# Neutralise the loggers of the data.preparation package
logging.getLogger('deep_learning_lab.data_preparation').disabled = True

    
from deep_learning_lab.data_preparation import orchestration, patch


test_workdir = "tests/integration/"

orchestration.RAW_DATA_DIR = test_workdir
patch.RESULT_DIR = os.path.join(test_workdir, "results")

set_labels = ['TextLine']

result_dir = '_'.join(set_labels)
result_dir_path = os.path.join(patch.RESULT_DIR, '_'.join(set_labels), "test_data")
raw_data_dir = os.path.join(test_workdir,"raw_page_dataset")


@pytest.fixture(scope='session', autouse= True)
def setup_session():
    """Sets up the test directory by testing if the required data has already been manually downloaded.
    
    Expected structure of the file system:
    integration/            (=test_workdir/)
    └── raw_page_dataset/   (=raw_data_dir)
        ├── "image".jpg
        ├── ...
        ├── "image".xml
        └── ...
    
    """
    raw_data = os.listdir(raw_data_dir)
    assert "M_Aigen_am_Inn_002-01_0000.jpg" in raw_data and "M_Aigen_am_Inn_002-01_0000.xml" in raw_data, "Run `download_test_data.sh`"
    
    for root, dirs, files in os.walk(patch.RESULT_DIR, topdown= False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)
    
    # All the code above has been executed at the beginning of the tests
    
    yield
    
    # All the code bellow will be executed at the end of the tests


def test_orch_filesystem():
    """Patches a minimal dataset and tests whether the file system has been correctly altered.
    
    Expected structure of the file system:
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
    
    """
    
    # Patches
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
    
    # Initializes assertions
    is_result_dir = is_images_dir =  is_labels_dir = is_class_file = is_copied_image = is_mask = False
    original_image = os.listdir(raw_data_dir)[0]
    
    # Tests assertions
    if os.path.isdir(result_dir_path):
        is_result_dir = True

    if os.path.isdir(os.path.join(result_dir_path, "images")):
        is_images_dir = True
    if os.path.isdir(os.path.join(result_dir_path, "labels")):
        is_labels_dir = True
    
    for path, currentDirectory, files in os.walk(test_workdir):
        for file in files:
            filepath = os.path.join(path, file)
            if filepath == os.path.join(path, "classfile.json"):
                is_class_file = True
            if filepath == os.path.join(path, original_image):
                is_copied_image = True
            if file == original_image[:original_image.find('.jpg')]+'.png':
                is_mask = True
    bools = (is_result_dir, is_images_dir, is_labels_dir, is_class_file, is_copied_image, is_mask)
    assert all(bools), "File system has not correctly been altered"
