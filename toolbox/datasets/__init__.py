from toolbox.datasets.camvid import CamVid


def get_dataset(cfg):
    crop_size = (cfg['image_h'], cfg['image_w'])
    num_class = cfg['n_classes']

    if cfg['dataset'] == 'camvid':
        root = 'D:/Python/DataSet/Camvid/'
        class_dict_path = root + 'class_dict.csv'
        TRAIN_ROOT = root + 'train'
        TRAIN_LABEL = root + 'train_labels'
        VAL_ROOT = root + 'val'
        VAL_LABEL = root + 'val_labels'
        TEST_ROOT = root + 'test'
        TEST_LABEL = root + 'test_labels'
        return CamVid([TRAIN_ROOT, TRAIN_LABEL, class_dict_path], crop_size, num_class), CamVid([VAL_ROOT, VAL_LABEL, class_dict_path], crop_size, num_class), CamVid([TEST_ROOT, TEST_LABEL, class_dict_path], crop_size, num_class)
    elif cfg['dataset'] == 'cityscapes':
        root = 'D:/Python/DataSet/Cityscapes/'
        class_dict_path = root + 'class_dict.csv'
        TRAIN_ROOT = root + 'train'
        TRAIN_LABEL = root + 'train_labels'
        VAL_ROOT = root + 'val'
        VAL_LABEL = root + 'val_labels'
        TEST_ROOT = root + 'test'
        TEST_LABEL = root + 'test_labels'
        return
