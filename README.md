# LayerFusion
Computer Vision Project on Self-Supervised Learning

# architecture
    core: 
    -> config.py
            -> AttrDict : subclass dict which will be used to save all configurations of model and data
            -> merge_dicts : replace default configurations by user defined one
            -> cfg_from_file : load configurations from specific file
            -> print_cfg : print configurations
            -> cfg_from_list : load configurations from a list object
            -> _decode_cfg_value : perform some transformations on dict value from various object to desired object
            -> _check_and_coerce_cfg_value_type : check the type of dict value


    data:
    -> datasets
            -> disk_dataset
                        -> DiskImageDataset(torch.utils.data.Dataset) (Not be used) : load data from disk
            -> ssl_dataset
                        -> GenericSSLDataset(torch.utils.data.Dataset) : load ssl dataset from source (contains different transformation)
    -> ssl_transform
            -> basic_transforms_wrapper.py
                        -> TORCHVISION_TRANSFORMS : dict contains several transformation methods


# TODO:
    - 11.19 ~ 11.21 : add more pretext task transformations, rotate already implemented in sslime.
      TODO: Exemplar, Rel. Patch Loc., Jigsaw (Revisiting Self-Supervised Visual Representation Learning, Alexander Kolesnikov etc.)
    - 12.2 ~ 12.9 : finish model part(VGG_A: derived from sslime) and generate 2 models: rotation and jigsaw, then perform visualization on each layer to see the results
