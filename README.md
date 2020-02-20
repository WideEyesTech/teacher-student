# Teacher-student project with SOTA teachers to improve student models results


- [Teacher-student project with SOTA teachers to improve student models results](#teacher-student-project-with-sota-teachers-to-improve-student-models-results)
  - [**Generate weak labels**](#generate-weak-labels)
    - [*With CenterNet*](#with-centernet)
    - [*With mmdetection (ATSS, GCNET)*](#with-mmdetection-atss-gcnet)


## **Generate weak labels**

### *With CenterNet*

First of all set the proper params on the config.

Choose over which data make the inferences:

Go to:
> teachers/CenterNet/config.py
And set (at least) params:

```python
# Directories
self._configs["data_dir"] = "/home/$USER/datasets/openimages"
self._configs["filenames_dir"] = "/tmp/oi_names.txt"
self._configs["cache_dir"] = "/home/$USER/Desktop/teacher-student/teachers/CenterNet/cache"
self._configs["snapshot_name"] = "CenterNet-104_480000"
self._configs["result_dir"] = "/opt/results"
self._configs["model_config"] = "/home/$USER/Desktop/teacher-student/teachers/CenterNet/config/CenterNet104_teacher_student.json"
```

* data_dir: Folder with images
* filenames_dir: ***.txt*** with image names
* cached_dir + snapshot_name: pretrained model should be in cache/nnet/$snapshot_name
* result_dir: folder to store the results
* model_config: hyper params config file

> **NOTE** Only JSON files will be saved as results. If you want to see also labeled images change *debug=False -> True* in teachers/CenterNet/test/openimages.py
```python
def testing(db, nnet, result_dir, debug=False):
    # This way images will be saved too
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=True)
```

When all config is ready just execute ***get_teacher_inferences.py centernet {gpu number}***


### *With mmdetection (ATSS, GCNET)*

Execute 
***get_teacher_inferences.py mmdetection {gpu number}*** and follow the steps

Recommended config:

    OS: Ubuntu 18.04
    CUDA: 9.0
    NVCC: 9.0
    GCC(G++): 5.5

