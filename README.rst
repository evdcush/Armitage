========
Armitage
========
Respite.

Built from:
https://github.com/open-mmlab/mmengine-template

Directory Structure
===================

::

    ├── configs                                 Commonly used base config file.
    ├── demo
    │   ├── mmengine_template_demo.py           General demo script
    ├── mmengine_template
    │   ├── datasets
    │   │   ├── __init__.py
    │   │   ├── datasets.py                     Customize your dataset here
    │   │   └── transforms.py                   Customize your data transform here
    │   ├── engine
    │   │   ├── __init__.py
    │   │   ├── hooks.py                        Customize your hooks here
    │   │   ├── optimizers.py                   Less commonly used. Customize your optimizer here
    │   │   ├── optim_wrappers.py               Less commonly used. Customize your optimizer wrapper here
    │   │   ├── optim_wrapper_constructors.py   Less commonly used. Customize your optimizer wrapper constructor here
    │   │   └── schedulers.py                   Customize your lr/momentum scheduler here
    │   ├── evaluation
    │   │   ├── __init__.py
    │   │   ├── evaluator.py                    Less commonly used. Customize your evaluator here
    │   │   └── metrics.py                      Customize your metric here.
    │   ├── infer
    │   │   ├── inference.py                    Used for demo script. Customize your inferencer here
    │   │   └── __init__.py
    │   ├── __init__.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── model.py                        Customize your model here.
    │   │   ├── weight_init.py                  Less commonly used here. Customize your initializer here.
    │   │   └── wrappers.py                     Less commonly used here. Customize your wrapper here.
    │   ├── registry.py
    │   └── version.py
    └── tools                                   General train/test script


**License**
    Except where noted otherwise, this project is licensed under the |SPDX-License-Name|_.

.. Substitutions:


.. PROJECT FILES:

.. LOCAL FILES:
.. _SPDX-License-Name: LICENSE
.. |SPDX-License-Name| replace:: Apache-2.0
