# Copyright 2023 evdcush
"""
Registry
========
The registry in MMEngine supports hierarchical registration, which enables
cross-project calls, meaning that modules from one project can be used in
another project. Though there are other ways to implement this, the registry
provides a much easier solution.

To easily make cross-library calls, MMEngine provides twenty two root
registries, including:

:``RUNNERS``:
    The registry for Runner.

:``RUNNER_CONSTRUCTORS``:
    The constructors for Runner.

:``LOOPS``:
    Manages training, validation and testing processes, such as
    EpochBasedTrainLoop.

:``HOOKS``: the hooks, such as CheckpointHook, and ParamSchedulerHook.

:``DATASETS``: the datasets.

:``DATA_SAMPLERS``:
    Sampler of DataLoader, used to sample the data.

:``TRANSFORMS``:
    Various data preprocessing methods, such as Resize, and Reshape.

:``MODELS``:
    Various modules of the model.

:``MODEL_WRAPPERS``:
    Model wrappers for parallelizing distributed data, such as
    MMDistributedDataParallel.

:``WEIGHT_INITIALIZERS``:
    The tools for weight initialization.

:``OPTIMIZERS``:
    Registers all Optimizers and custom Optimizers in PyTorch.

:``OPTIM_WRAPPER``:
    The wrapper for Optimizer-related operations such as
    OptimWrapper, and AmpOptimWrapper.

:``OPTIM_WRAPPER_CONSTRUCTORS``:
    The constructors for optimizer wrappers.

:``PARAM_SCHEDULERS``:
    Various parameter schedulers, such as MultiStepLR.

:``METRICS``:
    The evaluation metrics for computing model accuracy, such as Accuracy.

:``EVALUATOR``:
    One or more evaluation metrics used to calculate the model accuracy.

:``TASK_UTILS``:
    The task-intensive components, such as AnchorGenerator, and BboxCoder.

:``VISUALIZERS``:
    The management drawing module that draws prediction boxes on
    images, such as DetVisualizer.

:``VISBACKENDS``:
    The backend for storing training logs, such as LocalVisBackend,
    and TensorboardVisBackend.

:``LOG_PROCESSORS``:
    Controls the log statistics window and statistics methods, by
    default we use LogProcessor.
    You may customize LogProcessor if you have special needs.

:``FUNCTIONS``:
    Registers various functions, such as collate_fn in DataLoader.

:``INFERENCERS``:
    Registers inferencers of different tasks, such as DetInferencer,
    which is used to perform inference on the detection task.

"""
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

# If MMCV is installed, cnn block and transforms will be registered.
try:
    import mmcv  # noqa: F401
except:  # noqa: E722
    ...

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=MMENGINE_RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=MMENGINE_RUNNER_CONSTRUCTORS)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=MMENGINE_LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=MMENGINE_HOOKS,
    locations=['armitage.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    locations=['armitage.datasets'])

TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['armitage.datasets.transform'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, locations=['armitage.models'])

# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['armitage.models'])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['armitage.models'])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['armitage.engine.optimizer'])

# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['armitage.engine.optim_wrapper'])

# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['armitage.engine.optim_wrapper_constructor'])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['armitage.engine.scheduler'])
# manage all kinds of metrics
METRICS = Registry(
    'metric',
    parent=MMENGINE_METRICS,
    locations=['armitage.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator',
    parent=MMENGINE_EVALUATOR,
    locations=['armitage.evaluation'])

# NOTE: armitage does not define less commomly customized
# modules below, therefore locations are not specified for Registry.

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util', parent=MMENGINE_TASK_UTILS)

# manage visualizer
VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=MMENGINE_VISBACKENDS)

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor', parent=MMENGINE_LOG_PROCESSORS)
