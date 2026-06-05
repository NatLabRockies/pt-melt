Welcome to PT-MELT's documentation!
===================================

``PT-MELT`` (PyTorch Machine Learning Toolbox) is a collection of architectures,
processing, and utilities that are transferable over a range of ML applications.

``PT-MELT`` is a toolbox for researchers to use for machine learning applications in the
PyTorch language. The goal of this software is to enable fast start-up of machine
learning tasks and to provide a reliable and flexible framework for development and
deployment. The toolbox contains generalized methods for every aspect of the machine
learning workflow while simultaneously providing routines that can be tailored to
specific application spaces. 

``PT-MELT`` is developed alongside the ``TF-MELT`` toolbox
(https://github.com/NREL/tf-melt), with a similar design philosophy and structure.
Because of this, the two toolboxes share many of the same features, capabilities,
documentation, and examples. Eventually, the two toolboxes might be merged into a single
toolbox or leverage a separate, shared library.

The toolbox is structured with the following modules further described in the
``PT-MELT`` :ref:`Package <ptmelt-package>` section:

- ``PT-MELT`` :ref:`Models Module <ptmelt.models>` - Contains a collection of pre-built
  models that can be used for a variety of machine learning tasks.

  The models currently available are:

   - `Artificial Neural Network (ANN)` - A simple feedforward neural network with
     customizable layers and activation functions.
   - `Residual Neural Network (ResNet)` - A neural network architecture with
     customizable residual blocks.

- ``PT-MELT`` :ref:`Blocks Module <ptmelt.blocks>` - Contains a collection of pre-built
  blocks that can be used to build custom models. These blocks are designed to be easily
  imported and used in custom models. Refer to the :ref:`Models <ptmelt.models>` module
  for examples of how to use these effectively.

   The blocks currently available are:

   - `DenseBlock` - A dense block for fully-connected models.
   - `ResidualBlock` - A residual block with skip connections.
   - `DefualtOutput` - A single dense layer for output.
   - `MixtureDensityOutput` - A dense layer with mixture model output for multiple
     means and variances with learnable mixture coefficients.

- ``PT-MELT`` :ref:`Losses Module <ptmelt.losses>` - Contains a collection of pre-built
  loss functions that can be used for a variety of machine learning tasks. These loss
  functions are designed to be easily imported and used in custom models. Refer to the
  :ref:`Models <ptmelt.models>` module for examples of how to use these effectively.

   The loss functions currently available are:

   - `MixtureDensityLoss` - A negative log likelihood loss function for single and 
     multiple mixture models.

The toolbox also includes a :ref:`Utilities Subpackage <ptmelt.utils>`, which contains a
collection of functions useful for data preprocessing, model evaluation, visualization,
and other tasks.

The utility modules currently available are:

- :ref:`Evaluation Module <ptmelt.utils.evaluation>` - Contains a collection of
  functions for evaluating machine learning models. Useful for evaluating ``PT-MELT``
  model performance and extracting uncertainty quantification metrics.

- :ref:`Preprocessing Module <ptmelt.utils.preprocessing>` - Contains a collection of
  functions for preprocessing data for machine learning tasks. Leverages
  ``Scikit-learn`` preprocessing functions and implements additional helper functions.

- :ref:`Statistics Module <ptmelt.utils.statistics>` - Contains a collection of
  functions for calculating statistics and metrics for machine learning tasks. Designed
  to be utilized by the other utility functions.

- :ref:`Visualization Module <ptmelt.utils.visualization>` - Contains a collection of
  functions for visualizing data and model performance. Designed to easily generate
  plots of model performance, but can also be customized for user preferences.


Also included in the ``PT-MELT`` repo is an :ref:`Examples <examples>` directory, which
contains a set of jupyter notebooks that demonstrate how to use the different modules in
the toolbox for the full machine learning workflow (e.g., data preprocessing, model
creation, training, evaluation, and visualization).


Finally, these docs are contained in the **Docs** directory, which can be built using
Sphinx.

Contact
=======

If you have any questions, issues, or feedback regarding ``PT-MELT``, please feel free
to contact the authors:

- Email: [nwimer@nrel.gov]
- GitHub: [https://github.com/NREL/pt-melt]

We look forward to hearing from you!


.. toctree::
   :maxdepth: 2
   :caption: Contents

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

