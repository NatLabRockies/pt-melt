.. _ptmelt-package:

PT-MELT package
===============

``PT-MELT`` is composed of a series of modules that form the basis for building a 
variety of machine learning models. The structure of the packages is to encourage 
modularity and reusability of the code. The main modules are:

- :ref:`Blocks Module <ptmelt.blocks>`: Each of the blocks is a self-contained
  PyTorch model in itself, but can be combined with other blocks to form more complex
  models. The blocks are designed to be easily combined with other blocks, and to be
  easily extended.

- :ref:`Layers Module <ptmelt.layers>`: The layers are redefined PyTorch layers that
  are used in some of the blocks. The layers are designed to be plug-in replacements for
  some of the default PyTorch layers, but with additional functionality.

- :ref:`Losses Module <ptmelt.losses>`: The losses are custom loss functions that should
  be used with their respective models. Certain loss functions are designed to be used
  with specific models, but others can be used with any model.

- :ref:`Models Module <ptmelt.models>`: The models are the main machine learning models
  that are built using the blocks and losses. The models serve a dual purpose of being a
  standalone model, and also as a template for building more complex models with the
  ``PT-MELT`` blocks.

- :ref:`NN Utils Module <ptmelt.nn_utils>`: The nn_utils module contains utility
  functions that are used in the main modules. These functions are used for retrieving
  activation functions and initilizer functions via their string names.

Following is a detailed description of the modules and subpackages in the ``PT-MELT``.


.. _ptmelt.blocks:

Blocks Module
-------------

.. automodule:: ptmelt.blocks
   :members:
   :undoc-members:
   :show-inheritance:

.. _ptmelt.layers:

Layers Module
-------------

.. automodule:: ptmelt.layers
   :members:
   :undoc-members:
   :show-inheritance:


.. _ptmelt.losses:

Losses Module
-------------

.. automodule:: ptmelt.losses
   :members:
   :undoc-members:
   :show-inheritance:

.. _ptmelt.models:

Models Module
-------------

.. automodule:: ptmelt.models
   :members:
   :undoc-members:
   :show-inheritance:

.. _ptmelt.nn_utils:

NN Utils Module
---------------

.. automodule:: ptmelt.nn_utils
   :members:
   :undoc-members:
   :show-inheritance:


.. _ptmelt.subpackages:

Subpackages
-----------

In addition to the main modules, there are subpackages that contain various utility
functions that are used in the main modules. The subpackages are:

- :ref:`PT-MELT Utilities <ptmelt.utils>` : Contains utility functions that are used in 
   the main modules. These functions contain routines for data processing, model 
   evaluation, visualization, and other general-purpose functions.



.. toctree::
   :maxdepth: 1

   ptmelt.utils


