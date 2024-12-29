import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
  """
  Sets random seeds before each test.
  autouse=True ensures this fixture runs automatically for each test.
  """
  seed = 123
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)

  # If you're using CUDA: Force deterministic behavior
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
