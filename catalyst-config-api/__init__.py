from catalyst.registry import Registry

from .runner import MyConfigRunner
from .model import MyModel


Registry(MyConfigRunner)
Registry(MyModel)
