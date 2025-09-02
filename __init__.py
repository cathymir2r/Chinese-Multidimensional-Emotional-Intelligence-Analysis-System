# sentiment_analyzer/__init__.py
__version__ = '1.0.0'
name = "sentiment_analyzer"

from .data_processor import *
from .feature_extractor import *
from .models import *
from .trainer import *
from .evaluator import *
from .predictor import *
from .utils import *
from .web_interface import *
