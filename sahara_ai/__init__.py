"""SAHARA's Meta AI layer: mission commander, open-vocabulary perception, triage.

Pure Python (standard library only) so the ROS 2 nodes can import it without
touching the pinned simulation environment. Talks to the Meta Model API
(https://api.meta.ai/v1); with no MODEL_API_KEY set it runs in mock mode.
"""
from .config import Config
from .client import MetaClient
from .sam import SamRecord, parse_records

__all__ = ["Config", "MetaClient", "SamRecord", "parse_records"]
