from typing import Any, List, Optional

from core.Constants import *


class PlaybookStep:
    """Defines a step in the playbook"""

    def __init__(self, module: str, params: Optional[List[Any]], wait: Optional[int]):
        self.module = module
        self.params = params if params is not None else {}
        self.wait = wait
