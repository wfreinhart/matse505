from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

class Context(dict):
    """
    Universal Namespace for stateless modules.
    Standard keys:
    - 'data': pd.DataFrame
    - 'tensors': Dict[str, Any] (e.g., X_train, y_test)
    - 'model': Any (trained estimator)
    - 'viz': Any (figure object)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'data' not in self: self['data'] = None
        if 'tensors' not in self: self['tensors'] = {}
        if 'model' not in self: self['model'] = None
        if 'viz' not in self: self['viz'] = None

@dataclass
class DatasetArtifact:
    id: str
    domain: str
    load_logic: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConceptModule:
    id: str
    type: str  # Foundational | Comparative | Refinement | Generative
    requirements: List[str]
    markdown_content: str
    code_block: str
    visual_asset: Optional[str] = None
