from pydantic import BaseModel, create_model, RootModel
from typing import List, Literal
import re

POSTSYNTH = [
    "calcination", "recrystallization",
    "solvent_etching", "steam treatment", "ion_exchange",
    "chemical_liquid_deposition", "impregnation", "something_else"
]

SOURCE = ['commercial', 'hydrothermal crystallization', 'unknown']



def make_schema(zeolite_names: List[str]):
    class ZeoliteInfo(BaseModel):
        morphological_description: str
        zeolite_source: Literal.__getitem__(tuple(SOURCE))
        post_synthesis: List[Literal.__getitem__(tuple(POSTSYNTH))]

    fields = {
        name: (ZeoliteInfo, ...)
        for name in zeolite_names
    }

    ZeoliteStepsModel = create_model("ZeoliteStepsModel", **fields)

    class ZeoliteProcesses(RootModel[ZeoliteStepsModel]):
        pass

    return ZeoliteProcesses