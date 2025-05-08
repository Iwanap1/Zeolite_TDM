from pydantic import BaseModel, Field, create_model
from typing import Union, Tuple
import re

class CommercialSource(BaseModel):
    source: str = Field(default="purchased zeolite")
    did_step: Union[bool, str] = Field(alias="did step")
    name: str = ""
    supplier: str = ""
    form: str = ""
    cbv: str = ""
    Si_Al: str = Field(default="", alias="Si/Al")
    SiO2_Al2O3: str = Field(default="", alias="SiO2/Al2O3")
    M: str = ""
    Si_M: str = Field(default="", alias="Si/M")

class CalcinationProcess(BaseModel):
    process: str = Field(default="calcination")
    did_step: Union[bool, str] = Field(alias="did step")
    gas: str = ""
    gas_flow: str = ""
    heat_rate: str = ""
    temperature: str = ""
    time: str = ""

class IonExchangeProcess(BaseModel):
    process: str = ""
    did_step: Union[bool, str] = Field(alias="did step")
    initial_form: str = ""
    final_form: str = ""
    solutes: str = ""
    temperature: str = ""
    time: str = ""
    solvent: str = ""
    repeats: str = ""

class HydrothermalCrystallizationProcess(BaseModel):
    source: str = Field(default="hydrothermal crystallization")
    did_step: Union[bool, str] = Field(alias="did step")
    components: str = ""
    gel_composition: str = ""
    seed: str = ""
    ratios: str = ""
    pH: str = ""
    pH_adjusted_with: str = Field(alias="pH adjusted with")
    temperature: str = ""
    time: str = ""
    tumbling: str = ""

class SolventEtchingProcess(BaseModel):
    process: str = ""
    did_step: Union[bool, str] = Field(alias="did step")
    mass_parent: str = Field(alias="mass parent")
    solutes: str = ""
    solvent: str = ""
    temperature: str = ""
    time: str = ""
    microwave: str = ""
    sonication: str = ""
    repeats: str = ""
    pressure: str = ""

class SteamTreatmentProcess(BaseModel):
    process: str = ""
    did_step: Union[bool, str] = Field(alias="did step")
    mass_parent: str = Field(alias="mass parent")
    gas: str = ""
    temperature: str = ""
    time: str = ""
    pressure: str = ""
    WHSV: str = ""
    additives: str = ""

class ChemicalLiquidDepositionProcess(BaseModel):
    process: str = ""
    did_step: Union[bool, str] = Field(alias="did step")
    mass_parent: str = ""
    solutes: str = ""
    solvent: str = ""
    temperature: str = ""
    time: str = ""
    microwave: str = ""
    ultrasound: str = ""
    repeats: str = ""
    repeated_with_calcination: str = ""

class RecrystallizationProcess(BaseModel):
    process: str = "recrystallization",
    did_step: Union[bool, str] = Field(alias="did step")
    composite_name: str = "",
    dissolvent: str = "",
    glycerol: str = "",
    hydrothermal_1_temperature: str = "",
    hydrothermal_1_time: str = "",
    pH_adjusted_with: str = "",
    pH_adjustment: str = "",
    hydrothermal_2_temperature: str = "",
    hydrothermal_2_time: str = "",
    mass_parent: str = "",
    surfactant: str = ""


def get_process_class(process_name: str):
    base_map = {
        "commercial": CommercialSource,
        "calcination": CalcinationProcess,
        "alkaline treatment": SolventEtchingProcess,
        "acid treatment": SolventEtchingProcess,
        "hydrothermal crystallization": HydrothermalCrystallizationProcess,
        "chemical liquid deposition": ChemicalLiquidDepositionProcess,
        "purchased zeolite": CommercialSource,
        "steam treatment": SteamTreatmentProcess
    }

    if re.match(r"^-->\s*\w+", process_name):
        return IonExchangeProcess

    return base_map.get(process_name, None)



process_templates = {
    "commercial": {
        "source": "commercial",
        "did step": "true",
        "name": "",
        "supplier": "",
        "form": "",
        "cbv": "",
        "Si/Al": "",
        "SiO2/Al2O3": "",
        "M": "",
        "Si/M": ""
    },
    "calcination": {
        "process": "calcination",
        "did step": "true",
        "gas": "",
        "gas flow": "",
        "heat rate": "",
        "temperature": "",
        "time": ""
    },
    "ion_exchange": {
        "process": "ion_exchange",
        "did step": "true",
        "initial form": "",
        "final form": "",
        "solutes": "",
        "temperature": "",
        "time": "",
        "solvent": "",
        "repeats": ""
    },
    "hydrothermal crystallization": {
        "source": "hydrothermal crystallization",
        "did_step": "true",
        "components": "",
        "gel_composition": "",
        "seed": "",
        "ratios": "",
        "pH": "",
        "pH_adjusted_with": "",
        "temperature": "",
        "time": "",
        "tumbling": ""
    },
    "solvent_etching": {
        "process": "solvent etching",
        "did step": "true",
        "mass parent": "",
        "solutes": "",
        "solvent": "",
        "temperature": "",
        "time": "",
        "microwave": "",
        "sonication": "",
        "repeats": ""
    },
    "steam treatment": {
        "process": "",
        "did step": "true",
        "mass parent": "",
        "gas": "",
        "temperature": "",
        "time": "",
        "pressure": "",
        "WHSV": "",
        "additives": ""
    },
    "chemical liquid deposition": {
        "process": "chemical liquid deposition",
        "did_step": "true",
        "mass_parent": "",
        "solutes": "",
        "solvent": "",
        "temperature": "",
        "time": "",
        "microwave": "",
        "ultrasound": "",
        "repeats": "",
        "repeated_with_calcination": ""
    },
    "recrystallization": {
        "process": "recrystallization",
        "did step": "true",
        "composite": "",
        "dissolvent": "",
        "glycerol": "",
        "hydrothermal 1 temp": "",
        "hydrothermal 1 time": "",
        "pH adjusted with": "",
        "pH adjustment": "",
        "hydrothermal 2 temp": "",
        "hydrothermal 2 time": "",
        "parent mass": "",
        "surfactant": ""
    }
}