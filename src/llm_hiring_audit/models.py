"""Core data models for the audit.

Every CV is the combination of a *base profile* (the qualifications, held
constant) and a *treatment* (the demographic signals we manipulate).
Keeping the two separate is what makes the design a controlled experiment.
"""

import unicodedata
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Gender(StrEnum):
    FEMALE = "female"
    MALE = "male"


class Origin(StrEnum):
    """Perceived origin signalled by the name (not the candidate's identity)."""

    MAJORITY = "majority"
    MAGHREBI = "maghrebi"
    SUBSAHARAN = "subsaharan"


class AddressArea(StrEnum):
    """Residential signal, a known channel of discrimination in France."""

    CENTRAL = "central"
    DEPRIVED = "deprived"  # neighbourhood classified as QPV


class Treatment(BaseModel):
    """One cell of the factorial design."""

    model_config = ConfigDict(frozen=True)

    gender: Gender
    origin: Origin
    address_area: AddressArea


class Experience(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    employer: str
    start_year: int
    end_year: int
    description: str


class Education(BaseModel):
    model_config = ConfigDict(frozen=True)

    degree: str
    school: str
    year: int


class JobPosting(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    title: str
    description: str


class BaseProfile(BaseModel):
    """Everything on the CV that is held constant across treatments."""

    model_config = ConfigDict(frozen=True)

    profile_id: str
    job_id: str
    headline: str
    experiences: tuple[Experience, ...]
    education: tuple[Education, ...]
    skills: tuple[str, ...]
    languages: tuple[str, ...]


class Identity(BaseModel):
    """The manipulated fields: name and address."""

    model_config = ConfigDict(frozen=True)

    first_name: str
    last_name: str
    street: str
    postcode: str
    city: str

    @property
    def email(self) -> str:
        return f"{_ascii_slug(self.first_name)}.{_ascii_slug(self.last_name)}@email.fr"


class CV(BaseModel):
    model_config = ConfigDict(frozen=True)

    cv_id: str
    profile: BaseProfile
    treatment: Treatment
    identity: Identity
    name_index: int = Field(ge=0, description="Which name of the cell was drawn")


def _ascii_slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if c.isascii() and c.isalnum()).lower()
