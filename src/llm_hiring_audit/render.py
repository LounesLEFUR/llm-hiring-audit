"""Rendering of a CV as the plain French text an LLM screener would read."""

from typing import Any

from llm_hiring_audit.models import CV


def cv_to_record(cv: CV) -> dict[str, Any]:
    """Flat record written to the CV file: design variables plus the text."""
    return {
        "cv_id": cv.cv_id,
        "profile_id": cv.profile.profile_id,
        "job_id": cv.profile.job_id,
        "gender": cv.treatment.gender,
        "origin": cv.treatment.origin,
        "address_area": cv.treatment.address_area,
        "name_index": cv.name_index,
        "first_name": cv.identity.first_name,
        "last_name": cv.identity.last_name,
        "city": cv.identity.city,
        "text": render_cv(cv),
    }


def render_cv(cv: CV) -> str:
    p, who = cv.profile, cv.identity
    lines = [
        f"{who.first_name} {who.last_name}",
        f"{who.street}, {who.postcode} {who.city}",
        who.email,
        "",
        p.headline.upper(),
        "",
        "EXPÉRIENCE PROFESSIONNELLE",
    ]
    for exp in p.experiences:
        lines.append(f"- {exp.start_year}-{exp.end_year} : {exp.title}, {exp.employer}")
        lines.append(f"  {exp.description}")
    lines += ["", "FORMATION"]
    for edu in p.education:
        lines.append(f"- {edu.year} : {edu.degree}, {edu.school}")
    lines += [
        "",
        "COMPÉTENCES",
        ", ".join(p.skills),
        "",
        "LANGUES",
        ", ".join(p.languages),
    ]
    return "\n".join(lines)
