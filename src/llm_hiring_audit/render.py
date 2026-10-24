"""Rendering of a CV as the plain French text an LLM screener would read."""

from llm_hiring_audit.models import CV


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
