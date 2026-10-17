"""Run configuration and the three presets that trace the privacy/utility curve."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .spans import DIRECT_CATS

BALANCED_EXTRA = (
    "YEAR",
    "LOCATION",
    "FACILITY",
    "ORGANIZATION",
    "INSURER",
    "SPORTS_TEAM",
    "OCCUPATION",
    "HOLIDAY",
    "ACTIVITY",
)
AGGRESSIVE_EXTRA = ("OTHER_QI",)


@dataclass(frozen=True)
class Config:
    preset: str = "balanced"
    use_codes: bool = True  # False = "without ICD-10 codes at inference" bonus mode
    detectors: tuple[str, ...] = ("rules",)  # + "deid" (obi/deid_roberta_i2b2), "gliner", "llm" (Qwen via Ollama)
    remove: frozenset[str] = field(default_factory=lambda: frozenset(DIRECT_CATS + BALANCED_EXTRA))
    weak_name_threshold: float = 0.6  # minimum confidence for an uncorroborated person-name detection
    age_mode: str = "decade"  # decade | coarse
    explain_text: bool = False  # include original surface text in the decisions sidecar (debug only)
    repair_rounds: int = 2
    injury_context_shield: bool = True  # keep activities told as an injury mechanism (also with codes)
    llm_model: str = "qwen3.5:2b-q4_K_M"
    llm_num_ctx: int = 4096
    llm_log: str | None = None  # JSONL file with one line per LLM call (no text), for monitoring a run
    features: frozenset[str] = frozenset()  # M5 feature flags e.g. {"F0", "F2"}
    device: str = "auto"  # transformer detectors: GPU only while the total VRAM estimate stays < 4 GB


PRESETS: dict[str, Config] = {
    "conservative": Config(
        preset="conservative",
        remove=frozenset(DIRECT_CATS + ("YEAR",)),
        weak_name_threshold=0.7,
    ),
    "balanced": Config(),
    "aggressive": Config(
        preset="aggressive",
        remove=frozenset(DIRECT_CATS + BALANCED_EXTRA + AGGRESSIVE_EXTRA),
        weak_name_threshold=0.55,
        age_mode="coarse",
        injury_context_shield=False,  # with codes, only code evidence protects an activity
    ),
    # Balanced settings (name threshold, decade ages) plus low-risk details, with the code-evidence guard
    # (F0) that keeps traps intact. The agreement rule (F4, D7) was measured and not kept by default: on
    # tune it lowered low-risk removal (0.517 -> 0.381) with no trap or concept gain; it stays a flag.
    "privacy": Config(
        preset="privacy",
        remove=frozenset(DIRECT_CATS + BALANCED_EXTRA + AGGRESSIVE_EXTRA),
        features=frozenset({"F0", "F2"}),  # F2 only changes the LLM detector (modes llm and full)
    ),
}


def make_config(preset: str = "balanced", **overrides) -> Config:
    base = PRESETS[preset]
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if "features" in overrides:  # flags add to the preset's own features
        overrides["features"] = base.features | frozenset(overrides["features"])
    return replace(base, **overrides)
