# app/job_matching/knowledge/data_sources.py
"""
Lightweight data collection utilities using only built-in Python features.
"""

import re
from typing import Dict, List, Any
from datetime import datetime


class DataCollector:
    """
    Lightweight utility class for text processing and skill extraction.
    Uses only built-in Python features - no external dependencies.
    """

    def __init__(self):
        self.skill_patterns = self._build_skill_patterns()

    def _build_skill_patterns(self) -> Dict[str, List[str]]:
        """Build comprehensive skill detection patterns"""
        return {
            "programming_languages": [
                r"\bpython\b",
                r"\bjavascript\b",
                r"\btypescript\b",
                r"\bjava\b",
                r"\bc\+\+\b",
                r"\bc#\b",
                r"\bruby\b",
                r"\bgo\b",
                r"\brust\b",
                r"\bphp\b",
                r"\bswift\b",
                r"\bkotlin\b",
                r"\bscala\b",
                r"\br\b",
            ],
            "frameworks": [
                r"\breact\b",
                r"\bangular\b",
                r"\bvue\.?js\b",
                r"\bdjango\b",
                r"\bflask\b",
                r"\bspring\b",
                r"\bexpress\.?js\b",
                r"\bnode\.?js\b",
                r"\bnext\.?js\b",
                r"\bsvelte\b",
                r"\blaravel\b",
            ],
            "databases": [
                r"\bmysql\b",
                r"\bpostgresql\b",
                r"\bpostgres\b",
                r"\bmongodb\b",
                r"\bredis\b",
                r"\belasticsearch\b",
                r"\bcassandra\b",
                r"\boracle\b",
                r"\bsqlite\b",
                r"\bmariadb\b",
            ],
            "cloud_platforms": [
                r"\baws\b",
                r"\bazure\b",
                r"\bgcp\b",
                r"\bgoogle cloud\b",
                r"\bheroku\b",
                r"\bdigitalocean\b",
                r"\bvercel\b",
                r"\bnetlify\b",
            ],
            "tools": [
                r"\bdocker\b",
                r"\bkubernetes\b",
                r"\bgit\b",
                r"\bjenkins\b",
                r"\bterraform\b",
                r"\bansible\b",
                r"\bnginx\b",
                r"\bapache\b",
                r"\bwebpack\b",
                r"\bbabel\b",
                r"\bresux\b",
            ],
            "data_science": [
                r"\bmachine learning\b",
                r"\bml\b",
                r"\bai\b",
                r"\bdata science\b",
                r"\bpandas\b",
                r"\bnumpy\b",
                r"\bscikit-learn\b",
                r"\btensorflow\b",
                r"\bpytorch\b",
                r"\btableau\b",
                r"\bpower bi\b",
            ],
        }

    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract technical skills from text using regex patterns.
        Works with resumes, job descriptions, etc.
        """
        found_skills = []
        text_lower = text.lower()

        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    # Clean up matches
                    for match in matches:
                        clean_skill = match.strip().replace(".", "")
                        if len(clean_skill) > 1 and clean_skill not in found_skills:
                            found_skills.append(clean_skill)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_skills))

    def extract_experience_keywords(self, text: str) -> List[str]:
        """Extract experience-related keywords"""
        experience_patterns = [
            r"\b(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b",
            r"\bsenior\b",
            r"\bjunior\b",
            r"\blead\b",
            r"\bprincipal\b",
            r"\bmanager\b",
            r"\bdirector\b",
            r"\barchitect\b",
            r"\binternship\b",
            r"\bentry.level\b",
            r"\bmid.level\b",
        ]

        found_keywords = []
        text_lower = text.lower()

        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            found_keywords.extend(matches)

        return found_keywords

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap"""
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def extract_company_info(self, text: str) -> Dict[str, Any]:
        """Extract company-related information"""
        # Look for company size indicators
        size_indicators = {
            "startup": r"\bstartup\b|\bearly.stage\b|\bseed\b",
            "scale_up": r"\bscale.up\b|\bgrowth\b|\bseries [abc]\b",
            "enterprise": r"\benterprise\b|\bfortune 500\b|\blarge\b|\bcorporate\b",
        }

        company_info = {"size": "unknown", "indicators": []}
        text_lower = text.lower()

        for size, pattern in size_indicators.items():
            if re.search(pattern, text_lower):
                company_info["size"] = size
                company_info["indicators"].append(pattern)
                break

        return company_info


def extract_experience_level(text: str) -> str:
    """Extract experience level from job description or resume"""
    text_lower = text.lower()

    # Senior level indicators
    senior_patterns = [
        r"\bsenior\b",
        r"\blead\b",
        r"\bprincipal\b",
        r"\barchitect\b",
        r"\b(?:5|6|7|8|9|\d{2,})\+?\s*years?\b",
        r"\bexpert\b",
    ]

    # Junior level indicators
    junior_patterns = [
        r"\bjunior\b",
        r"\bentry.level\b",
        r"\bgraduate\b",
        r"\bintern\b",
        r"\b(?:0|1|2)\s*years?\b",
        r"\bnew grad\b",
        r"\bfresh\b",
    ]

    # Mid level indicators
    mid_patterns = [
        r"\bmid.level\b",
        r"\bintermediate\b",
        r"\bmiddle\b",
        r"\b(?:3|4|5)\s*years?\b",
    ]

    for pattern in senior_patterns:
        if re.search(pattern, text_lower):
            return "senior"

    for pattern in junior_patterns:
        if re.search(pattern, text_lower):
            return "junior"

    for pattern in mid_patterns:
        if re.search(pattern, text_lower):
            return "mid"

    return "mid"  # Default assumption


def normalize_skill_name(skill: str) -> str:
    """Normalize skill names for consistent comparison"""
    return re.sub(r"[^\w]", "", skill.lower().strip())


def calculate_skill_score(skill: str, context: str = "") -> float:
    """Calculate a score for a skill based on context"""
    base_score = 50.0
    skill_lower = skill.lower()

    # High-demand skills get bonus points
    high_demand_skills = [
        "python",
        "javascript",
        "react",
        "aws",
        "kubernetes",
        "docker",
        "typescript",
        "node.js",
        "sql",
    ]

    if any(high_skill in skill_lower for high_skill in high_demand_skills):
        base_score += 30.0

    # Emerging/trending skills
    emerging_skills = ["rust", "go", "svelte", "deno", "webassembly"]
    if any(emerging in skill_lower for emerging in emerging_skills):
        base_score += 20.0

    # Context-based adjustments
    if context:
        context_lower = context.lower()
        if "senior" in context_lower and skill_lower in [
            "python",
            "java",
            "javascript",
        ]:
            base_score += 15.0
        elif "data science" in context_lower and skill_lower in ["python", "r", "sql"]:
            base_score += 25.0

    return min(100.0, base_score)
