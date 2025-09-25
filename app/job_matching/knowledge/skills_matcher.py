# app/job_matching/knowledge/skills_matcher.py
"""
Universal Skills Matching Algorithm
Compares resume skills against job requirements across all industries
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher
from .industry_detector import UniversalIndustryDetector, IndustryType
from .universal_knowledge_data import (
    UNIVERSAL_INDUSTRY_CLASSIFICATION,
    UNIVERSAL_TRANSFERABLE_SKILLS,
)


class MatchStrength(Enum):
    EXACT = "exact"
    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"
    NONE = "none"


@dataclass
class SkillMatch:
    skill: str
    match_strength: MatchStrength
    resume_evidence: List[str]
    importance_weight: float
    match_score: float


@dataclass
class ExperienceMatch:
    required_level: str
    candidate_level: str
    years_required: Optional[int]
    years_candidate: Optional[int]
    match_score: float
    reasoning: str


@dataclass
class QualificationMatch:
    qualification_type: str  # education, license, certification
    required: List[str]
    candidate_has: List[str]
    missing: List[str]
    match_percentage: float


@dataclass
class UniversalMatchResult:
    overall_match_score: float
    industry: str
    role: str
    experience_match: ExperienceMatch
    skill_matches: List[SkillMatch]
    qualification_matches: List[QualificationMatch]
    strengths: List[str]
    gaps: List[str]
    recommendations: List[str]
    should_apply: bool
    confidence_level: str


class UniversalSkillsMatcher:
    """
    Matches candidate profiles against job requirements across all industries
    """

    def __init__(self):
        self.industry_detector = UniversalIndustryDetector()
        self.industry_data = UNIVERSAL_INDUSTRY_CLASSIFICATION
        self.transferable_skills = UNIVERSAL_TRANSFERABLE_SKILLS

    def analyze_match(
        self, resume_text: str, job_title: str, job_description: str
    ) -> UniversalMatchResult:
        """
        Complete matching analysis between resume and job posting
        """

        # Step 1: Analyze the job posting
        job_analysis = self.industry_detector.analyze_job_posting(
            job_title, job_description
        )

        # Step 2: Analyze the resume
        resume_analysis = self._analyze_resume(resume_text)

        # Step 3: Match experience levels
        experience_match = self._match_experience_level(
            resume_analysis["experience"], job_analysis["experience_level"]
        )

        # Step 4: Match skills
        skill_matches = self._match_skills(
            resume_analysis["skills"],
            job_analysis["skills_mentioned"],
            job_analysis["industry_requirements"],
            job_analysis["industry_detection"]["primary_industry"],
        )

        # Step 5: Match qualifications
        qualification_matches = self._match_qualifications(
            resume_analysis["qualifications"], job_analysis["industry_requirements"]
        )

        # Step 6: Calculate overall score
        overall_score = self._calculate_overall_score(
            experience_match, skill_matches, qualification_matches
        )

        # Step 7: Generate insights
        strengths, gaps, recommendations = self._generate_insights(
            skill_matches, qualification_matches, experience_match, job_analysis
        )

        # Step 8: Determine recommendation
        should_apply, confidence = self._determine_application_recommendation(
            overall_score, experience_match, skill_matches, qualification_matches
        )

        return UniversalMatchResult(
            overall_match_score=overall_score,
            industry=job_analysis["industry_detection"]["primary_industry"],
            role=job_analysis["industry_detection"]["detected_role"] or job_title,
            experience_match=experience_match,
            skill_matches=skill_matches,
            qualification_matches=qualification_matches,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
            should_apply=should_apply,
            confidence_level=confidence,
        )

    def _analyze_resume(self, resume_text: str) -> Dict[str, any]:
        """Extract information from resume text"""

        resume_lower = resume_text.lower()

        # Extract experience duration
        experience_years = self._extract_experience_years(resume_text)
        experience_level = self._determine_candidate_level(
            resume_text, experience_years
        )

        # Extract skills
        skills = self._extract_resume_skills(resume_text)

        # Extract qualifications
        qualifications = self._extract_qualifications(resume_text)

        # Extract work history indicators
        work_history = self._extract_work_history_indicators(resume_text)

        return {
            "experience": {
                "years": experience_years,
                "level": experience_level,
                "work_history": work_history,
            },
            "skills": skills,
            "qualifications": qualifications,
            "text": resume_text,
        }

    def _extract_experience_years(self, resume_text: str) -> Optional[int]:
        """Extract years of experience from resume"""

        patterns = [
            r"(\d+)\s*(?:\+|plus)?\s*years?\s*(?:of\s*)?experience",
            r"(\d+)\s*(?:\+|plus)?\s*years?\s*(?:in|with|of)",
            r"over\s*(\d+)\s*years?",
            r"more than\s*(\d+)\s*years?",
        ]

        years_found = []
        for pattern in patterns:
            matches = re.findall(pattern, resume_text.lower())
            years_found.extend([int(match) for match in matches])

        # Also try to calculate from work history dates
        date_ranges = self._extract_date_ranges(resume_text)
        if date_ranges:
            calculated_years = sum(date_ranges)
            years_found.append(calculated_years)

        return max(years_found) if years_found else None

    def _extract_date_ranges(self, resume_text: str) -> List[int]:
        """Extract work duration from date ranges in resume"""

        # Look for date patterns like "2020-2023", "Jan 2020 - Present", etc.
        date_patterns = [
            r"(\d{4})\s*[-–]\s*(\d{4})",
            r"(\d{4})\s*[-–]\s*(?:present|current|now)",
        ]

        durations = []
        current_year = 2024  # You might want to make this dynamic

        for pattern in date_patterns:
            matches = re.findall(pattern, resume_text.lower())
            for match in matches:
                start_year = int(match[0])
                if len(match) > 1 and match[1].isdigit():
                    end_year = int(match[1])
                else:
                    end_year = current_year

                duration = end_year - start_year
                if 0 <= duration <= 50:  # Sanity check
                    durations.append(duration)

        return durations

    def _determine_candidate_level(
        self, resume_text: str, experience_years: Optional[int]
    ) -> str:
        """Determine candidate's experience level"""

        text_lower = resume_text.lower()

        # Check for explicit level indicators
        senior_indicators = [
            "senior",
            "lead",
            "principal",
            "manager",
            "director",
            "head of",
            "chief",
        ]
        entry_indicators = ["entry", "junior", "graduate", "new", "trainee", "intern"]

        if any(indicator in text_lower for indicator in senior_indicators):
            return "Senior Level"
        elif any(indicator in text_lower for indicator in entry_indicators):
            return "Entry Level"

        # Use years of experience
        if experience_years:
            if experience_years >= 8:
                return "Senior Level"
            elif experience_years >= 3:
                return "Mid Level"
            else:
                return "Entry Level"

        return "Mid Level"  # Default

    def _extract_resume_skills(self, resume_text: str) -> Dict[str, List[str]]:
        """Extract skills from resume text"""

        text_lower = resume_text.lower()

        skills = {
            "technical_skills": [],
            "soft_skills": [],
            "certifications": [],
            "software_tools": [],
            "programming_languages": [],
            "frameworks": [],
        }

        # Define skill databases
        skill_databases = {
            "programming_languages": [
                "python",
                "java",
                "javascript",
                "c++",
                "c#",
                "php",
                "ruby",
                "go",
                "rust",
                "typescript",
                "swift",
                "kotlin",
                "scala",
                "r",
                "sql",
            ],
            "frameworks": [
                "react",
                "angular",
                "vue",
                "node",
                "django",
                "flask",
                "spring",
                "laravel",
                "express",
                "next.js",
                "svelte",
            ],
            "software_tools": [
                "excel",
                "word",
                "powerpoint",
                "photoshop",
                "autocad",
                "salesforce",
                "quickbooks",
                "sap",
                "oracle",
                "tableau",
                "power bi",
                "git",
                "docker",
            ],
            "certifications": [
                "cpa",
                "cfa",
                "pmp",
                "cissp",
                "aws",
                "azure",
                "google cloud",
                "scrum master",
                "six sigma",
                "itil",
                "cisco",
                "microsoft",
            ],
        }

        # Extract from predefined lists
        for category, skill_list in skill_databases.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    skills[category].append(skill)

        # Extract transferable skills
        for skill_name, skill_data in self.transferable_skills.items():
            for variation in skill_data["variations"]:
                if variation.lower() in text_lower:
                    skills["soft_skills"].append(variation)

        # Extract technical skills using patterns
        technical_patterns = [
            r"experience with ([a-zA-Z\s]+)",
            r"proficient in ([a-zA-Z\s]+)",
            r"skilled in ([a-zA-Z\s]+)",
            r"knowledge of ([a-zA-Z\s]+)",
        ]

        for pattern in technical_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned.split()) <= 3:  # Avoid long phrases
                    skills["technical_skills"].append(cleaned)

        # Remove duplicates and clean up
        for category in skills:
            skills[category] = list(
                set(
                    [
                        skill.strip().title()
                        for skill in skills[category]
                        if skill.strip()
                    ]
                )
            )

        return skills

    def _extract_qualifications(self, resume_text: str) -> Dict[str, List[str]]:
        """Extract education, licenses, and certifications"""

        text_lower = resume_text.lower()

        qualifications = {"education": [], "licenses": [], "certifications": []}

        # Education patterns
        education_patterns = [
            r"bachelor'?s?\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)",
            r"master'?s?\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)",
            r"phd\s+(?:in\s+)?([a-zA-Z\s]+)",
            r"associate'?s?\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)",
            r"(\w+)\s+degree",
            r"graduated\s+(?:from\s+)?([a-zA-Z\s]+)",
        ]

        for pattern in education_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) < 50:  # Reasonable length
                    qualifications["education"].append(cleaned.title())

        # License patterns
        license_patterns = [
            r"licensed\s+([a-zA-Z\s]+)",
            r"(\w+)\s+license",
            r"state\s+licensed\s+([a-zA-Z\s]+)",
        ]

        for pattern in license_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) < 30:
                    qualifications["licenses"].append(cleaned.title())

        # Certification patterns
        cert_patterns = [
            r"certified\s+([a-zA-Z\s]+)",
            r"(\w+)\s+certified",
            r"certification\s+in\s+([a-zA-Z\s]+)",
        ]

        for pattern in cert_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) < 30:
                    qualifications["certifications"].append(cleaned.title())

        # Remove duplicates
        for category in qualifications:
            qualifications[category] = list(set(qualifications[category]))

        return qualifications

    def _extract_work_history_indicators(self, resume_text: str) -> List[str]:
        """Extract work history indicators for experience assessment"""

        indicators = []
        text_lower = resume_text.lower()

        # Look for progression indicators
        progression_terms = [
            "promoted",
            "advanced",
            "progressed",
            "elevated",
            "moved up",
            "increased responsibility",
            "expanded role",
            "team lead",
            "management",
        ]

        for term in progression_terms:
            if term in text_lower:
                indicators.append(f"Career progression: {term}")

        # Look for leadership indicators
        leadership_terms = [
            "managed",
            "led",
            "supervised",
            "directed",
            "coordinated",
            "oversaw",
            "mentored",
            "trained",
            "guided",
        ]

        for term in leadership_terms:
            if term in text_lower:
                indicators.append(f"Leadership: {term}")

        return indicators

    def _match_experience_level(
        self, candidate_exp: Dict, job_exp: Dict
    ) -> ExperienceMatch:
        """Match candidate experience against job requirements"""

        candidate_level = candidate_exp["level"]
        required_level = job_exp["level"]
        candidate_years = candidate_exp.get("years")

        # Extract required years from job description
        required_years = None
        if job_exp.get("reasoning"):
            for reason in job_exp["reasoning"]:
                if "years" in reason:
                    match = re.search(r"(\d+)", reason)
                    if match:
                        required_years = int(match.group(1))
                        break

        # Calculate match score
        level_hierarchy = {
            "Entry Level": 1,
            "Mid Level": 2,
            "Senior Level": 3,
            "Management Level": 4,
        }

        candidate_rank = level_hierarchy.get(candidate_level, 2)
        required_rank = level_hierarchy.get(required_level, 2)

        # Base score on level match
        if candidate_rank == required_rank:
            base_score = 100
        elif abs(candidate_rank - required_rank) == 1:
            base_score = 75
        else:
            base_score = 50

        # Adjust based on years if available
        years_adjustment = 0
        if candidate_years and required_years:
            if candidate_years >= required_years:
                years_adjustment = min(20, (candidate_years - required_years) * 5)
            else:
                years_adjustment = -((required_years - candidate_years) * 10)

        final_score = max(0, min(100, base_score + years_adjustment))

        # Generate reasoning
        if candidate_rank > required_rank:
            reasoning = (
                f"Overqualified: {candidate_level} for {required_level} position"
            )
        elif candidate_rank < required_rank:
            reasoning = (
                f"Underqualified: {candidate_level} for {required_level} position"
            )
        else:
            reasoning = f"Good match: {candidate_level} aligns with {required_level} requirements"

        if candidate_years and required_years:
            reasoning += f" ({candidate_years} years vs {required_years} required)"

        return ExperienceMatch(
            required_level=required_level,
            candidate_level=candidate_level,
            years_required=required_years,
            years_candidate=candidate_years,
            match_score=final_score,
            reasoning=reasoning,
        )

    def _match_skills(
        self,
        candidate_skills: Dict,
        job_skills: Dict,
        requirements: Dict,
        industry: str,
    ) -> List[SkillMatch]:
        """Match candidate skills against job requirements"""

        skill_matches = []

        # Get all candidate skills in one list
        all_candidate_skills = []
        for category, skills in candidate_skills.items():
            all_candidate_skills.extend([skill.lower() for skill in skills])

        # Get all required skills
        all_required_skills = []
        for category, skills in job_skills.items():
            all_required_skills.extend([skill.lower() for skill in skills])

        # Add skills from industry requirements
        for category, skills in requirements.items():
            if category in ["critical_skills", "technology_skills"]:
                all_required_skills.extend([skill.lower() for skill in skills])

        # Remove duplicates
        all_required_skills = list(set(all_required_skills))

        # Match each required skill
        for required_skill in all_required_skills:
            match = self._find_skill_match(
                required_skill, all_candidate_skills, candidate_skills
            )
            if match:
                skill_matches.append(match)

        # Sort by importance (exact matches first, then by match score)
        skill_matches.sort(
            key=lambda x: (x.match_strength.value != "exact", -x.match_score)
        )

        return skill_matches

    def _find_skill_match(
        self,
        required_skill: str,
        candidate_skills: List[str],
        candidate_skills_dict: Dict,
    ) -> Optional[SkillMatch]:
        """Find the best match for a required skill"""

        # Exact match
        if required_skill in candidate_skills:
            return SkillMatch(
                skill=required_skill.title(),
                match_strength=MatchStrength.EXACT,
                resume_evidence=[required_skill.title()],
                importance_weight=1.0,
                match_score=100,
            )

        # Partial matches using string similarity
        best_match = None
        best_score = 0

        for candidate_skill in candidate_skills:
            similarity = SequenceMatcher(None, required_skill, candidate_skill).ratio()

            if similarity > 0.7:  # Strong similarity
                if similarity > best_score:
                    best_score = similarity
                    best_match = SkillMatch(
                        skill=required_skill.title(),
                        match_strength=MatchStrength.STRONG
                        if similarity > 0.85
                        else MatchStrength.PARTIAL,
                        resume_evidence=[candidate_skill.title()],
                        importance_weight=0.8,
                        match_score=similarity * 100,
                    )

        # Check for related skills (e.g., "patient care" matches "healthcare")
        if not best_match:
            related_match = self._find_related_skill_match(
                required_skill, candidate_skills
            )
            if related_match:
                return related_match

        return best_match

    def _find_related_skill_match(
        self, required_skill: str, candidate_skills: List[str]
    ) -> Optional[SkillMatch]:
        """Find related skills that might satisfy the requirement"""

        # Define skill relationships
        skill_relations = {
            "patient care": ["healthcare", "nursing", "medical", "clinical"],
            "customer service": ["client relations", "customer support", "sales"],
            "programming": ["coding", "development", "software"],
            "leadership": ["management", "supervision", "team lead"],
            "communication": ["presentation", "public speaking", "writing"],
        }

        required_words = set(required_skill.split())

        for candidate_skill in candidate_skills:
            candidate_words = set(candidate_skill.split())

            # Check for word overlap
            if required_words & candidate_words:
                return SkillMatch(
                    skill=required_skill.title(),
                    match_strength=MatchStrength.WEAK,
                    resume_evidence=[candidate_skill.title()],
                    importance_weight=0.5,
                    match_score=50,
                )

        return None

    def _match_qualifications(
        self, candidate_quals: Dict, requirements: Dict
    ) -> List[QualificationMatch]:
        """Match qualifications (education, licenses, certifications)"""

        qualification_matches = []

        # Map requirement categories to candidate categories
        qual_mapping = {
            "required_education": "education",
            "required_licenses": "licenses",
            "critical_certifications": "certifications",
        }

        for req_category, candidate_category in qual_mapping.items():
            if req_category in requirements:
                required = requirements[req_category]
                candidate_has = candidate_quals.get(candidate_category, [])

                if required:
                    # Find matches
                    matches = []
                    missing = []

                    for req in required:
                        found = False
                        for candidate_qual in candidate_has:
                            if (
                                req.lower() in candidate_qual.lower()
                                or candidate_qual.lower() in req.lower()
                            ):
                                matches.append(candidate_qual)
                                found = True
                                break

                        if not found:
                            missing.append(req)

                    match_percentage = (
                        (len(matches) / len(required)) * 100 if required else 100
                    )

                    qualification_matches.append(
                        QualificationMatch(
                            qualification_type=req_category.replace(
                                "required_", ""
                            ).replace("critical_", ""),
                            required=required,
                            candidate_has=matches,
                            missing=missing,
                            match_percentage=match_percentage,
                        )
                    )

        return qualification_matches

    def _calculate_overall_score(
        self,
        experience_match: ExperienceMatch,
        skill_matches: List[SkillMatch],
        qual_matches: List[QualificationMatch],
    ) -> float:
        """Calculate overall match score"""

        # Weights for different components
        weights = {"experience": 0.3, "skills": 0.5, "qualifications": 0.2}

        # Experience score
        experience_score = experience_match.match_score

        # Skills score
        if skill_matches:
            skill_scores = [
                match.match_score * match.importance_weight for match in skill_matches
            ]
            skills_score = sum(skill_scores) / len(skill_scores)
        else:
            skills_score = 50  # Neutral if no specific skills identified

        # Qualifications score
        if qual_matches:
            qual_scores = [match.match_percentage for match in qual_matches]
            qualifications_score = sum(qual_scores) / len(qual_scores)
        else:
            qualifications_score = 75  # Assume reasonable if no specific requirements

        # Calculate weighted average
        overall_score = (
            experience_score * weights["experience"]
            + skills_score * weights["skills"]
            + qualifications_score * weights["qualifications"]
        )

        return round(overall_score, 1)

    def _generate_insights(
        self,
        skill_matches: List[SkillMatch],
        qual_matches: List[QualificationMatch],
        experience_match: ExperienceMatch,
        job_analysis: Dict,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate strengths, gaps, and recommendations"""

        strengths = []
        gaps = []
        recommendations = []

        # Experience insights
        if experience_match.match_score >= 80:
            strengths.append(f"Strong experience match: {experience_match.reasoning}")
        elif experience_match.match_score < 60:
            gaps.append(f"Experience gap: {experience_match.reasoning}")

        # Skills insights
        exact_matches = [
            m for m in skill_matches if m.match_strength == MatchStrength.EXACT
        ]
        if exact_matches:
            strengths.append(
                f"Direct skill matches: {', '.join([m.skill for m in exact_matches[:3]])}"
            )

        missing_skills = []
        for match in skill_matches:
            if match.match_score < 70:
                missing_skills.append(match.skill)

        if missing_skills:
            gaps.append(f"Skill gaps in: {', '.join(missing_skills[:3])}")

        # Qualification insights
        for qual_match in qual_matches:
            if qual_match.match_percentage >= 80:
                strengths.append(f"Meets {qual_match.qualification_type} requirements")
            elif qual_match.missing:
                gaps.append(
                    f"Missing {qual_match.qualification_type}: {', '.join(qual_match.missing[:2])}"
                )

        # Generate recommendations
        if experience_match.match_score < 70:
            if (
                "Entry Level" in experience_match.candidate_level
                and "Senior" in experience_match.required_level
            ):
                recommendations.append(
                    "Consider applying to entry or mid-level positions to build experience"
                )
            else:
                recommendations.append(
                    "Highlight relevant experience and transferable skills"
                )

        if missing_skills:
            recommendations.append(
                f"Consider developing skills in: {', '.join(missing_skills[:2])}"
            )

        for qual_match in qual_matches:
            if qual_match.missing:
                recommendations.append(
                    f"Consider obtaining: {', '.join(qual_match.missing[:2])}"
                )

        # Industry-specific recommendations
        industry = job_analysis["industry_detection"]["primary_industry"]
        if industry == "healthcare_medical":
            recommendations.append(
                "Emphasize patient care experience and clinical skills"
            )
        elif industry == "education":
            recommendations.append(
                "Highlight classroom management and student engagement experience"
            )
        elif industry == "technology":
            recommendations.append(
                "Showcase technical projects and problem-solving abilities"
            )

        return strengths, gaps, recommendations

    def _determine_application_recommendation(
        self,
        overall_score: float,
        experience_match: ExperienceMatch,
        skill_matches: List[SkillMatch],
        qual_matches: List[QualificationMatch],
    ) -> Tuple[bool, str]:
        """Determine whether candidate should apply and confidence level"""

        # Strong match criteria
        if overall_score >= 80:
            return True, "High"

        # Good match criteria
        elif overall_score >= 65:
            return True, "Medium"

        # Borderline criteria
        elif overall_score >= 50:
            # Check for deal-breakers
            critical_missing = any(qual.match_percentage < 50 for qual in qual_matches)
            severe_experience_gap = experience_match.match_score < 40

            if critical_missing or severe_experience_gap:
                return False, "Low"
            else:
                return True, "Low"

        # Poor match
        else:
            return False, "Very Low"
