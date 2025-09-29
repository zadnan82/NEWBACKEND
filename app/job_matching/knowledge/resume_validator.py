# app/job_matching/knowledge/resume_validator.py
"""
Resume content validator to prevent empty/insufficient resumes from getting high scores
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResumeValidationResult:
    is_valid: bool
    content_score: float  # 0-100 based on content quality
    issues: List[str]
    warnings: List[str]
    word_count: int
    has_work_experience: bool
    has_education: bool
    has_skills: bool
    has_contact_info: bool


class ResumeContentValidator:
    """Validates resume content quality before job matching"""

    def __init__(self):
        self.min_word_count = 50
        self.min_content_score = 30

    def validate_resume(self, resume_text: str) -> ResumeValidationResult:
        """Comprehensive resume validation"""

        if not resume_text or len(resume_text.strip()) < 10:
            return ResumeValidationResult(
                is_valid=False,
                content_score=0.0,
                issues=["Resume is empty or too short"],
                warnings=[],
                word_count=0,
                has_work_experience=False,
                has_education=False,
                has_skills=False,
                has_contact_info=False,
            )

        text_lower = resume_text.lower()
        word_count = len(resume_text.split())

        issues = []
        warnings = []
        content_score = 0

        # Check minimum length
        if word_count < self.min_word_count:
            issues.append(
                f"Resume too short ({word_count} words, minimum {self.min_word_count})"
            )
        else:
            content_score += 20

        # Check for work experience indicators
        has_work_experience = self._check_work_experience(text_lower)
        if has_work_experience:
            content_score += 25
        else:
            issues.append("No clear work experience found")

        # Check for education indicators
        has_education = self._check_education(text_lower)
        if has_education:
            content_score += 15
        else:
            warnings.append("No education information found")

        # Check for skills
        has_skills = self._check_skills(text_lower)
        if has_skills:
            content_score += 25
        else:
            issues.append("No clear skills or competencies found")

        # Check for contact information
        has_contact_info = self._check_contact_info(text_lower)
        if has_contact_info:
            content_score += 15
        else:
            warnings.append("Limited contact information")

        # Check for resume structure
        if self._check_resume_structure(resume_text):
            content_score += 10
        else:
            warnings.append("Resume lacks clear structure/formatting")

        # Determine if valid
        is_valid = (
            len(issues) == 0
            and word_count >= self.min_word_count
            and content_score >= self.min_content_score
        )

        return ResumeValidationResult(
            is_valid=is_valid,
            content_score=min(100, content_score),
            issues=issues,
            warnings=warnings,
            word_count=word_count,
            has_work_experience=has_work_experience,
            has_education=has_education,
            has_skills=has_skills,
            has_contact_info=has_contact_info,
        )

    def _check_work_experience(self, text: str) -> bool:
        """Check for work experience indicators"""
        experience_indicators = [
            # Job titles
            r"\b(manager|developer|engineer|analyst|specialist|coordinator|assistant|director|supervisor|lead|senior|junior)\b",
            # Work-related verbs
            r"\b(worked|employed|managed|developed|created|implemented|designed|led|supervised|coordinated|responsible for)\b",
            # Company/employment indicators
            r"\b(company|corporation|inc\.|llc|organization|employer|workplace)\b",
            # Time periods that suggest employment
            r"\b(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|present|current)\b",
            # Work experience section headers
            r"\b(experience|employment|work history|professional experience|career)\b",
        ]

        found_indicators = 0
        for pattern in experience_indicators:
            if re.search(pattern, text):
                found_indicators += 1

        return found_indicators >= 2  # Need at least 2 indicators

    def _check_education(self, text: str) -> bool:
        """Check for education indicators"""
        education_indicators = [
            r"\b(bachelor|master|phd|doctorate|degree|diploma|certificate|graduation|graduated|university|college|school)\b",
            r"\b(bs|ba|ms|ma|mba|bsc|msc)\b",
            r"\b(associate|undergraduate|graduate|postgraduate)\b",
        ]

        for pattern in education_indicators:
            if re.search(pattern, text):
                return True
        return False

    def _check_skills(self, text: str) -> bool:
        """Check for skills/competencies"""
        skills_indicators = [
            # Skills section
            r"\b(skills|competencies|abilities|expertise|proficient|experienced|knowledge)\b",
            # Technical skills
            r"\b(programming|coding|software|technology|technical|computer|system)\b",
            # Soft skills
            r"\b(communication|leadership|teamwork|problem solving|analytical|creative)\b",
            # Professional skills
            r"\b(management|planning|organization|negotiation|presentation)\b",
        ]

        found_indicators = 0
        for pattern in skills_indicators:
            if re.search(pattern, text):
                found_indicators += 1

        return found_indicators >= 2

    def _check_contact_info(self, text: str) -> bool:
        """Check for contact information"""
        contact_patterns = [
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",  # Email
            r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone
            r"\b(phone|email|contact|address)\b",  # Contact labels
        ]

        for pattern in contact_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _check_resume_structure(self, text: str) -> bool:
        """Check for resume structure"""
        structure_indicators = [
            r"\b(summary|objective|experience|education|skills|employment|work history)\b",
            # Section headers (lines that end with colons or are in caps)
            r"^[A-Z\s]+:?\s*$",
            # Bullet points
            r"^\s*[•\-\*]\s+",
        ]

        found_indicators = 0
        lines = text.split("\n")

        for line in lines:
            for pattern in structure_indicators:
                if re.search(pattern, line, re.MULTILINE):
                    found_indicators += 1
                    break

        return found_indicators >= 3


# Updated scoring logic for skills_matcher.py
class FixedScoring:
    """Fixed scoring that penalizes empty/insufficient resumes"""

    @staticmethod
    def calculate_realistic_overall_score(
        validation_result: ResumeValidationResult,
        experience_match,
        skill_matches: List,
        qual_matches: List,
    ) -> float:
        """Calculate score that properly penalizes insufficient content"""

        # If resume is invalid, cap the score severely
        if not validation_result.is_valid:
            base_penalty = 100 - validation_result.content_score
            return max(10, 40 - base_penalty)  # Max 40% for invalid resumes

        # Content quality modifier (0.5 to 1.0)
        content_modifier = validation_result.content_score / 100

        # Component scores
        experience_score = (
            experience_match.match_score
            if hasattr(experience_match, "match_score")
            else 50
        )

        # Skills score - penalize heavily for no skills found
        if skill_matches and len(skill_matches) > 0:
            skill_scores = [
                match.match_score * getattr(match, "importance_weight", 1.0)
                for match in skill_matches
            ]
            skills_score = sum(skill_scores) / len(skill_scores)
        else:
            # FIXED: Heavy penalty for no skills
            skills_score = 15 if validation_result.has_skills else 5

        # Qualifications score - more realistic defaults
        if qual_matches and len(qual_matches) > 0:
            qual_scores = [
                getattr(match, "match_percentage", 50) for match in qual_matches
            ]
            qualifications_score = sum(qual_scores) / len(qual_scores)
        else:
            # FIXED: Modest score only if education found
            qualifications_score = 50 if validation_result.has_education else 25

        # Weights
        weights = {"experience": 0.3, "skills": 0.5, "qualifications": 0.2}

        # Calculate base score
        base_score = (
            experience_score * weights["experience"]
            + skills_score * weights["skills"]
            + qualifications_score * weights["qualifications"]
        )

        # Apply content quality modifier
        final_score = base_score * content_modifier

        # Additional penalties for missing critical elements
        if not validation_result.has_work_experience:
            final_score *= 0.7  # 30% penalty for no work experience

        if not validation_result.has_skills:
            final_score *= 0.6  # 40% penalty for no skills

        return round(max(5, min(95, final_score)), 1)


def validate_and_score_resume(
    resume_text: str, job_title: str = "", job_description: str = ""
) -> Dict:
    """Utility function to validate resume and return early if insufficient"""

    validator = ResumeContentValidator()
    validation = validator.validate_resume(resume_text)

    if not validation.is_valid:
        return {
            "overall_match_score": max(
                5, validation.content_score * 0.4
            ),  # Cap at 40% of content score
            "validation_issues": validation.issues,
            "validation_warnings": validation.warnings,
            "content_quality": validation.content_score,
            "is_valid_resume": False,
            "recommendation": "Please provide a more complete resume with work experience and skills",
            "should_apply": False,
            "confidence_level": "Very Low",
            "strengths": [],
            "gaps": [
                "Insufficient resume content",
                "Missing work experience details",
                "Missing skills information",
            ],
            "recommendations": [
                "Add detailed work experience with specific accomplishments",
                "Include a skills section with relevant technical and soft skills",
                f"Expand resume to at least {validator.min_word_count} words",
                "Include education background and certifications",
            ],
        }

    # If validation passes, return None to continue with normal matching
    return None
