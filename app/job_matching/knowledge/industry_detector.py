# app/job_matching/knowledge/industry_detector.py
"""
Universal Industry Detection Engine
Automatically identifies industry and role from job postings and resumes
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from .universal_knowledge_data import (
    UNIVERSAL_INDUSTRY_CLASSIFICATION,
    UNIVERSAL_TRANSFERABLE_SKILLS,
    UNIVERSAL_EXPERIENCE_LEVELS,
)


class IndustryType(Enum):
    HEALTHCARE_MEDICAL = "healthcare_medical"
    EDUCATION = "education"
    FINANCE_BANKING = "finance_banking"
    TECHNOLOGY = "technology"
    RETAIL_SALES = "retail_sales"
    MANUFACTURING = "manufacturing"
    LEGAL = "legal"
    HOSPITALITY_FOOD = "hospitality_food"
    CONSTRUCTION = "construction"
    TRANSPORTATION = "transportation"
    GOVERNMENT = "government"
    UNKNOWN = "unknown"


@dataclass
class IndustryDetectionResult:
    primary_industry: IndustryType
    confidence_score: float
    detected_role: Optional[str]
    alternative_industries: List[Tuple[IndustryType, float]]
    reasoning: List[str]


class UniversalIndustryDetector:
    """
    Detects industry and role from job titles, descriptions, and resume content
    """

    def __init__(self):
        self.industry_data = UNIVERSAL_INDUSTRY_CLASSIFICATION
        self.industry_keywords = self._build_industry_keyword_index()
        self.role_keywords = self._build_role_keyword_index()

    def _build_industry_keyword_index(self) -> Dict[IndustryType, List[str]]:
        """Build comprehensive keyword index for each industry"""

        return {
            IndustryType.HEALTHCARE_MEDICAL: [
                # Direct terms
                "nurse",
                "nursing",
                "medical",
                "healthcare",
                "health",
                "hospital",
                "clinic",
                "patient",
                "clinical",
                "physician",
                "doctor",
                "rn",
                "lpn",
                "cna",
                "bsn",
                "medical assistant",
                "pharmacist",
                "pharmacy",
                "radiology",
                "laboratory",
                "surgery",
                "surgical",
                "emergency",
                "icu",
                "critical care",
                "pediatric",
                "oncology",
                "cardiology",
                "neurology",
                "psychiatry",
                "mental health",
                # Skills and certifications
                "acls",
                "bls",
                "pals",
                "ccrn",
                "hipaa",
                "medication",
                "iv",
                "vital signs",
                "infection control",
                "patient care",
                "medical records",
                "ehr",
                "emr",
                # Work environments
                "hospital",
                "clinic",
                "medical center",
                "nursing home",
                "rehab",
                "dialysis",
            ],
            IndustryType.EDUCATION: [
                # Direct terms
                "teacher",
                "teaching",
                "education",
                "school",
                "university",
                "college",
                "instructor",
                "professor",
                "academic",
                "classroom",
                "student",
                "curriculum",
                "lesson",
                "grade",
                "elementary",
                "middle school",
                "high school",
                "kindergarten",
                "principal",
                "administrator",
                "superintendent",
                "counselor",
                "librarian",
                # Skills and concepts
                "lesson planning",
                "classroom management",
                "assessment",
                "pedagogy",
                "differentiated instruction",
                "iep",
                "504",
                "special education",
                "esl",
                "state standards",
                "common core",
                "standardized testing",
                "parent communication",
                # Subjects
                "mathematics",
                "science",
                "english",
                "history",
                "art",
                "physical education",
                "music",
                "foreign language",
                "reading",
                "writing",
            ],
            IndustryType.FINANCE_BANKING: [
                # Direct terms
                "finance",
                "financial",
                "banking",
                "bank",
                "accounting",
                "accountant",
                "investment",
                "insurance",
                "credit",
                "loan",
                "mortgage",
                "tax",
                "audit",
                "budget",
                "analyst",
                "advisor",
                "planner",
                "controller",
                "treasurer",
                # Skills and certifications
                "cpa",
                "cfa",
                "cma",
                "cia",
                "gaap",
                "financial statements",
                "reconciliation",
                "accounts payable",
                "accounts receivable",
                "payroll",
                "bookkeeping",
                "financial analysis",
                "risk management",
                "compliance",
                "regulations",
                # Software and tools
                "quickbooks",
                "sap",
                "oracle",
                "excel",
                "financial modeling",
                "tax software",
                # Work environments
                "bank",
                "credit union",
                "investment firm",
                "accounting firm",
                "brokerage",
            ],
            IndustryType.TECHNOLOGY: [
                # Direct terms
                "software",
                "developer",
                "engineer",
                "programmer",
                "coding",
                "programming",
                "it",
                "tech",
                "computer",
                "system",
                "network",
                "database",
                "web",
                "mobile",
                "data scientist",
                "analyst",
                "architect",
                "devops",
                "cybersecurity",
                "qa",
                # Programming languages
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
                "html",
                "css",
                # Frameworks and tools
                "react",
                "angular",
                "vue",
                "node",
                "django",
                "flask",
                "spring",
                "laravel",
                "docker",
                "kubernetes",
                "aws",
                "azure",
                "gcp",
                "git",
                "linux",
                "agile",
                # Concepts
                "machine learning",
                "artificial intelligence",
                "ai",
                "ml",
                "cloud computing",
                "api",
                "microservices",
                "frontend",
                "backend",
                "full stack",
                "ui",
                "ux",
            ],
            IndustryType.RETAIL_SALES: [
                # Direct terms
                "retail",
                "sales",
                "cashier",
                "customer service",
                "store",
                "shop",
                "mall",
                "merchandise",
                "inventory",
                "pos",
                "commission",
                "associate",
                "representative",
                "manager",
                "supervisor",
                "assistant",
                "clerk",
                "specialist",
                # Skills and activities
                "customer relations",
                "product knowledge",
                "upselling",
                "cross-selling",
                "cash handling",
                "payment processing",
                "returns",
                "exchanges",
                "complaints",
                "visual merchandising",
                "stock",
                "warehouse",
                "shipping",
                "receiving",
                # Work environments
                "department store",
                "boutique",
                "chain store",
                "outlet",
                "showroom",
                "e-commerce",
                "online retail",
                "call center",
                "customer support",
            ],
            IndustryType.MANUFACTURING: [
                # Direct terms
                "manufacturing",
                "production",
                "factory",
                "plant",
                "assembly",
                "fabrication",
                "operator",
                "technician",
                "maintenance",
                "quality",
                "control",
                "inspector",
                "supervisor",
                "foreman",
                "engineer",
                "manager",
                "coordinator",
                # Processes and concepts
                "lean manufacturing",
                "six sigma",
                "5s",
                "kaizen",
                "continuous improvement",
                "safety protocols",
                "osha",
                "quality assurance",
                "iso",
                "sop",
                "procedures",
                "machinery",
                "equipment",
                "tools",
                "automation",
                "robotics",
                "cnc",
                # Materials and products
                "automotive",
                "aerospace",
                "electronics",
                "textiles",
                "chemical",
                "food processing",
                "pharmaceutical",
                "medical device",
                "consumer goods",
                "industrial",
            ],
            IndustryType.LEGAL: [
                # Direct terms
                "legal",
                "law",
                "attorney",
                "lawyer",
                "paralegal",
                "counsel",
                "advocate",
                "judge",
                "court",
                "litigation",
                "contract",
                "agreement",
                "case",
                "client",
                "firm",
                "practice",
                "bar",
                "license",
                "admitted",
                "juris doctor",
                "jd",
                # Legal areas
                "corporate law",
                "criminal law",
                "family law",
                "personal injury",
                "real estate",
                "intellectual property",
                "immigration",
                "employment law",
                "tax law",
                "bankruptcy",
                "estate planning",
                "mergers",
                "acquisitions",
                "compliance",
                "regulatory",
                # Skills and activities
                "legal research",
                "document review",
                "discovery",
                "depositions",
                "motions",
                "briefs",
                "pleadings",
                "negotiations",
                "mediation",
                "arbitration",
                "trial",
                "legal writing",
                "case management",
                "client counseling",
                "due diligence",
            ],
            IndustryType.HOSPITALITY_FOOD: [
                # Direct terms
                "restaurant",
                "hotel",
                "hospitality",
                "food",
                "service",
                "server",
                "waiter",
                "waitress",
                "bartender",
                "chef",
                "cook",
                "kitchen",
                "dining",
                "catering",
                "banquet",
                "event",
                "tourism",
                "travel",
                "resort",
                "lodging",
                "concierge",
                # Skills and activities
                "customer service",
                "food safety",
                "menu",
                "orders",
                "pos",
                "cash handling",
                "food preparation",
                "cooking",
                "baking",
                "culinary",
                "wine",
                "beverage",
                "cleaning",
                "sanitation",
                "inventory",
                "scheduling",
                "reservations",
                # Work environments
                "fine dining",
                "fast food",
                "casual dining",
                "cafe",
                "bar",
                "pub",
                "nightclub",
                "hotel",
                "motel",
                "inn",
                "bed and breakfast",
                "cruise",
                "airline",
                "casino",
            ],
        }

    def _build_role_keyword_index(self) -> Dict[str, List[str]]:
        """Build keyword index for specific roles"""

        role_keywords = {}

        for industry_key, industry_data in self.industry_data.items():
            for role_key, role_data in industry_data.get("common_roles", {}).items():
                # Combine all keywords for this role
                keywords = []

                # Add role name variations
                role_name = role_key.replace("_", " ")
                keywords.append(role_name)
                keywords.extend(role_name.split())

                # Add from critical skills
                keywords.extend(role_data.get("critical_skills", []))

                # Add from experience level titles
                for level_data in role_data.get("experience_levels", {}).values():
                    keywords.extend(level_data.get("typical_titles", []))

                # Add specializations
                keywords.extend(role_data.get("specializations", []))

                role_keywords[f"{industry_key}_{role_key}"] = [
                    kw.lower() for kw in keywords
                ]

        return role_keywords

    def detect_industry_from_text(
        self, text: str, job_title: str = ""
    ) -> IndustryDetectionResult:
        """
        Detect industry from combined text (job description + title)
        """

        # Combine and normalize text
        combined_text = f"{job_title} {text}".lower()

        # Remove common noise words
        noise_words = ["the", "and", "or", "of", "in", "at", "to", "for", "with", "by"]
        words = [word for word in combined_text.split() if word not in noise_words]
        combined_clean = " ".join(words)

        # Score each industry
        industry_scores = {}
        reasoning = []

        for industry, keywords in self.industry_keywords.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                # Exact phrase matching
                if keyword.lower() in combined_clean:
                    score += 2
                    matched_keywords.append(keyword)

                # Individual word matching
                elif any(word in combined_clean for word in keyword.split()):
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                industry_scores[industry] = score
                reasoning.append(
                    f"{industry.value}: {score} points ({', '.join(matched_keywords[:5])})"
                )

        # Handle case where no industry is detected
        if not industry_scores:
            return IndustryDetectionResult(
                primary_industry=IndustryType.UNKNOWN,
                confidence_score=0.0,
                detected_role=None,
                alternative_industries=[],
                reasoning=["No clear industry indicators found"],
            )

        # Sort by score
        sorted_industries = sorted(
            industry_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Calculate confidence (normalized score)
        max_score = sorted_industries[0][1]
        total_score = sum(industry_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0

        # Detect specific role
        detected_role = self._detect_specific_role(
            combined_clean, sorted_industries[0][0]
        )

        # Build alternative industries
        alternatives = [
            (industry, score / total_score)
            for industry, score in sorted_industries[1:4]
        ]

        return IndustryDetectionResult(
            primary_industry=sorted_industries[0][0],
            confidence_score=confidence,
            detected_role=detected_role,
            alternative_industries=alternatives,
            reasoning=reasoning,
        )

    def _detect_specific_role(self, text: str, industry: IndustryType) -> Optional[str]:
        """Detect specific role within an industry"""

        industry_key = industry.value
        if industry_key not in self.industry_data:
            return None

        role_scores = {}

        # Check each role in the detected industry
        for role_key, role_data in (
            self.industry_data[industry_key].get("common_roles", {}).items()
        ):
            score = 0

            # Check role name
            role_name = role_key.replace("_", " ")
            if role_name in text:
                score += 10

            # Check typical titles
            for level_data in role_data.get("experience_levels", {}).values():
                for title in level_data.get("typical_titles", []):
                    if title.lower() in text:
                        score += 5

            # Check specializations
            for spec in role_data.get("specializations", []):
                if spec.lower() in text:
                    score += 3

            if score > 0:
                role_scores[role_key] = score

        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            return best_role.replace("_", " ").title()

        return None

    def extract_experience_level(self, text: str) -> Tuple[str, List[str]]:
        """Extract experience level from text"""

        text_lower = text.lower()
        level_indicators = []

        for level, level_data in UNIVERSAL_EXPERIENCE_LEVELS.items():
            for keyword in level_data["keywords"]:
                if keyword in text_lower:
                    level_indicators.append((level, keyword))

        # Extract years of experience
        year_patterns = [
            r"(\d+)\s*(?:\+|plus)?\s*years?\s*(?:of\s*)?experience",
            r"(\d+)\s*(?:\+|plus)?\s*years?\s*(?:in|with)",
            r"minimum\s*(\d+)\s*years?",
            r"at least\s*(\d+)\s*years?",
        ]

        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            years_found.extend([int(match) for match in matches])

        # Determine primary level
        if level_indicators:
            # Use most specific indicator
            priority_order = [
                "management_level",
                "senior_level",
                "mid_level",
                "entry_level",
            ]
            for level in priority_order:
                if any(indicator[0] == level for indicator in level_indicators):
                    reasoning = [
                        f"Found: {indicator[1]}"
                        for indicator in level_indicators
                        if indicator[0] == level
                    ]
                    return level.replace("_", " ").title(), reasoning

        # Fallback to years of experience
        if years_found:
            max_years = max(years_found)
            if max_years >= 8:
                return "Senior Level", [f"{max_years} years experience"]
            elif max_years >= 3:
                return "Mid Level", [f"{max_years} years experience"]
            else:
                return "Entry Level", [f"{max_years} years experience"]

        return "Mid Level", ["No clear experience indicators - defaulting to mid-level"]

    def extract_skills_from_text(
        self, text: str, industry: IndustryType = None
    ) -> Dict[str, List[str]]:
        """Extract skills from text, optionally filtered by industry"""

        text_lower = text.lower()
        found_skills = {
            "technical_skills": [],
            "soft_skills": [],
            "certifications": [],
            "software_tools": [],
        }

        # If industry is specified, get industry-specific skills
        if industry and industry.value in self.industry_data:
            industry_data = self.industry_data[industry.value]

            # Extract from all roles in the industry
            for role_data in industry_data.get("common_roles", {}).values():
                # Technical skills
                for skill in role_data.get("critical_skills", []):
                    if skill.lower() in text_lower:
                        found_skills["technical_skills"].append(skill)

                # Soft skills
                for skill in role_data.get("soft_skills", []):
                    if skill.lower() in text_lower:
                        found_skills["soft_skills"].append(skill)

                # Certifications
                for cert in role_data.get(
                    "critical_certifications", []
                ) + role_data.get("optional_certifications", []):
                    if cert.lower() in text_lower:
                        found_skills["certifications"].append(cert)

                # Technology skills
                for tech in role_data.get("technology_skills", []):
                    if tech.lower() in text_lower:
                        found_skills["software_tools"].append(tech)

        # Extract transferable skills
        for skill, skill_data in UNIVERSAL_TRANSFERABLE_SKILLS.items():
            for variation in skill_data["variations"]:
                if variation.lower() in text_lower:
                    found_skills["soft_skills"].append(variation)

        # Remove duplicates
        for category in found_skills:
            found_skills[category] = list(set(found_skills[category]))

        return found_skills

    def get_industry_requirements(
        self, industry: IndustryType, role: str = None
    ) -> Dict[str, List[str]]:
        """Get requirements for an industry/role combination"""

        if industry.value not in self.industry_data:
            return {}

        industry_data = self.industry_data[industry.value]

        if role:
            # Find specific role
            role_key = role.lower().replace(" ", "_")
            if role_key in industry_data.get("common_roles", {}):
                role_data = industry_data["common_roles"][role_key]
                return {
                    "required_education": role_data.get("required_education", []),
                    "required_licenses": role_data.get("required_licenses", []),
                    "critical_skills": role_data.get("critical_skills", []),
                    "certifications": role_data.get("critical_certifications", []),
                    "technology_skills": role_data.get("technology_skills", []),
                }

        # Return general industry requirements
        all_requirements = {
            "required_education": [],
            "required_licenses": [],
            "critical_skills": [],
            "certifications": [],
            "technology_skills": [],
        }

        # Aggregate from all roles
        for role_data in industry_data.get("common_roles", {}).values():
            for req_type in all_requirements:
                all_requirements[req_type].extend(role_data.get(req_type, []))

        # Remove duplicates
        for req_type in all_requirements:
            all_requirements[req_type] = list(set(all_requirements[req_type]))

        return all_requirements

    def analyze_job_posting(
        self, job_title: str, job_description: str
    ) -> Dict[str, any]:
        """Complete analysis of a job posting"""

        # Detect industry
        industry_result = self.detect_industry_from_text(job_description, job_title)

        # Extract experience level
        experience_level, exp_reasoning = self.extract_experience_level(job_description)

        # Extract skills mentioned
        skills = self.extract_skills_from_text(
            job_description, industry_result.primary_industry
        )

        # Get industry requirements
        requirements = self.get_industry_requirements(
            industry_result.primary_industry, industry_result.detected_role
        )

        return {
            "job_title": job_title,
            "industry_detection": {
                "primary_industry": industry_result.primary_industry.value,
                "confidence": industry_result.confidence_score,
                "detected_role": industry_result.detected_role,
                "reasoning": industry_result.reasoning,
            },
            "experience_level": {"level": experience_level, "reasoning": exp_reasoning},
            "skills_mentioned": skills,
            "industry_requirements": requirements,
            "analysis_summary": f"Detected as {industry_result.primary_industry.value.replace('_', ' ').title()} role with {experience_level.lower()} requirements",
        }
