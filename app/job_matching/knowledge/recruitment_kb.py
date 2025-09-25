# app/job_matching/knowledge/recruitment_kb.py - FIXED role normalization
"""
Lightweight Recruitment Knowledge Base without external vector dependencies.
Uses built-in Python features for semantic matching and knowledge retrieval.
FIXED: Proper role normalization for Python developers.
"""

import json
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib

from .knowledge_data import RECRUITMENT_KNOWLEDGE, HIRING_INSIGHTS


class RecruitmentKnowledgeBase:
    """
    Lightweight knowledge base for recruitment insights using built-in Python features.
    Provides domain-specific knowledge without external vector database dependencies.
    """

    def __init__(self):
        self.knowledge_data = RECRUITMENT_KNOWLEDGE
        self.insights = HIRING_INSIGHTS
        self.knowledge_index = self._build_knowledge_index()
        print("✅ RAG Knowledge Base initialized successfully (Lightweight Mode)")

    def _build_knowledge_index(self) -> Dict[str, List[Dict]]:
        """Build searchable index of knowledge without external dependencies"""
        index = defaultdict(list)

        # Index salary data
        for role, levels in self.knowledge_data["salary_benchmarks"].items():
            for level, data in levels.items():
                entry = {
                    "type": "salary",
                    "role": role,
                    "level": level,
                    "data": data,
                    "searchable_text": f"{role} {level} salary benchmark ${data['avg']}",
                }
                index["salary"].append(entry)
                index["all"].append(entry)

        # Index skill data
        for category, skills in self.knowledge_data["skill_demand"].items():
            for skill, data in skills.items():
                entry = {
                    "type": "skill",
                    "skill": skill,
                    "category": category,
                    "data": data,
                    "searchable_text": f"{skill} {category} {data['demand']} demand {data['growth']} growth",
                }
                index["skills"].append(entry)
                index["all"].append(entry)

        # Index ATS keywords
        for role, keywords in self.knowledge_data["ats_optimization"][
            "critical_keywords"
        ].items():
            entry = {
                "type": "ats",
                "role": role,
                "keywords": keywords,
                "searchable_text": f"{role} ATS keywords {' '.join(keywords)}",
            }
            index["ats"].append(entry)
            index["all"].append(entry)

        # Index insights
        for i, insight in enumerate(self.insights):
            entry = {
                "type": "insight",
                "id": i,
                "text": insight,
                "searchable_text": insight,
            }
            index["insights"].append(entry)
            index["all"].append(entry)

        return dict(index)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using built-in difflib"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _search_knowledge(
        self, query: str, knowledge_type: str = "all", top_k: int = 5
    ) -> List[Dict]:
        """Search knowledge base using text similarity"""
        query_lower = query.lower()
        results = []

        for entry in self.knowledge_index.get(knowledge_type, []):
            similarity = self._calculate_similarity(
                query_lower, entry["searchable_text"]
            )
            if similarity > 0.1:  # Basic threshold
                results.append({"entry": entry, "similarity": similarity})

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return [r["entry"] for r in results[:top_k]]

    def query_knowledge(self, query: str, k: int = 5) -> List[str]:
        """Query the knowledge base for relevant information"""
        results = self._search_knowledge(query, "all", k)
        formatted_results = []

        for result in results:
            if result["type"] == "salary":
                text = f"Salary Info - {result['role']} ({result['level']}): ${result['data']['avg']:,} average"
            elif result["type"] == "skill":
                text = f"Skill Analysis - {result['skill']}: {result['data']['demand']} demand, {result['data']['growth']} growth"
            elif result["type"] == "ats":
                text = f"ATS Keywords for {result['role']}: {', '.join(result['keywords'][:5])}"
            elif result["type"] == "insight":
                text = f"Market Insight: {result['text']}"
            else:
                text = str(result)

            formatted_results.append(text)

        return formatted_results

    def get_job_analysis_context(self, job_title: str, skills: List[str] = None) -> str:
        """Get comprehensive context for job analysis"""
        context_parts = []

        # Get role-specific information
        role_query = f"job role {job_title} requirements salary"
        role_results = self._search_knowledge(role_query, "all", 3)

        context_parts.append("=== ROLE-SPECIFIC KNOWLEDGE ===")
        for result in role_results:
            if result["type"] == "salary":
                context_parts.append(
                    f"Salary Benchmark: {result['role']} ({result['level']}) - ${result['data']['avg']:,} average"
                )
            elif result["type"] == "ats":
                context_parts.append(
                    f"ATS Keywords: {', '.join(result['keywords'][:8])}"
                )

        # Get skills analysis if provided
        if skills:
            context_parts.append("\n=== SKILLS MARKET ANALYSIS ===")
            for skill in skills[:5]:  # Limit to avoid too much data
                skill_results = self._search_knowledge(f"skill {skill}", "skills", 1)
                for result in skill_results:
                    if (
                        result["skill"].lower() == skill.lower()
                        or skill.lower() in result["skill"].lower()
                    ):
                        context_parts.append(
                            f"{result['skill']}: {result['data']['demand']} demand, "
                            f"{result['data']['growth']} growth, {result['data'].get('salary_impact', 'unknown')} salary impact"
                        )
                        break

        # Get hiring insights
        context_parts.append("\n=== HIRING INSIGHTS ===")
        insights_results = self._search_knowledge(f"hiring {job_title}", "insights", 2)
        for result in insights_results:
            context_parts.append(f"• {result['text']}")

        return "\n".join(context_parts)

    def get_salary_benchmark(self, role: str, level: str = "mid") -> Dict[str, Any]:
        """Get specific salary information for a role"""
        role_normalized = self._normalize_role_name(role)

        # Search in salary data
        for entry in self.knowledge_index["salary"]:
            if entry["role"] == role_normalized and entry["level"] == level:
                data = entry["data"]
                return {
                    "role": role,
                    "level": level,
                    "salary_range": f"${data['min']:,} - ${data['max']:,}",
                    "average": f"${data['avg']:,}",
                    "min": data["min"],
                    "max": data["max"],
                    "avg": data["avg"],
                }

        # If exact match not found, try fuzzy matching
        for entry in self.knowledge_index["salary"]:
            if self._calculate_similarity(role_normalized, entry["role"]) > 0.6:
                data = entry["data"]
                return {
                    "role": role,
                    "level": level,
                    "salary_range": f"${data['min']:,} - ${data['max']:,}",
                    "average": f"${data['avg']:,}",
                    "min": data["min"],
                    "max": data["max"],
                    "avg": data["avg"],
                    "note": f"Based on similar role: {entry['role']}",
                }

        return {"role": role, "level": level, "message": "Salary data not available"}

    def get_skill_insights(self, skills: List[str]) -> List[Dict[str, Any]]:
        """Get detailed insights about specific skills"""
        insights = []

        for skill in skills:
            skill_lower = skill.lower()

            # Search for exact or similar skill matches
            for entry in self.knowledge_index["skills"]:
                entry_skill = entry["skill"].lower()
                if (
                    skill_lower == entry_skill
                    or skill_lower in entry_skill
                    or entry_skill in skill_lower
                    or self._calculate_similarity(skill_lower, entry_skill) > 0.7
                ):
                    insights.append(
                        {
                            "skill": skill,
                            "matched_skill": entry["skill"],
                            "category": entry["category"],
                            "demand": entry["data"]["demand"],
                            "growth": entry["data"]["growth"],
                            "difficulty": entry["data"]["difficulty"],
                            "salary_impact": entry["data"].get(
                                "salary_impact", "Unknown"
                            ),
                            "insight": f"{skill} shows {entry['data']['demand']} demand with {entry['data']['growth']} growth",
                        }
                    )
                    break  # Take first good match

        return insights

    def get_ats_keywords(self, role: str) -> List[str]:
        """Get critical ATS keywords for a role"""
        role_normalized = self._normalize_role_name(role)

        # Look for exact match first
        for entry in self.knowledge_index["ats"]:
            if entry["role"] == role_normalized:
                return entry["keywords"]

        # Try fuzzy matching
        for entry in self.knowledge_index["ats"]:
            if self._calculate_similarity(role_normalized, entry["role"]) > 0.6:
                return entry["keywords"]

        # Return general keywords if no match
        return [
            "professional experience",
            "team collaboration",
            "problem solving",
            "communication",
        ]

    def analyze_market_fit(
        self, role: str, skills: List[str], experience_level: str = "mid"
    ) -> Dict[str, Any]:
        """Comprehensive market fit analysis"""
        # Get salary data
        salary_info = self.get_salary_benchmark(role, experience_level)

        # Analyze skills
        skill_insights = self.get_skill_insights(skills)

        # Get ATS keywords
        ats_keywords = self.get_ats_keywords(role)

        # Calculate skill match score
        high_demand_skills = [
            s for s in skill_insights if s.get("demand") in ["high", "very_high"]
        ]
        skill_match_score = min(
            100, (len(high_demand_skills) / max(1, len(skills))) * 100
        )

        return {
            "role": role,
            "salary_benchmark": salary_info,
            "skill_analysis": skill_insights,
            "ats_keywords": ats_keywords,
            "skill_match_score": skill_match_score,
            "high_demand_skills": [s["skill"] for s in high_demand_skills],
            "market_summary": f"Found {len(high_demand_skills)} high-demand skills out of {len(skills)} total skills",
        }

    def _normalize_role_name(self, role: str) -> str:
        """FIXED: Normalize role name for consistent lookup"""
        normalized = re.sub(r"[^\w\s]", "", role.lower())
        normalized = re.sub(r"\s+", "_", normalized.strip())

        # FIXED: Handle common variations with proper Python role mapping
        role_mappings = {
            # General software development
            "software_developer": "software_engineer",
            "programmer": "software_engineer",
            "full_stack_developer": "software_engineer",
            # Frontend roles
            "web_developer": "frontend_developer",
            "ui_developer": "frontend_developer",
            "frontend_engineer": "frontend_developer",
            # Backend roles (FIXED: Added Python mappings)
            "api_developer": "backend_developer",
            "server_developer": "backend_developer",
            "backend_engineer": "backend_developer",
            # Python-specific mappings (CRITICAL FIX)
            "python_developer": "python_developer",  # Keep as-is
            "python_engineer": "python_developer",
            "django_developer": "python_developer",
            "flask_developer": "python_developer",
            "fastapi_developer": "python_developer",
            "py_developer": "python_developer",
            # Data roles
            "data_analyst": "data_scientist",
            "ml_engineer": "data_scientist",
            "machine_learning_engineer": "data_scientist",
            # Product roles
            "product_owner": "product_manager",
            "pm": "product_manager",
        }

        # First try direct mapping
        if normalized in role_mappings:
            return role_mappings[normalized]

        # If no direct mapping, check if it contains "python"
        if "python" in normalized:
            return "python_developer"

        # Check if it contains "backend"
        if "backend" in normalized:
            return "backend_developer"

        # Check if it contains "frontend"
        if "frontend" in normalized or "front_end" in normalized:
            return "frontend_developer"

        return normalized

    def get_red_flags(self) -> List[str]:
        """Get list of resume red flags"""
        return self.knowledge_data["ats_optimization"]["red_flags"]

    def get_green_flags(self) -> List[str]:
        """Get list of resume positive indicators"""
        return self.knowledge_data["ats_optimization"]["green_flags"]

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of available knowledge"""
        return {
            "roles_covered": list(self.knowledge_data["salary_benchmarks"].keys()),
            "skill_categories": list(self.knowledge_data["skill_demand"].keys()),
            "total_skills": sum(
                len(skills) for skills in self.knowledge_data["skill_demand"].values()
            ),
            "insights_count": len(self.insights),
            "implementation": "Lightweight (No external dependencies)",
            "features": [
                "Salary benchmarks",
                "Skill analysis",
                "ATS optimization",
                "Market insights",
                "Python-specific knowledge",
                "Backend developer expertise",
            ],
            "last_updated": "2024",
        }
