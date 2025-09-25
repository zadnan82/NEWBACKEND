# app/job_matching/knowledge/universal_knowledge_data.py
"""
Universal Knowledge Base for All Industries - Skills & Qualifications Focus
No salary data - focuses on what actually matters for job matching
"""

UNIVERSAL_INDUSTRY_CLASSIFICATION = {
    "healthcare_medical": {
        "sector_description": "Healthcare and medical services",
        "growth_outlook": "Much faster than average",
        "common_roles": {
            "registered_nurse": {
                "experience_levels": {
                    "new_graduate": {
                        "years_experience": "0-1",
                        "typical_titles": ["Graduate Nurse", "New Grad RN", "RN I"],
                        "expectations": "Supervised practice, basic patient care, learning protocols",
                    },
                    "staff_nurse": {
                        "years_experience": "1-5",
                        "typical_titles": ["Staff Nurse", "RN II", "Registered Nurse"],
                        "expectations": "Independent patient care, medication administration, patient assessment",
                    },
                    "charge_nurse": {
                        "years_experience": "5-10",
                        "typical_titles": ["Charge Nurse", "Team Leader", "RN III"],
                        "expectations": "Leadership, staff supervision, complex patient care",
                    },
                    "nurse_manager": {
                        "years_experience": "8+",
                        "typical_titles": [
                            "Nurse Manager",
                            "Unit Manager",
                            "Nursing Supervisor",
                        ],
                        "expectations": "Unit management, budget oversight, staff development",
                    },
                },
                "specializations": [
                    "ICU",
                    "Emergency",
                    "Pediatric",
                    "Surgical",
                    "Mental Health",
                    "Oncology",
                    "Labor & Delivery",
                ],
                "required_education": [
                    "Associate Degree in Nursing (ADN)",
                    "Bachelor of Science in Nursing (BSN)",
                ],
                "required_licenses": ["RN License", "State Nursing License"],
                "critical_certifications": [
                    "BLS (Basic Life Support)",
                    "ACLS (Advanced Cardiac Life Support)",
                    "PALS (Pediatric Advanced Life Support)",
                ],
                "optional_certifications": [
                    "CCRN (Critical Care)",
                    "CEN (Emergency Nursing)",
                    "CNOR (Operating Room)",
                ],
                "critical_skills": [
                    "Patient care and assessment",
                    "Medication administration",
                    "IV therapy",
                    "Wound care",
                    "Vital signs monitoring",
                    "Medical documentation",
                    "Infection control",
                    "Patient safety protocols",
                    "Emergency response",
                    "Healthcare team collaboration",
                    "Patient education",
                    "Medical terminology",
                    "Clinical decision making",
                    "Time management",
                    "Stress management",
                ],
                "soft_skills": [
                    "Compassion",
                    "Communication",
                    "Critical thinking",
                    "Attention to detail",
                    "Emotional resilience",
                    "Teamwork",
                    "Adaptability",
                    "Cultural sensitivity",
                ],
                "technology_skills": [
                    "Electronic Health Records (EHR)",
                    "Medical equipment operation",
                    "IV pumps",
                    "Patient monitoring systems",
                    "Electronic medication administration",
                ],
                "career_progression": [
                    "New Graduate RN → Staff Nurse → Charge Nurse → Nurse Manager",
                    "Specialization path: Staff Nurse → Specialized Certification → Advanced Practice",
                    "Education path: RN → BSN → MSN → Nurse Practitioner",
                ],
            },
            "medical_assistant": {
                "experience_levels": {
                    "entry_level": {
                        "years_experience": "0-2",
                        "expectations": "Basic clinical and administrative tasks under supervision",
                    },
                    "experienced": {
                        "years_experience": "2-5",
                        "expectations": "Independent clinical tasks, patient care coordination",
                    },
                },
                "required_education": [
                    "Medical Assistant Certificate",
                    "Associate Degree in Medical Assisting",
                ],
                "required_licenses": ["State certification (where required)"],
                "critical_skills": [
                    "Patient intake",
                    "Vital signs",
                    "Medical records management",
                    "Appointment scheduling",
                    "Insurance verification",
                    "Medical terminology",
                    "HIPAA compliance",
                    "Basic clinical procedures",
                    "Laboratory specimen collection",
                ],
                "technology_skills": [
                    "EHR systems",
                    "Medical billing software",
                    "Office equipment",
                ],
            },
            "physician": {
                "experience_levels": {
                    "resident": {
                        "years_experience": "0-7 (post-medical school)",
                        "expectations": "Supervised patient care, learning specialization",
                    },
                    "attending": {
                        "years_experience": "7+",
                        "expectations": "Independent practice, patient management, teaching",
                    },
                },
                "required_education": [
                    "Medical Degree (MD or DO)",
                    "Completed Residency",
                ],
                "required_licenses": ["Medical License", "DEA Registration"],
                "specializations": [
                    "Internal Medicine",
                    "Surgery",
                    "Cardiology",
                    "Neurology",
                    "Pediatrics",
                    "Psychiatry",
                ],
                "critical_skills": [
                    "Medical diagnosis",
                    "Patient treatment planning",
                    "Clinical decision making",
                    "Medical procedures",
                    "Patient consultation",
                    "Medical research",
                    "Treatment protocols",
                    "Medical documentation",
                ],
            },
        },
    },
    "education": {
        "sector_description": "Education and training services",
        "growth_outlook": "Average growth",
        "common_roles": {
            "elementary_teacher": {
                "experience_levels": {
                    "first_year": {
                        "years_experience": "0-1",
                        "expectations": "Classroom setup, basic lesson delivery, mentor support",
                    },
                    "experienced": {
                        "years_experience": "3-10",
                        "expectations": "Independent teaching, curriculum development, student assessment",
                    },
                    "master_teacher": {
                        "years_experience": "10+",
                        "expectations": "Mentoring new teachers, curriculum leadership, advanced instructional strategies",
                    },
                },
                "grade_levels": [
                    "Kindergarten",
                    "1st Grade",
                    "2nd Grade",
                    "3rd Grade",
                    "4th Grade",
                    "5th Grade",
                ],
                "required_education": [
                    "Bachelor's Degree in Education",
                    "Bachelor's in subject area + teaching credential",
                ],
                "required_licenses": [
                    "State Teaching License",
                    "Elementary Education Credential",
                ],
                "critical_skills": [
                    "Classroom management",
                    "Lesson planning",
                    "Curriculum development",
                    "Student assessment",
                    "Differentiated instruction",
                    "Parent communication",
                    "Behavior management",
                    "Learning objectives",
                    "Educational technology",
                    "Special needs accommodation",
                    "Student engagement",
                ],
                "technology_skills": [
                    "Learning Management Systems",
                    "Educational software",
                    "Interactive whiteboards",
                    "Student information systems",
                    "Online assessment tools",
                ],
                "certifications": [
                    "ESL Certification",
                    "Special Education Endorsement",
                    "Reading Specialist",
                ],
            },
            "high_school_teacher": {
                "experience_levels": {
                    "first_year": {
                        "years_experience": "0-1",
                        "expectations": "Subject delivery, classroom management basics",
                    },
                    "experienced": {
                        "years_experience": "3-10",
                        "expectations": "Advanced pedagogy, student mentoring, department collaboration",
                    },
                },
                "subjects": [
                    "Mathematics",
                    "Science",
                    "English",
                    "History",
                    "Art",
                    "Physical Education",
                    "Foreign Languages",
                ],
                "required_education": [
                    "Bachelor's in subject area",
                    "Master's preferred",
                ],
                "required_licenses": [
                    "State Teaching License",
                    "Subject Area Credential",
                ],
                "critical_skills": [
                    "Subject matter expertise",
                    "Curriculum alignment",
                    "Standardized test preparation",
                    "Classroom management",
                    "Student motivation",
                    "College preparation",
                    "Lesson planning",
                    "Assessment design",
                    "Technology integration",
                ],
            },
        },
    },
    "finance_banking": {
        "sector_description": "Financial services and banking",
        "growth_outlook": "Slower than average (automation impact)",
        "common_roles": {
            "accountant": {
                "experience_levels": {
                    "staff": {
                        "years_experience": "0-3",
                        "expectations": "Basic accounting tasks, journal entries, reconciliations",
                    },
                    "senior": {
                        "years_experience": "3-7",
                        "expectations": "Complex accounting, supervision, financial analysis",
                    },
                    "manager": {
                        "years_experience": "7+",
                        "expectations": "Team management, strategic financial planning",
                    },
                },
                "specializations": [
                    "Tax Accounting",
                    "Audit",
                    "Management Accounting",
                    "Forensic Accounting",
                ],
                "required_education": [
                    "Bachelor's in Accounting",
                    "Bachelor's in Finance",
                ],
                "certifications": [
                    "CPA (Certified Public Accountant)",
                    "CMA (Certified Management Accountant)",
                    "CIA (Certified Internal Auditor)",
                ],
                "critical_skills": [
                    "Financial statement preparation",
                    "Tax preparation",
                    "Audit procedures",
                    "GAAP knowledge",
                    "Financial analysis",
                    "Budgeting",
                    "Cost accounting",
                    "Regulatory compliance",
                    "Risk assessment",
                    "Financial reporting",
                ],
                "technology_skills": [
                    "QuickBooks",
                    "Excel",
                    "SAP",
                    "Oracle",
                    "Tax software",
                    "Financial modeling",
                    "Database management",
                ],
                "soft_skills": [
                    "Attention to detail",
                    "Analytical thinking",
                    "Problem solving",
                    "Communication",
                    "Ethics",
                    "Time management",
                ],
            },
            "financial_analyst": {
                "experience_levels": {
                    "analyst": {
                        "years_experience": "0-3",
                        "expectations": "Financial modeling, data analysis, report preparation",
                    },
                    "senior_analyst": {
                        "years_experience": "3-6",
                        "expectations": "Complex analysis, client interaction, project leadership",
                    },
                },
                "required_education": [
                    "Bachelor's in Finance",
                    "Bachelor's in Economics",
                    "MBA preferred",
                ],
                "certifications": [
                    "CFA (Chartered Financial Analyst)",
                    "FRM (Financial Risk Manager)",
                ],
                "critical_skills": [
                    "Financial modeling",
                    "Investment analysis",
                    "Risk assessment",
                    "Market research",
                    "Data analysis",
                    "Financial forecasting",
                    "Valuation techniques",
                    "Portfolio management",
                ],
            },
        },
    },
    "technology": {
        "sector_description": "Information technology and software",
        "growth_outlook": "Much faster than average",
        "common_roles": {
            "software_engineer": {
                "experience_levels": {
                    "junior": {
                        "years_experience": "0-2",
                        "expectations": "Code implementation, debugging, learning frameworks",
                    },
                    "mid_level": {
                        "years_experience": "2-5",
                        "expectations": "Independent development, code reviews, mentoring juniors",
                    },
                    "senior": {
                        "years_experience": "5-8",
                        "expectations": "Architecture decisions, technical leadership, complex problem solving",
                    },
                    "lead": {
                        "years_experience": "8+",
                        "expectations": "Team leadership, strategic planning, technology decisions",
                    },
                },
                "specializations": [
                    "Frontend",
                    "Backend",
                    "Full-Stack",
                    "Mobile",
                    "DevOps",
                    "Machine Learning",
                ],
                "required_education": [
                    "Bachelor's in Computer Science",
                    "Related technical degree",
                    "Bootcamp + portfolio",
                ],
                "critical_skills": [
                    "Programming languages",
                    "Problem solving",
                    "Debugging",
                    "Testing",
                    "Version control",
                    "Database design",
                    "API development",
                    "Software architecture",
                    "Code review",
                    "Documentation",
                ],
                "technology_skills": [
                    "Git",
                    "Linux",
                    "Docker",
                    "Cloud platforms",
                    "CI/CD",
                    "Agile methodologies",
                    "Database systems",
                ],
                "programming_languages": [
                    "Python",
                    "JavaScript",
                    "Java",
                    "C++",
                    "Go",
                    "Rust",
                ],
                "frameworks": ["React", "Angular", "Django", "Spring", "Node.js"],
            },
            "data_scientist": {
                "experience_levels": {
                    "junior": {
                        "years_experience": "0-2",
                        "expectations": "Data analysis, model implementation, visualization",
                    },
                    "senior": {
                        "years_experience": "3-7",
                        "expectations": "Complex modeling, business insights, project leadership",
                    },
                },
                "required_education": [
                    "Master's in Data Science",
                    "PhD in quantitative field",
                    "Bachelor's + experience",
                ],
                "critical_skills": [
                    "Statistical analysis",
                    "Machine learning",
                    "Data visualization",
                    "Programming",
                    "Business acumen",
                    "Communication",
                    "Experimental design",
                    "Data cleaning",
                ],
                "technology_skills": [
                    "Python",
                    "R",
                    "SQL",
                    "Tableau",
                    "TensorFlow",
                    "PyTorch",
                ],
            },
        },
    },
    "retail_sales": {
        "sector_description": "Retail trade and customer service",
        "growth_outlook": "Little to no change",
        "common_roles": {
            "sales_associate": {
                "experience_levels": {
                    "entry": {
                        "years_experience": "0-1",
                        "expectations": "Customer assistance, basic product knowledge, POS operation",
                    },
                    "experienced": {
                        "years_experience": "1-5",
                        "expectations": "Advanced sales techniques, customer relationship building",
                    },
                    "team_lead": {
                        "years_experience": "3+",
                        "expectations": "Team supervision, training, inventory management",
                    },
                },
                "work_environments": [
                    "Retail stores",
                    "Department stores",
                    "Specialty shops",
                    "Online retail",
                ],
                "critical_skills": [
                    "Customer service",
                    "Sales techniques",
                    "Product knowledge",
                    "Cash handling",
                    "POS systems",
                    "Inventory management",
                    "Visual merchandising",
                    "Problem solving",
                    "Communication",
                ],
                "soft_skills": [
                    "Patience",
                    "Enthusiasm",
                    "Teamwork",
                    "Adaptability",
                    "Time management",
                    "Stress management",
                ],
            }
        },
    },
    "manufacturing": {
        "sector_description": "Manufacturing and production",
        "growth_outlook": "Declining (automation)",
        "common_roles": {
            "production_worker": {
                "experience_levels": {
                    "entry": {
                        "years_experience": "0-1",
                        "expectations": "Basic assembly, safety protocol adherence",
                    },
                    "experienced": {
                        "years_experience": "2-8",
                        "expectations": "Complex assembly, quality control, equipment operation",
                    },
                    "supervisor": {
                        "years_experience": "5+",
                        "expectations": "Team leadership, production planning, safety management",
                    },
                },
                "work_types": [
                    "Assembly",
                    "Machine operation",
                    "Quality control",
                    "Packaging",
                ],
                "critical_skills": [
                    "Safety protocols",
                    "Quality control",
                    "Equipment operation",
                    "Assembly techniques",
                    "Problem solving",
                    "Attention to detail",
                    "Physical stamina",
                    "Team collaboration",
                ],
                "certifications": [
                    "Safety training",
                    "Equipment certification",
                    "Quality standards",
                ],
            }
        },
    },
    "legal": {
        "sector_description": "Legal services",
        "growth_outlook": "Average growth",
        "common_roles": {
            "paralegal": {
                "experience_levels": {
                    "entry": {
                        "years_experience": "0-2",
                        "expectations": "Document preparation, basic research, file management",
                    },
                    "experienced": {
                        "years_experience": "3-8",
                        "expectations": "Complex research, client interaction, case management",
                    },
                },
                "specializations": [
                    "Litigation",
                    "Corporate law",
                    "Family law",
                    "Criminal law",
                    "Real estate",
                ],
                "required_education": [
                    "Paralegal certificate",
                    "Associate degree in paralegal studies",
                ],
                "critical_skills": [
                    "Legal research",
                    "Document preparation",
                    "Case management",
                    "Client communication",
                    "Court filings",
                    "Legal software",
                    "Legal writing",
                    "Confidentiality",
                    "Attention to detail",
                ],
                "technology_skills": [
                    "Legal databases",
                    "Case management software",
                    "Document review platforms",
                ],
            }
        },
    },
    "hospitality_food": {
        "sector_description": "Hospitality and food service",
        "growth_outlook": "Average growth",
        "common_roles": {
            "server": {
                "experience_levels": {
                    "entry": {
                        "years_experience": "0-1",
                        "expectations": "Order taking, basic service, learning menu",
                    },
                    "experienced": {
                        "years_experience": "2+",
                        "expectations": "Advanced service, wine knowledge, customer relationship building",
                    },
                },
                "work_environments": ["Restaurants", "Hotels", "Catering", "Events"],
                "critical_skills": [
                    "Customer service",
                    "Multitasking",
                    "Menu knowledge",
                    "POS systems",
                    "Food safety",
                    "Communication",
                    "Physical stamina",
                    "Stress management",
                ],
                "certifications": [
                    "Food safety certification",
                    "Alcohol service permit",
                ],
            }
        },
    },
}

# Universal transferable skills that apply across industries
UNIVERSAL_TRANSFERABLE_SKILLS = {
    "communication": {
        "variations": [
            "verbal communication",
            "written communication",
            "presentation skills",
            "public speaking",
        ],
        "relevant_industries": ["all"],
    },
    "leadership": {
        "variations": [
            "team leadership",
            "project management",
            "supervision",
            "mentoring",
        ],
        "relevant_industries": ["all"],
    },
    "problem_solving": {
        "variations": [
            "analytical thinking",
            "troubleshooting",
            "critical thinking",
            "decision making",
        ],
        "relevant_industries": ["all"],
    },
    "customer_service": {
        "variations": [
            "client relations",
            "customer support",
            "patient care",
            "client management",
        ],
        "relevant_industries": [
            "healthcare_medical",
            "retail_sales",
            "hospitality_food",
            "finance_banking",
        ],
    },
    "time_management": {
        "variations": [
            "prioritization",
            "multitasking",
            "deadline management",
            "organization",
        ],
        "relevant_industries": ["all"],
    },
    "teamwork": {
        "variations": [
            "collaboration",
            "team collaboration",
            "cross-functional teamwork",
        ],
        "relevant_industries": ["all"],
    },
    "attention_to_detail": {
        "variations": ["accuracy", "precision", "quality focus", "thoroughness"],
        "relevant_industries": [
            "healthcare_medical",
            "finance_banking",
            "legal",
            "manufacturing",
        ],
    },
}

# Experience level framework that works across all industries
UNIVERSAL_EXPERIENCE_LEVELS = {
    "entry_level": {
        "keywords": [
            "entry level",
            "new grad",
            "recent graduate",
            "trainee",
            "apprentice",
            "junior",
            "associate",
        ],
        "experience_range": "0-2 years",
        "expectations": "Learning role, supervised work, basic responsibilities",
    },
    "mid_level": {
        "keywords": ["experienced", "mid-level", "intermediate", "specialist"],
        "experience_range": "2-5 years",
        "expectations": "Independent work, specialized skills, some mentoring",
    },
    "senior_level": {
        "keywords": ["senior", "lead", "principal", "advanced", "expert"],
        "experience_range": "5-8 years",
        "expectations": "Leadership, mentoring, strategic thinking, complex problem solving",
    },
    "management_level": {
        "keywords": ["manager", "supervisor", "director", "head of", "chief"],
        "experience_range": "8+ years",
        "expectations": "Team management, strategic planning, business oversight",
    },
}
