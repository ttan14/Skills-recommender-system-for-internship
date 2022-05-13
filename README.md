**Building skills/courses recommender system for internship**

**Files**:

- **ETL scripts**: Contains the script used to extract and transform data from original data sources (original data in xx file)
    - Extract_ Course description from Coursera data.ipynb: output folder -> coursefeaturesoutput.xlsx
    - Extract_ job description with skills tagging (competency framework).ipynb: output folder -> JDoutput.xlsx
    - Extract_ proficiency description from comp framework.ipynb: output folder -> proficiency_description.xlsx
    
- **Datasources**: Contains extracted and cleaned data from SkillsFuture framework and Coursera. 
    - coursefeaturesoutput.xlsx: Course description data from Coursera 
    - Coursera.csv: Full course details from Coursera for lookup (after recommendation)
    - JDoutput.xlsx: Job scope and competency data from SkillsFuture, i.e. what competencies required for what job
    - proficiency_description.xlsx: Corresponding proficiency descriptions for each competency (Level 1-6)


Part 1: Flag out employee skill gaps
Part 2: Recommend courses to address skill gaps

