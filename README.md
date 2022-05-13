**Building skills/courses recommender system for internship**

**Files**:

- **Original data.zip**: contains the raw data from the SkillsFuture competency frameworks (135 Excel for competency descriptions, 104 PDF Job description/competency frameworks)

- **ETL scripts**: Contains the script used to extract and transform data from original data sources (original data in xx file)
    - **Extract_ Course description from Coursera data.ipynb**: output folder -> coursefeaturesoutput.xlsx
    - **Extract_ job description with skills tagging (competency framework).ipynb**: output folder -> JDoutput.xlsx
    - **Extract_ proficiency description from comp framework.ipynb**: output folder -> proficiency_description.xlsx
    
- **Datasources**: Contains extracted and cleaned data from SkillsFuture framework and Coursera. 
    - **coursefeaturesoutput.xlsx**: Course description data from Coursera 
    - **Coursera.csv**: Full course details from Coursera for lookup (after recommendation)
    - **JDoutput.xlsx**: Job scope and competency data from SkillsFuture, i.e. what competencies required for what job
    - **proficiency_description.xlsx**: Corresponding proficiency descriptions for each competency (Level 1-6)

- **Model script**:  Contains the script to build and run the actual model (reads from the above files in **Datasources**)
    - **Model Part 1_pull out skill gaps.ipynb**: Part 1 of the recommender system that helps employees identify their skill gaps vis-a-vis their current job title
    - **Model Part 2_pull proficiency description based on skill gaps.ipynb**: Added Part 2 of the recommender system that calculates similarity between proficiency and course descriptions and returns user the most similar courses, to address their skill gaps (Part 1 of the model also included in this script so it can run)

- **final_demo_frontend.py**: Contains the full reco system (front end and back end), to be _run on streamlit_. 

- **CSIT internship presentation (13 May).pptx**: Final presentation to company

- 
