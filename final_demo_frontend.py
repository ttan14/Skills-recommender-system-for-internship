# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk #for NLP
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('universal_tagset')
nltk.download('reuters')
nltk.download('averaged_perceptron_tagger')
###vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
### similarity measure
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt


header = st.container()
part_1 = st.container()
part_2 = st.container()


## in terminal: cd Documents/'Tessas Internship'/'Phase 2 preso' -> streamlit run demo_frontend.py
    

##### reading data sources #####   
coursedata = '/Users/somethingcreative-7/Documents/Tessas Internship/Phase 2 preso/Supplementary documents/coursefeaturesoutput.xlsx'
coursedata = pd.read_excel(coursedata)

courseradb = pd.read_csv('/Users/somethingcreative-7/Documents/Tessas Internship/Phase 2 preso/Supplementary documents/Coursera.csv') #only used to pull out course details for user at the end

userdata = '/Users/somethingcreative-7/Documents/Tessas Internship/Phase 2 preso/Supplementary documents/JDoutput.xlsx'
userdata = pd.read_excel(userdata)
userdata['Job Role'] = userdata['Job Role'].apply(lambda x: x.lstrip(' '))
## Pull out proficiency description (input for Reco model)
###load proficiency descriptions data from database
skill_prof_data = '/Users/somethingcreative-7/Documents/Tessas Internship/Phase 2 preso/Supplementary documents/proficiency_description.xlsx'
skill_prof_df = pd.read_excel(skill_prof_data)
skill_prof_df=skill_prof_df.drop(columns=['Description'],axis=1)


skill_prof_df['skill_name']= skill_prof_df['Skill-Level']
skill_prof_df['Level']= skill_prof_df['Skill-Level']

##extract skill name 
skill_prof_df['skill_name']= skill_prof_df['skill_name'].apply(lambda i: i.split("-Level",-1)[0])
##extract level as numeric
skill_prof_df['Level']= skill_prof_df['Level'].apply(lambda i: i.rsplit('-Level', 1)[-1])

##### Demo UI #####

with header:
    st.markdown("<h1 style='text-align: center; color: #f63366; font-size: 70px; '>Skills and Courses Recommender System</h1>", unsafe_allow_html=True)
############################################## 
    st.markdown("***")    

##### PART 1 #####    

with part_1:
    ##header
    st.markdown("<h2 style=' color: black; font-size: 40px;'>Part 1: Identifying Competency Gaps </h2>", unsafe_allow_html=True)
    ##description
    st.markdown("<p style=' color: black; font-size: 15px;'>This part of the system helps the employee to identify their competency gaps, vis-a-vis their job(target job) </p>", unsafe_allow_html=True)

    user_instruction_job = '<p style="font-family:sans-serif; color:Orange; font-size: 24px;">Select your target job:</p>'
    st.markdown(user_instruction_job,unsafe_allow_html=True)
    job_list= list(sorted(userdata['Job Role']))
    
    ###dropdown for USER to select target job
    target_job= st.selectbox('', options=job_list)
    target_job_row = userdata.loc[userdata['Job Role']==target_job]
    target_job_df = pd.DataFrame(target_job_row)
    st.table(target_job_df[['Job Role', 'Track', 'Sub-track', 'Occupation']])
    
    ##extract skills col required for job # start with technical skills?
    skills_df=pd.DataFrame(target_job_df['Skills'])
    ##transpose and clean, only technical skills pulled out- contains("-Level[0-9]")
    skills_df=skills_df.assign(Skills=skills_df['Skills'].str.split(",")).explode('Skills')
    skills_df['Skills'] = skills_df['Skills'].apply(lambda x: x.lstrip(' '))
    skills_df = skills_df[skills_df['Skills'].str.contains("-Level[0-9]")]
    skills_df['skills_original']=skills_df['Skills']
    
    ##extract skill levels into new column
    skills_df['expected_level']= skills_df['Skills']
    skills_df['expected_level']= skills_df['expected_level'].apply(lambda i: i.rsplit('-Level', 1)[-1])
    skills_df['Skills']= skills_df['Skills'].apply(lambda i: i.split("-Level",-1)[0])
    
    #prep df for users' inputs
    skills_df['user_level']=skills_df['expected_level']
    skills_df=skills_df.reset_index(level=None, drop=True, inplace=False)
    st.text('') 
    st.text('')
    st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 20px;">Here are the competencies required for this job:</p>',unsafe_allow_html=True)
    text= (str(len(skills_df)) + ' competencies found for '+ target_job)
    st.text(text)
    st.write(skills_df['skills_original'])
    
############################################## 
    st.markdown("***")
   
    ## user input their own actual skill level
    user_instruction_prof = '<p style="font-family:sans-serif; color:Orange; font-size: 24px;">Next, you will be required to indicate your own proficiency level for each of these skills:</p>'
    st.markdown(user_instruction_prof,unsafe_allow_html=True)
    ##iterate down skills list to get user input for each skill
    for i in skills_df['Skills']:
        index= skills_df.index[skills_df['Skills'] == i]
        #get user input
        text= "Enter your proficiency for " + str(i)
        
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
        new=st.radio(text,('0','1','2','3','4','5','6')) 
        ##new = int(input(text))
        skills_df.at[index, 'user_level']= new

    ##skills_df['user_level']=skills_df['expected_level'].apply(lambda i: int(i)-1) ###used for level-by-level testing
    ## Store gap calculation as new column in df
    skills_df['gap'] = skills_df.apply(lambda row: int(row.user_level) - int(row.expected_level), axis=1)    

    ###differentiate between skills that the user already has and those that are lacking
    ##pull out gaps only, into new df
    gaps_df = skills_df.loc[skills_df['gap'] < 0]
    ##reset index for gaps_df
    gaps_df=gaps_df.reset_index(level=None, drop=True, inplace=False)

############################################## 
    st.markdown("***") 
    ## Return list of skill gaps (list as input for Reco/similarity measure)
    st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 20px;">Here are your identified competency gaps:</p>',unsafe_allow_html=True)

    gaps_df['gap']=gaps_df['gap'].apply(int)
    gaps_df['gap']=gaps_df['gap'].apply(abs)
    ##gaps_df
   
   ####BAR CHART
    gaps_bar= gaps_df[["Skills", "user_level","gap","expected_level"]]   ###can change gaps_df to skills_df to display all competencies instead of only gaps
    gaps_bar['expected_level']=gaps_bar['expected_level'].apply(int)
    gaps_bar['user_level']=gaps_bar['user_level'].apply(int)
    
     
    expected_bar = alt.Chart(gaps_bar).mark_bar(color='grey',opacity=0.6).encode(
        x=alt.X('Skills', axis=alt.Axis(title='Competency')), y=alt.Y('expected_level', axis=alt.Axis(
        title='Proficiency level',values=[0,1,2,3,4,5,6]), scale=alt.Scale(domain=[0, 6]))).properties(height=600,
        width=alt.Step(40))  # controls width of bar.
    
                                                                                                       
    selection = alt.selection_multi(fields=['user_level'], bind='legend')

    user_bar = alt.Chart(gaps_bar).mark_bar(color='orange').encode(   # controls width of tick.
        x='Skills',
        y='user_level'
    )

   
    expected_bar + user_bar
    

 
############################################## 
    st.markdown("***")

##### end of part 1 ####

##### PART 2 #####
   
###pull out 'expected level' prof description from skill_prof_df db

reco_gap = pd.DataFrame() 

for i in gaps_df['skills_original']:
    row=skill_prof_df.loc[skill_prof_df['Skill-Level']==i]
    reco_gap=reco_gap.append(row) 

###create a col in df that pulls out a list of the prof levels the user needs to pick up (numeric)
gaps_df['all_lvls']= gaps_df['expected_level']
for index, row in gaps_df.iterrows():
    x= list(range(int(row.user_level)+1, int(row.expected_level)+1))
    gaps_df.at[index, 'all_lvls']=x
gaps_df['all_lvls_text']=gaps_df['Skills']


###match skill name and proficiency level across gaps_df and prof desc df
    
for index, row in gaps_df.iterrows():
    if len(row.all_lvls)==1:
        name= row.skills_original
        text= skill_prof_df.loc[skill_prof_df['Skill-Level']==name]['clean'].values[0]
        row.all_lvls_text=str(text)
    else:
        for i in row.all_lvls:        ##for i in list ##one list per row
            name= row.Skills+'-Level'+ str(i)
            ###find name in skill_prof_df list
            text= skill_prof_df.loc[skill_prof_df['Skill-Level']==name]['clean'].values[0]
            if pd.isna(text)==True:
                continue
            ### append new text to end of all_lvls_text
            row.all_lvls_text = str(row.all_lvls_text) +' '+ str(text)            
    gaps_df.at[index, 'all_lvls_text'] = row.all_lvls_text

###pull out proficiency data (skill gaps) for reco model
reco_gap= gaps_df[['Skills','all_lvls_text']]
#tokenize proficiency description (after combining across skill levels)
reco_gap['all_lvls_text'] = reco_gap['all_lvls_text'].apply(lambda i: pos_tag(word_tokenize(i),tagset='universal'))

###lemmatize proficiency description 

##define lem function
wnl = nltk.WordNetLemmatizer()
def lemmaNVAR(wpos):
    lemmas = []
    for w, pos in wpos:
        if pos == 'NOUN':
            lemmas.append(wnl.lemmatize(w.lower(), pos = 'n'))
    return lemmas

reco_gap['all_lvls_text'] = reco_gap['all_lvls_text'].apply(lemmaNVAR)
print(reco_gap['all_lvls_text'])

###identify stopwords
mystopwords=stopwords.words("english") + ['nan'] #remove NAs here

##remove stopwords
reco_gap['all_lvls_text'] = reco_gap['all_lvls_text'].apply(lambda i: [t for t in i if t not in mystopwords])

###pre-processing of prof data
##prep df structure as we want to use it as input for reco model
##rename to column headings so it can appended to course data later on
reco_gap.rename(columns = {'Skills': 'course_name'}, inplace = True)
reco_gap.rename(columns = {'all_lvls_text': 'tags'}, inplace = True)
## clean text
reco_gap['tags'] = reco_gap.tags.apply(lambda x: ', '.join([str(i) for i in x]))

reco_gap['tags'] = reco_gap['tags'].str.replace(',',' ')
reco_gap['tags'] = reco_gap['tags'].apply(lambda x:x.lower()) #lower casing the tags column 
##reset index for df
reco_gap=reco_gap.reset_index(level=None, drop=True, inplace=False)

#defining the stemming function to be used later on (see Model Function below)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)
     
with part_2:
    ##header
    st.markdown("<h2 style=' color: black; font-size: 40px;'>Part 2: Course Recommendations </h2>", unsafe_allow_html=True)
    ##description
    st.markdown("<p style=' color: black; font-size: 15px;'>This part of the system recommends courses to employees, based on their competency gaps above </p>", unsafe_allow_html=True)

    
    st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 20px;">Here are the Top 5 recommended course for the competency gaps identified:</p>',unsafe_allow_html=True)

    ###courses data
    new_df_courses=coursedata[['course_name','tags']]
    
    ###Reco based on TF_IDF + cosine similarity
    cv = TfidfVectorizer(min_df = 2, max_df=0.9,stop_words='english') ##TF-IDF count vec
    ##create df to save course recommendations
    for_course_lookup = {}
    
    ###for every skill gap (where name and tag already clean/processed), pull out description and run TFIDF and the similarity model to push out 1 recommendation list per skill gap
    ###iter over df
    for index, row in reco_gap.iterrows():
        model_df=new_df_courses
        target=row['course_name']
        ###combine prof desc row to courses df to create a main df that goes into TF-IDF and then reco model
        combined=model_df.append(row)
        ###applying stemming on the tags column
        combined['tags'] = combined['tags'].apply(stem)  
        ###applying vectorisation on tags
        vectors = cv.fit_transform(combined['tags']).toarray()
        
        ###TF-IDF###
        ##Compute the Cosine Similarity for all courses to job title (last row) based on the count_matrix
        cosine_sim = cosine_similarity(vectors) 
        vec_into_df = pd.DataFrame(cosine_sim[-1],index=combined.course_name,columns=[target])
        
        ###sort by most similar/ closest datapoint to target job title
        recommended_courses= vec_into_df[target].sort_values(ascending=False)[:6]
    
        recommended_courses=list(recommended_courses.index.values)
        text1= str(index+1)+ '. ' + 'To address '+ str(recommended_courses[0]) + ', the Top 5 most relevant courses are: ' 
        st.markdown("<p style=' color: black; font-size: 15px;text1", unsafe_allow_html=True)

        #st.markdown(df.index.tolist())
        st.markdown(text1)
        df = pd.DataFrame(recommended_courses[1:],columns=['Course Title'])
        df.index = np.arange(1, len(df) + 1)
        st.table(df)

    st.markdown('<p style="font-family:sans-serif; color:seagreen; text-align:centre; opacity:0.6 ;font-size: 22px;">END</p>',unsafe_allow_html=True)


############################################## 
    #st.markdown(recommended_courses.index.tolist())

     #row=recommended_courses.loc[skill_prof_df['Skill-Level']==i]
    #for_course_lookup[recommended_courses[0]]= recommended_courses[1:]# st.markdown("***")   
    # ##### Look-up for course recommendations #####
    # user_instruction_prof = '<p style="font-family:sans-serif; color:Orange; font-size: 20px;">Please select a competency to view course details:</p>'
    # st.markdown(user_instruction_prof,unsafe_allow_html=True)
   
    # ##create empty df for course details
    # course_details = pd.DataFrame(columns=['Course Name','Course Description','University','Course Rating','Course URL','Skills'])
    
    # ###USER select competency to lookup courses
    # user_comp_gaps= for_course_lookup.keys()
    # i= st.selectbox('', options=user_comp_gaps)
    # recommended_courses= for_course_lookup[i]
    
    # for i in list(recommended_courses):
    #     #row=courseradb.iloc[courseradb['Course Name']==i]
    #     index=courseradb[courseradb['Course Name']==i].index.values   
    #     row=courseradb.iloc[index]
    #     course_details=course_details.append(row) 
        
    # course_details.drop(labels=['Course Rating','Difficulty level'], axis=1)
    # course_details.index = np.arange(1, len(course_details) + 1) #reset df index to start from 1
    # st.write(course_details)    

    
