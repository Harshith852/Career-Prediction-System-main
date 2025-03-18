import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import pickle
import os
from db import add_data, create_table
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests
import json

def clean_percentage(text):
    """Safely extract percentage from text."""
    try:
        if '(' not in text or ')' not in text:
            return 0.0
        percentage = text.split('(')[1].replace('%)', '').strip()
        return float(percentage)
    except (IndexError, ValueError, AttributeError):
        return 0.0

def load_model_and_encoder():
    """Load the model, label encoder, and category encoders."""
    required_files = {
        'model1.pkl': None,
        'label_encoder.pkl': None,
        'category_encoders.pkl': None
    }
    
    try:
        pkl_dir = os.path.join(os.path.dirname(__file__), 'pkl')
        if not os.path.exists(pkl_dir):
            st.error(f"pkl directory not found. Please run training.py first.")
            return None, None, None
            
        for filename in required_files:
            file_path = os.path.join(pkl_dir, filename)
            if not os.path.exists(file_path):
                st.error(f"Required file {filename} not found. Please run training.py first.")
                return None, None, None
            
            with open(file_path, 'rb') as file:
                required_files[filename] = pickle.load(file)
                
        return (required_files['model1.pkl'], 
                required_files['label_encoder.pkl'], 
                required_files['category_encoders.pkl'])
                
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottie_local(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def inputlist(Name, Contact_Number, Email_address,
             Logical_quotient_rating, coding_skills_rating, hackathons, 
             public_speaking_points, self_learning_capability, 
             Extra_courses_did, Taken_inputs_from_seniors_or_elders,
             worked_in_teams_ever, Introvert, reading_and_writing_skills,
             memory_capability_score, smart_or_hard_work, Management_or_Technical,
             Interested_subjects, Interested_Type_of_Books, certifications, 
             workshops, Type_of_company_want_to_settle_in, interested_career_area):
    
    try:
        # Convert yes/no responses to 1/0
        binary_map = {'yes': 1, 'no': 0}
        skill_map = {'poor': 0, 'medium': 1, 'excellent': 2}
        
        # Load the model and encoders
        model, label_encoder, category_encoders = load_model_and_encoder()
        if model is None or label_encoder is None or category_encoders is None:
            return ["Error: Failed to load required models"]
        
        # Initialize features list
        features = []
        
        # Numeric features
        features.extend([
            float(Logical_quotient_rating),
            float(coding_skills_rating),
            float(hackathons),
            float(public_speaking_points)
        ])
        
        # Binary features
        binary_features = [
            self_learning_capability.lower(),
            Extra_courses_did.lower(),
            Taken_inputs_from_seniors_or_elders.lower(),
            worked_in_teams_ever.lower(),
            Introvert.lower()
        ]
        features.extend([binary_map.get(feat, 0) for feat in binary_features])
        
        # Skill-based features
        features.extend([
            skill_map.get(reading_and_writing_skills.lower(), 0),
            skill_map.get(memory_capability_score.lower(), 0)
        ])
        
        # Smart/Hard worker features (one-hot encoded)
        features.extend([
            1 if smart_or_hard_work == 'Hard Worker' else 0,
            1 if smart_or_hard_work == 'Smart worker' else 0
        ])
        
        # Management/Technical features (one-hot encoded)
        features.extend([
            1 if Management_or_Technical == 'Management' else 0,
            1 if Management_or_Technical == 'Technical' else 0
        ])
        
        # Categorical features
        categorical_inputs = {
            'Interested subjects': Interested_subjects,
            'Interested Type of Books': Interested_Type_of_Books,
            'certifications': certifications,
            'workshops': workshops,
            'Type of company want to settle in?': Type_of_company_want_to_settle_in,
            'interested career area ': interested_career_area
        }
        
        for category, value in categorical_inputs.items():
            if category not in category_encoders:
                return [f"Error: Missing encoder for {category}"]
            encoded_value = category_encoders[category].get(value, 0)
            features.append(encoded_value)
        
        # Verify feature count
        expected_features = 21  # Update this number based on your model's requirements
        if len(features) != expected_features:
            return [f"Error: Feature mismatch. Expected {expected_features} features, got {len(features)}"]
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        probabilities = model.predict_proba(features_array)[0]
        
        # Get ALL possible career paths and their probabilities
        career_paths = label_encoder.classes_
        career_probs = list(zip(career_paths, probabilities))
        
        # Sort by probability in descending order
        career_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Format results - Include ALL careers with their probabilities
        results = []
        for career, prob in career_probs:
            # Include all careers with non-zero probability
            if prob > 0.001:  # Lower threshold to include more careers
                results.append(f"{career} ({prob*100:.1f}%)")
        
        return results if results else ["No suitable career matches found"]
    
    except Exception as e:
        return [f"Error: {str(e)}"]

def main():
    st.set_page_config(page_title="Career Prediction System", layout="wide")
    
    # Header section
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <h1 style='text-align: center; color: #2E3192; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);' class='floating-element'>
                🎯 Career Path Predictor AI
            </h1>
            <div style='text-align: center; padding: 20px;'>
                <h2>🚀 Discover Your Ideal Career Path</h2>
            </div>
            """, unsafe_allow_html=True)

    # Custom CSS with animation
    st.markdown("""
        <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0px); }
        }
        .floating-element {
            animation: float 3s ease-in-out infinite;
        }
        .main {
            padding: 0rem 0rem;
        }
        .st-emotion-cache-18ni7ap {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .st-emotion-cache-18ni7ap:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            background: linear-gradient(45deg, #2E3192, #1BFFFF);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .career-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .career-card:hover {
            transform: translateY(-5px);
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar for user information
    with st.sidebar:
        st.markdown("""
            <h2 style='text-align: center; color: #2E3192;'>Personal Information</h2>
            """, unsafe_allow_html=True)
        
        Name = st.text_input("Full Name")
        Contact_Number = st.text_input("Contact Number")
        Email_address = st.text_input("Email Address")
        
        if not Name and Email_address:
            st.warning("Please fill out your name and email address")

    # Main content in tabs
    tab1, tab2 = st.tabs(["Career Prediction", "Analytics Dashboard"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #2E3192;'>Technical Skills</h3>", unsafe_allow_html=True)
            
            Logical_quotient_rating = st.slider("Logical Quotient Rating", 1, 10, 5)
            coding_skills_rating = st.slider("Coding Skills Rating", 1, 10, 5)
            hackathons = st.number_input("Number of Hackathons", 0, 50, 0)
            public_speaking_points = st.slider("Public Speaking Points", 1, 10, 5)
            
            self_learning_capability = st.selectbox(
                'Self Learning Capability',
                ('yes', 'no')
            )
            
            Extra_courses_did = st.selectbox(
                'Extra Courses',
                ('yes', 'no')
            )

        with col2:
            st.markdown("<h3 style='color: #2E3192;'>Soft Skills & Preferences</h3>", unsafe_allow_html=True)
            
            Taken_inputs_from_seniors_or_elders = st.selectbox(
                'Takes Input from Seniors/Elders',
                ('yes', 'no')
            )
            
            worked_in_teams_ever = st.selectbox(
                'Team Work Experience',
                ('yes', 'no')
            )
            
            Introvert = st.selectbox(
                'Are you an Introvert',
                ('yes', 'no')
            )
            
            reading_and_writing_skills = st.selectbox(
                'Reading and Writing Skills',
                ('poor', 'medium', 'excellent')
            )
            
            memory_capability_score = st.selectbox(
                'Memory Capability Score',
                ('poor', 'medium', 'excellent')
            )

        st.markdown("<h3 style='color: #2E3192;'>Career Preferences</h3>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            smart_or_hard_work = st.selectbox(
                'Work Style',
                ('Smart worker', 'Hard Worker')
            )
            
            Management_or_Technical = st.selectbox(
                'Career Track',
                ('Management', 'Technical')
            )
            
            Interested_subjects = st.selectbox(
                'Interested Subjects',
                ('programming', 'Management', 'data engineering', 'networks', 
                 'Software Engineering', 'cloud computing', 'parallel computing', 
                 'IOT', 'Computer Architecture', 'hacking')
            )
            
            Interested_Type_of_Books = st.selectbox(
                'Preferred Book Category',
                ('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 
                 'Anthology', 'Dictionaries', 'Prayer books', 'Art', 'Encyclopedias', 
                 'Religion-Spirituality', 'Action and Adventure', 'Comics', 'Horror', 
                 'Satire', 'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 
                 'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 'Science', 
                 'Trilogy', 'Fantasy', 'Childrens', 'Mystery')
            )

        with col4:
            certifications = st.selectbox(
                'Certifications',
                ('information security', 'shell programming', 'r programming', 
                 'distro making', 'machine learning', 'full stack', 'hadoop', 
                 'app development', 'python')
            )
            
            workshops = st.selectbox(
                'Workshops',
                ('Testing', 'database security', 'game development', 'data science', 
                 'system designing', 'hacking', 'cloud computing', 'web technologies')
            )
            
            Type_of_company_want_to_settle_in = st.selectbox(
                'Preferred Company Type',
                ('BPA', 'Cloud Services', 'product development', 
                 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 
                 'Finance', 'Sales and Marketing', 'Product based', 'Service Based')
            )
            
            interested_career_area = st.selectbox(
                'Target Career Area',
                ('testing', 'system developer', 'Business process analyst', 
                 'security', 'developer', 'cloud computing')
            )

        if st.button("Predict Career Path", key="predict"):
            with st.spinner("Analyzing your profile..."):
                results = inputlist(Name, Contact_Number, Email_address,
                                  Logical_quotient_rating, coding_skills_rating,
                                  hackathons, public_speaking_points,
                                  self_learning_capability, Extra_courses_did,
                                  Taken_inputs_from_seniors_or_elders,
                                  worked_in_teams_ever, Introvert,
                                  reading_and_writing_skills, memory_capability_score,
                                  smart_or_hard_work, Management_or_Technical,
                                  Interested_subjects, Interested_Type_of_Books,
                                  certifications, workshops,
                                  Type_of_company_want_to_settle_in,
                                  interested_career_area)
                
                # Check if results contain an error message
                if len(results) == 1 and results[0].startswith("Error"):
                    st.error(results[0])
                    return
                
                # Success celebration
                st.balloons()
                st.markdown("🎉 **Analysis Complete!**")
                
                try:
                    # Create DataFrame for visualization
                    career_data = pd.DataFrame({
                        'Career': [cp.split(' (')[0].strip() for cp in results],
                        'Match %': [float(cp.split('(')[1].strip('%)')) for cp in results]
                    })
                    
                    # Display results and create visualizations
                    cols = st.columns(len(results))
                    for idx, (result, col) in enumerate(zip(results, cols)):
                        try:
                            career_name = result.split(' (')[0].strip()
                            match_percentage = clean_percentage(result)
                            
                            with col:
                                st.markdown(f"""
                                    <div style='background-color: {'#2E3192' if idx == 0 else '#4CAF50'}; 
                                              color: white; 
                                              padding: 15px; 
                                              border-radius: 10px; 
                                              text-align: center;
                                              margin: 5px;'>
                                        <h4 style='margin:0;'>#{idx + 1}</h4>
                                        <p style='font-size: 18px; margin:10px 0;'>{career_name}</p>
                                        <p style='font-size: 16px; margin:5px 0;'>{match_percentage:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error displaying prediction {idx + 1}: {str(e)}")
                            continue
                    
                    # Create the bar chart only if we have valid data
                    if not career_data.empty:
                        fig = px.bar(career_data, x='Career', y='Match %',
                                    title='Career Path Match Percentages',
                                    color='Match %',
                                    color_continuous_scale='Viridis')
                        
                        fig.update_layout(
                            xaxis_title="Career Paths",
                            yaxis_title="Match Percentage",
                            showlegend=False,
                            xaxis_tickangle=-45,
                            margin=dict(b=100)
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Display career insights
                    st.markdown("""
                        <div style='margin-top: 30px;' class='floating-element'>
                            <h3 style='color: #2E3192;'>Career Insights</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create metrics for top skills that influenced the prediction
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Technical Skills", f"{coding_skills_rating}/10", 
                                 delta="Critical for top prediction" if coding_skills_rating >= 7 else "Room for improvement")
                    with col2:
                        st.metric("Logical Quotient", f"{Logical_quotient_rating}/10",
                                 delta="Strong" if Logical_quotient_rating >= 7 else "Can improve")
                    with col3:
                        st.metric("Public Speaking", f"{public_speaking_points}/10",
                                 delta="Good" if public_speaking_points >= 7 else "Needs focus")
                    
                    # Add a career path comparison chart
                    career_data = pd.DataFrame({
                        'Career': [cp.split(' (')[0] for cp in results],
                        'Match %': [float(cp.split('(')[1].strip('%)')) for cp in results]
                    })
                    
                    fig = px.bar(career_data, x='Career', y='Match %',
                                title='Career Path Match Percentages',
                                color='Match %',
                                color_continuous_scale='Viridis')
                    
                    fig.update_layout(
                        xaxis_title="Career Paths",
                        yaxis_title="Match Percentage",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Add recommendations based on skills
                    st.markdown("""
                        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                            <h4 style='color: #2E3192;'>Personalized Recommendations</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    recommendations = []
                    if coding_skills_rating < 7:
                        recommendations.append("Consider taking additional programming courses to improve coding skills")
                    if public_speaking_points < 7:
                        recommendations.append("Join public speaking workshops or Toastmasters to enhance communication")
                    if Logical_quotient_rating < 7:
                        recommendations.append("Practice logical reasoning and problem-solving exercises")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")

                    # Advanced Analytics
                    st.markdown("<h3 style='color: #2E3192;'>Career Analytics</h3>", 
                              unsafe_allow_html=True)
                    
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        # Skills Radar Chart
                        skills_data = {
                            'Skills': ['Logic', 'Coding', 'Public Speaking', 
                                     'Team Work', 'Technical'],
                            'Score': [Logical_quotient_rating, coding_skills_rating,
                                    public_speaking_points, 
                                    1 if worked_in_teams_ever == 'yes' else 0,
                                    1 if Management_or_Technical == 'Technical' else 0]
                        }
                        
                        fig = go.Figure(data=go.Scatterpolar(
                            r=skills_data['Score'],
                            theta=skills_data['Skills'],
                            fill='toself'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 10]
                                )),
                            showlegend=False,
                            title="Your Skills Analysis"
                        )
                        st.plotly_chart(fig)

                    with col6:
                        # Career Compatibility Score
                        compatibility_score = (Logical_quotient_rating + 
                                            coding_skills_rating + 
                                            public_speaking_points) / 3
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = compatibility_score,
                            title = {'text': "Career Compatibility Score"},
                            gauge = {
                                'axis': {'range': [0, 10]},
                                'bar': {'color': "#2E3192"},
                                'steps': [
                                    {'range': [0, 3], 'color': "lightgray"},
                                    {'range': [3, 7], 'color': "gray"},
                                    {'range': [7, 10], 'color': "darkgray"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig)

                    # Save to database
                    create_table()
                    add_data(Name, Contact_Number, Email_address, Logical_quotient_rating,
                            coding_skills_rating, public_speaking_points, hackathons,
                            self_learning_capability, Extra_courses_did,
                            Taken_inputs_from_seniors_or_elders, worked_in_teams_ever,
                            Introvert, reading_and_writing_skills, memory_capability_score,
                            smart_or_hard_work, Management_or_Technical, Interested_subjects,
                            Interested_Type_of_Books, certifications, workshops,
                            Type_of_company_want_to_settle_in, interested_career_area)

                except Exception as e:
                    st.error(f"An error occurred while processing the results: {str(e)}")
                    return

    with tab2:
        st.markdown("<h3 style='color: #2E3192;'>Career Trends Dashboard</h3>", 
                   unsafe_allow_html=True)
        
        # Load the dataset
        df = pd.read_csv(os.path.join('data', 'mldata.csv'))
        
        # Calculate correlation
        corr = df[['Logical quotient rating', 'hackathons', 
                  'coding skills rating', 'public speaking points']].corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr, square=True, annot=True, linewidth=.4, center=2, ax=ax)
        st.pyplot(fig)
        
        # Add career distribution plot
        career_dist = df['Suggested Job Role'].value_counts()
        fig = px.pie(values=career_dist.values, 
                    names=career_dist.index,
                    title='Distribution of Career Paths')
        st.plotly_chart(fig)
        
        # Add skills importance plot
        skills_importance = pd.DataFrame({
            'Skill': ['Logical Quotient', 'Coding Skills', 'Public Speaking', 
                     'Team Work', 'Self Learning'],
            'Importance': [0.8, 0.9, 0.7, 0.6, 0.75]
        })
        fig = px.bar(skills_importance, x='Skill', y='Importance',
                    title='Skills Importance Analysis')
        st.plotly_chart(fig)

    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 20px; color: gray;'>
            Developed with ❤️ by <a href='https://github.com/harshith852/'>Harshith</a>
        </div>
        """, unsafe_allow_html=True)

def get_career_description(career_name):
    """Return a description for each career path"""
    descriptions = {
        'Software Engineer': "Design and develop software solutions, write code, and solve complex technical problems.",
        'Data Scientist': "Analyze complex data sets, create predictive models, and extract valuable insights.",
        'Web Developer': "Create and maintain websites, work with various web technologies and frameworks.",
        'AI/ML Engineer': "Develop machine learning models and artificial intelligence solutions.",
        'Systems Analyst': "Analyze and optimize organizational systems and processes.",
        'Database Administrator': "Manage and maintain database systems, ensure data security and accessibility.",
        'Network Engineer': "Design and maintain computer networks and communication systems.",
        'Cybersecurity Specialist': "Protect systems and networks from security threats and breaches.",
        'Cloud Engineer': "Design and manage cloud infrastructure and services.",
        'DevOps Engineer': "Implement and maintain development and deployment pipelines.",
        # Add more careers and descriptions as needed
    }
    return descriptions.get(career_name, "Explore this exciting career path and its opportunities.")

if __name__ == '__main__':
    main()
