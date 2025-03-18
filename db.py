import sqlite3
from contextlib import contextmanager

@contextmanager
def get_connection():
    conn = sqlite3.connect('data/career_predictions.db')
    try:
        yield conn
    finally:
        conn.close()

def create_table():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS predictions')
        
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact TEXT,
                email TEXT,
                logical_quotient INTEGER,
                coding_skills INTEGER,
                public_speaking INTEGER,
                hackathons INTEGER,
                self_learning INTEGER,
                extra_courses INTEGER,
                senior_input TEXT,
                team_work TEXT,
                introvert TEXT,
                reading_writing INTEGER,
                memory_capability INTEGER,
                smart_hard_work TEXT,
                management_technical TEXT,
                interested_subjects TEXT,
                interested_books TEXT,
                certifications INTEGER,
                workshops INTEGER,
                company_type TEXT,
                career_area TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def add_data(name, contact, email, logical_quotient, coding_skills, 
            public_speaking, hackathons, self_learning, extra_courses,
            senior_input, team_work, introvert, reading_writing, 
            memory_capability, smart_hard_work, management_technical,
            interested_subjects, interested_books, certifications, 
            workshops, company_type, career_area):
    
    data = (name, contact, email, logical_quotient, coding_skills,
            public_speaking, hackathons, self_learning, extra_courses,
            senior_input, team_work, introvert, reading_writing, 
            memory_capability, smart_hard_work, management_technical,
            interested_subjects, interested_books, certifications, 
            workshops, company_type, career_area)
            
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                name, contact, email, logical_quotient, coding_skills,
                public_speaking, hackathons, self_learning, extra_courses,
                senior_input, team_work, introvert, reading_writing,
                memory_capability, smart_hard_work, management_technical,
                interested_subjects, interested_books, certifications,
                workshops, company_type, career_area
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', data)
        conn.commit()

# Initialize the database
def init_db():
    create_table()

if __name__ == "__main__":
    init_db()
