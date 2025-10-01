import sqlite3
import json

schema = """
CREATE TABLE IF NOT EXISTS courses (
    code TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    hours_lecture INTEGER,
    hours_tutorial INTEGER,
    hours_practical INTEGER,
    credits INTEGER,
    prereq TEXT,
    overlap TEXT
);

CREATE TABLE IF NOT EXISTS overlaps (
    id INTEGER PRIMARY KEY,
    code_1 TEXT REFERENCES courses(code),
    code_2 TEXT REFERENCES courses(code)
);

CREATE TABLE IF NOT EXISTS offerings (
    id INTEGER PRIMARY KEY,
    code TEXT REFERENCES courses(code),
    year TEXT,
    semester INTEGER,
    coordinator TEXT,
    slot TEXT
);

CREATE TABLE IF NOT EXISTS user (
    entry_no TEXT PRIMARY KEY,
    name TEXT,
    hostel TEXT,
    course TEXT,
    department TEXT,
    grad_year TEXT
);

CREATE TABLE IF NOT EXISTS user_courses (
    id INTEGER PRIMARY KEY,
    entry_no TEXT REFERENCES user(entry_no),
    offering_id INTEGER REFERENCES offerings(id)
);
"""

def init_db(db_path="university.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(schema)
    conn.commit()
    conn.close()

def parse_json(filename):
    courses = None
    with open(filename) as file:
        courses = json.load(file)
    return courses

def insert_in_db(cursor, course): # code name desc hours (lec, tut, practical), credits, prereqs
    data = (course['code'], course['name'], course['description'], course['hours']['lecture'], course['hours']['tutorial'], course['hours']['practical'], course['credits'], course['prereqs'], course['overlap'])
    cursor.execute("INSERT OR REPLACE INTO courses (code, name, description, hours_lecture, hours_tutorial, hours_practical, credits, prereq, overlap) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
    # add overlaps
    for overlap in course['overlap'].split(','):
        code = overlap.strip()
        if not code: continue
        data = (course['code'], code)
        cursor.execute("INSERT INTO overlaps (code_1, code_2) VALUES (?, ?)", data)

def read_jsonl(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            res.append(json.loads(line))
    return res

def insert_offering(cursor, offering): # code year semester coordinator slot
    data = (offering['course_code'], offering['year'], offering['semester'], offering['instructor'], offering['slot'])
    cursor.execute("INSERT INTO offerings (code, year, semester, coordinator, slot) VALUES (?, ?, ?, ?, ?)", data)

if __name__ == "__main__":
    init_db(db_path="courses.sqlite")
    courses = parse_json('sources/processed/courses.json')
    offerings = read_jsonl('sources/jsonl/courses_offered.jsonl')

    conn = sqlite3.connect("courses.sqlite")
    cursor = conn.cursor()
    
    for code, course in courses.items():
        insert_in_db(cursor, course)
    for offering in offerings:
        insert_offering(cursor, offering)
    conn.commit()
    conn.close()
 
