import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool

load_dotenv()

llm = LLM(model="gpt-4o-mini")
current_dir = os.path.dirname(os.path.abspath(__file__))


coding_agent = Agent(
    role="Senior Python Developer",
    goal="Craft well-designed and thought-out code. Also perform thorough and concise code reviews.",
    backstory="""You are a senior Python developer with extensive experience 
    in software architecture, programming best practices and code reviews.""",
    allow_code_execution=True,
    llm=llm,
)
review_agent = Agent(
    role="Senior Python Developer",
    goal="Perform thorough and concise code reviews.",
    backstory="""You are a senior Python developer with extensive experience 
    in software architecture, programming best practices and code reviews.""",
    llm=llm,
)

generate_hello_task = Task(
    description="Generate a simple python program that receive the name of the user as input and prints a greeting with the name and a random integer between 1 and 10.",
    expected_output="""A valid and well-written python file with the described functionality implemented. 
    Don't include ticks at the beginning or end of the file.""",
    agent=coding_agent,
    output_file="hello_gen.py"
)

read_og_file_task = Task(
    description="Read the original program implementation.",
    expected_output="""The full original implementation in python with comments.""",
    agent=review_agent,
    tools=[FileReadTool(file_path=os.path.join(current_dir, "hello.py"))],
)

analyze_og_file_task = Task(
    description="Read the original program implementation and compare with the program generated.",
    expected_output="""A file with a detailed comparison and analysis of code practices between the 
    two files. Also make improvement recommendations for the original implementation.""",
    agent=review_agent,
    output_file="analysis.md"
)

dev_crew = Crew(
    agents=[coding_agent, review_agent],
    tasks=[generate_hello_task, read_og_file_task, analyze_og_file_task],
    verbose=True
)

result = dev_crew.kickoff()

print(result)