import os
import pathlib
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
from tools import calculate

current_dir = pathlib.Path(__file__).parent.resolve()
env_dir = os.path.join(current_dir, ".env")
print(f"current_dir: {env_dir}")
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(os.getenv("OPENAI_API_KEY"))
math_input = input("what is your math question: ")

math_agent = Agent(
    role="Math Magician",
    goal="You are able to evaluate any math expression",
    backstory="YOU ARE A MATH WHIZ",
    verbose=True,
    tools=[calculate]
)

writer = Agent(
    role="Writer",
    goal="Craft compelling explanations based from results of math equations.",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    """,
    verbose=True
)

task1 = Task(
    description=f"{math_input}",
    expected_output="Give full details in bullet points.",
    agent=math_agent
)

task2 = Task(
    description="Using insights provided, explain in great detail how the equation and result were formed.",
    expected_output="Explain in great detail and save in markdown. Do not add thje tripple tick marks at the beginning or end of the file. Don't say what type it is in the first line.",
    output_file="math.md",
    agent=writer
)

crew = Crew(
    agents=[math_agent, writer],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()

print(result)