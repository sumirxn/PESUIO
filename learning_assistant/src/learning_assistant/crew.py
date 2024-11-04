from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from learning_assistant.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool
import os 
os.environ['SERPER_API_KEY'] = 'b7a51a6c06763663f08cfc544494d090ef9c1803'

@CrewBase
class LearningAssistantCrew():
	"""LearningAssistant crew"""

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=[SerperDevTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True
		)

	@agent
	def summarizer(self) -> Agent:
		return Agent(
			config=self.agents_config['summarizer'],
			verbose=True
		)
	
	@agent
	def questions(self) -> Agent:
		return Agent(
			config=self.agents_config['questions'],
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def summarizing_task(self) -> Task:
		return Task(
			config=self.tasks_config['summarizing_task'],
			output_file='summary.md'
		)
	
	@task
	def question_task(self) -> Task:
		return Task(
			config=self.tasks_config['question_task'],
			output_file='questions.md'
		)	

	@crew
	def crew(self) -> Crew:
		"""Creates the LearningAssistant crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)