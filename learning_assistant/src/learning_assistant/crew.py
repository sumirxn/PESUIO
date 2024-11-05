from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import tool
import base64
import requests
import json
import os
from typing import Optional

# # Uncomment the following line to use an example of a custom tool
# # from learning_assistant.tools.custom_tool import MyCustomTool

# # Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool
# import os 
# os.environ['SERPER_API_KEY'] = 'b7a51a6c06763663f08cfc544494d090ef9c1803'

@tool
def mermaid_to_image_tool(mermaid_code: str, img_type: str = 'png') -> str:
    """
    CrewAI Tool: Convert Mermaid diagram code to an image using an online API.
    
    Args:
        mermaid_code (str): The Mermaid diagram code as a string.
            Example valid syntax:
            
            graph LR
            A[Start] --> B[Process]
            B --> C[End]
            
            
            Example with labels:
            
            graph TD
            A[Start] -->|trigger| B[Process]
            B -->|complete| C[End]
            
            
            Note: Avoid using special characters like '|>' or '<|' in arrows.
            Use standard '-->' arrows instead.
            
        img_type (str): Image format - 'png' or 'svg'. Defaults to 'png'.
    
    Returns:
        str: Path to the saved image file (e.g., 'mermaid_diagram.png')
        
    Raises:
        ValueError: If invalid image type is specified or invalid Mermaid syntax
        requests.RequestException: If API request fails
        
    Example:
        >>> code = '''
        graph LR
        A[Start] --> B[Process]
        B --> C[End]
        '''
        >>> mermaid_to_image_tool(code, 'png')
        'mermaid_diagram.png'
    """
    if img_type not in ['png', 'svg']:
        raise ValueError("img_type must be either 'png' or 'svg'")
    
    # Sanitize mermaid code
    mermaid_code = mermaid_code.replace('|>', '-->')
    mermaid_code = mermaid_code.replace('-->|', '-->|text|')
    
    try:
        # Encode the Mermaid code to base64
        encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        api_url = f"https://mermaid.ink/img/{encoded}"
        
        # Make the request
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        # Generate output path
        output_path = f"mermaid_diagram.{img_type}"
        os.makedirs('.', exist_ok=True)
        
        # Save the image
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to generate Mermaid diagram: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing Mermaid diagram: {str(e)}")

# @CrewBase
# class LearningAssistantCrew():
# 	"""LearningAssistant crew"""

# @agent
# def researcher(self) -> Agent:
# 	return Agent(
# 		config=self.agents_config['researcher'],
# 		tools=[SerperDevTool()], # Example of custom tool, loaded on the beginning of file
# 		verbose=True
# 	)

# @agent
# def summarizer(self) -> Agent:
# 	return Agent(
# 		config=self.agents_config['summarizer'],
# 		# tools=[mermaid_to_image_tool],
# 		verbose=True
# 	)
	
# @agent
# def questions(self) -> Agent:
# 	return Agent(
# 		config=self.agents_config['questions'],
# 		verbose=True
# 	)

# @task
# def research_task(self) -> Task:
# 	return Task(
# 		config=self.tasks_config['research_task'],
# 	)

# @task
# def summarizing_task(self) -> Task:
# 	return Task(
# 		config=self.tasks_config['summarizing_task'],
# 		output_file='summary.md'
# 	)
	
# @task
# def question_task(self) -> Task:
# 	return Task(
# 		config=self.tasks_config['question_task'],
# 		output_file='questions.md'
# 	)	

# @crew
# def crew(self) -> Crew:
# 	"""Creates the LearningAssistant crew"""
# 	return Crew(
# 		agents=self.agents, # Automatically created by the @agent decorator
# 		tasks=self.tasks, # Automatically created by the @task decorator
# 		process=Process.sequential,
# 		verbose=True,
# 		# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
# 	)

# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task
# from crewai_tools import tool

# Uncomment the following line to use an example of a custom tool
# from learning_assistant.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool
# import os 
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
			tools=[mermaid_to_image_tool],
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
	def flowchart_task(self) -> Task:
		return Task(
			config=self.tasks_config['flowchart_task'],
			# output_file='summary.md' this is not needed as it generates an imaeg
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