"""Expert Programmer Prompt - https://prompts.chat"""

from pydantic import BaseModel


class ExpertProgrammerInput(BaseModel):
    topic: str


class ExpertProgrammer:
    def __init__(self, topic):
        self.INSTRUCTION = f"""
        As an experienced expert programmer and technical writer, generate an outline for a blog with examples about {topic}.
        Provide a brief introduction to the topic, explain the importance of the topic, and provide examples of how the topic is used in real-world applications.
        The out should be generated in markdown format.
        """


if __name__ == "__main__":
    input_data = ExpertProgrammerInput(topic="Artificial Intelligence")
    expert_programmer = ExpertProgrammer(topic=input_data.topic)
    print(expert_programmer.INSTRUCTION)
