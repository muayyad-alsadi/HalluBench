
from dataclasses import dataclass

from openai import OpenAI

from ollama import Client as OllamaClient

from google import genai


def completion_openai(api_key, model, messages, max_completion_tokens=None):
    client = OpenAI(
        api_key=api_key
    )
    temperature = 1 if model.endswith('-5') or "-5-" in model else 0.0
    if max_completion_tokens is None:
        max_completion_tokens = 16000 if '4o' else 32000
    kwargs = dict(
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )
    #if model.endswith('-5') or "-5-" in model: kwargs['reasoning_effort']="minimal"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    print(response.usage)
    # Extract and return the response text
    #response_text = response.choices[0].message.content
    return response

@dataclass
class Message:
    content: str

@dataclass
class Choice:
    message: Message

@dataclass
class FakeResponse:
    choices: list

def completion_ollama(api_key, model, messages, max_completion_tokens=None):
    client = OllamaClient(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + api_key}
    )
    res=client.chat(model, messages=messages, stream=False)
    content = res['message']['content']
    return FakeResponse(choices=[Choice(Message(content))])

def completion_google(api_key, model, messages, max_completion_tokens=None):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents="\n\n".join([m["content"] for m in messages]),
    )
    content = response.text
    return FakeResponse(choices=[Choice(Message(content))])


def completion(api_key, provider, model, messages):
    if provider=='openai':
        return completion_openai(api_key, model, messages)
    if provider=='ollama':
        return completion_ollama(api_key, model, messages)
    if provider=='google' or provider=='gemini':
        return completion_google(api_key, model, messages)
    raise NotImplementedError(f"provider not implemented: {provider}")
