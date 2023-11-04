import os
import pytube
from pytube import YouTube
# Import Azure OpenAI
from langchain.llms import AzureOpenAI
import whisper
import streamlit as st
import pandas as pd
import numpy as np
from typing import List
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = st.secrets["openaiUrl"]
os.environ["OPENAI_API_KEY"] = st.secrets["openaiKey"]
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

model = whisper.load_model('base')

secondself_template = """You are a chatbot that helps with answers for questions about this context: {context}.
  You have all the information here: {information}
  You need to answer the question you receive, nothing else.
  Don't add more questions and answers, just answer the question asked by the human.
  Do not repeat yourself.
  If in some context, you feel the answer or context could be amazing, add at the end of your answer: NIIIIICE!!
  When you finish answering the question, do not add more information.
  {chat_history}
  Human: {human_input}
  Chatbot:
  """

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def transcript_video(youtube_url):
    youtube_link = youtube_url
    youtube_video = YouTube(youtube_link)
    video_id = pytube.extract.video_id(youtube_link)
    streams = youtube_video.streams.filter(only_audio=True)
    stream = streams.first()
    mp4_video = stream.download(filename='youtube_video.mp4')
    audio_file = open(mp4_video, 'rb')
    output = model.transcribe("youtube_video.mp4")
    transcription = { "title": youtube_video.title.strip(), "transcription": output['text'] }
    return transcription

def get_answer(question, transcription):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a chatbot having a conversation with a human about this context: """+transcription["title"]+""".
                        You have all the information here: """+transcription["transcription"]+"""
                        You need to answer the question you receive, nothing else.
                        Give short and precise answers.
                        Don't add more questions and answers, just answer the question asked by the human.
                        Do not repeat yourself.
                        If in some context, you feel the answer or context could be amazing, add at the end of your answer: NIIIIICE!!
                        When you finish answering the question, do not add more information."""
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
    )

    # Create an instance of Azure OpenAI
    # Replace the deployment name with your own
    llm = AzureOpenAI(
        deployment_name="secondself",
        model_name="gpt-35-turbo"
    )
    qa = LLMChain(llm=llm, prompt=prompt,verbose=True, memory=memory)

    result = qa.predict(human_input = question)
    return result


html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

with st.sidebar:
    st.markdown("""
    # About 
    SecondSelf is a helper tool built ...
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Ask...
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    Made by..
    """,
    unsafe_allow_html=True,
    )

st.markdown("""
# SecondSelf
""")

st.markdown("""
### Ask the questions and start remembering your experiences
""")

youtube_video = st.text_input("Youtube video: ", placeholder="https://youtu.be/...")

if youtube_video:
    st.video(youtube_video)

large_text = st.text_area("Ask your question: ", placeholder="What ...?")

if st.button("Get an answer"):
    transcription = transcript_video(youtube_video)
    st.write("Answer: ", get_answer(large_text, transcription))


