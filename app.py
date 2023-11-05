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
from elevenlabs import set_api_key, generate, save


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = st.secrets["openaiUrl"]
os.environ["OPENAI_API_KEY"] = st.secrets["openaiKey"]
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

elevenlabs_api_key = st.secrets["elevenlabs_api_key"]
set_api_key(elevenlabs_api_key)

model = whisper.load_model('base')

secondself_template = """You are a chatbot that helps with answers for questions about this context: {context}.
  You have all the information here: {information}
  You need to answer the question you receive, nothing else.
  Don't add more questions and answers, just answer the question asked by the human.
  Do not repeat yourself.
  Try to keep each answer to a maximum of 3 sentences.
  If in some context, you feel the answer or context could be amazing, add at the end of your answer: NIIIIICE!!
  When you finish answering the question, do not add more information.
  {chat_history}
  Human: {human_input}
  Chatbot:
  """

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def transcript_video(youtube_url):
    youtube_link = youtube_url
    video_id = pytube.extract.video_id(youtube_link)
    if os.path.exists(video_id+".csv"):
        existing_transcription = pd.read_csv('./'+video_id+'.csv', sep=";")
        return existing_transcription
    youtube_video = YouTube(youtube_link)
    streams = youtube_video.streams.filter(only_audio=True)
    stream = streams.first()
    mp4_video = stream.download(filename='youtube_video.mp4')
    audio_file = open(mp4_video, 'rb')
    output = model.transcribe("youtube_video.mp4")
    transcription = { "title": youtube_video.title.strip(), "transcription": output['text'] }
    pd.DataFrame(transcription).to_csv('./'+video_id+'.csv', sep=";")
    return transcription

def get_answer(question, transcription):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""I am a bot. My name is Fran, and I can have a conversation with another human about this context: """+transcription["title"]+""".
                          I have all the information here: """+transcription["transcription"]+"""
                          I am going to answer the question I receive, nothing else.
                          I will give you short and precise answers.
                          I don't add more questions and answers, just answer the question asked by the human.
                          I do not repeat myself.
                          Only answer one question at a time, no more.
                          If something is not clear, I can infer it, but I must make it clear to you.
                          My favorite word is "NICE".
                          I will add at the end of my answer: NIIIIICE!!
                          When I finish answering the question, I do not add more information. """
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

    result = qa.run(question)
    return result


html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

with st.sidebar:
    st.markdown("""
    # About 
    SecondSelf can extend your memory beyond the limits.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    - Enter the video of your memories that you want to consult.
    - Ask the question you want and interact with yourself.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)

# Configure the UI
image = Image.open('logo.PNG')
st.set_page_config(page_title="Second Self")

st.image(image, caption='Second Self')

st.markdown("""
### Elevate every decision with the power of undiluted memory. Your life, our lens, no moment missed.
""")

youtube_video = st.text_input("Video of your memory: ", placeholder="https://youtu.be/...")

if youtube_video:
    st.video(youtube_video)

large_text = st.text_area("Ask your question: ", placeholder="What ...?")

if st.button("Get an answer"):
    transcription = transcript_video(youtube_video)
    answer = get_answer(large_text, transcription)
    st.write("Answer: ", answer)

    audio = generate(
        text=answer,
        voice="Bella",
        model='eleven_monolingual_v1'
    )

    save(audio, "./output.mp3")
    audio_file = open('./output.mp3', 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/mpeg')

    audio_nice_file = open('./nice.ogg', 'rb')
    audio_nice_bytes = audio_nice_file.read()
    st.audio(audio_nice_bytes, format='audio/ogg')


