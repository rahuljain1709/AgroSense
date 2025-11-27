AgroSense: The Curious Farming Assistant

Team STAR – A multi-turn, voice-enabled farming chatbot assistant that provides personalized crop recommendations and agronomy advice.

Project Overview

AgroSense is a Streamlit-based web application that helps farmers make informed decisions about what crops to plant. It leverages OpenAI GPT models and the LangChain/LangGraph framework to enable a conversational AI assistant that can engage in multi-turn dialogue. Farmers can interact with AgroSense through either text or voice (including Hindi, Hinglish, and English), and the assistant will respond with helpful advice. By combining real data on soil nutrients and weather with GPT’s natural language understanding, AgroSense can recommend suitable crops for given conditions and answer follow-up questions. The app maintains persistent conversation memory, so it remembers context across the chat for a more natural, consultative experience.

Features

Intelligent Crop Recommendations: Suggests the best crop to cultivate based on soil parameters (like NPK nutrient levels, pH) and weather conditions (temperature, humidity, rainfall). The assistant analyzes user-provided data and matches it to optimal crop requirements.

Voice and Text Interaction (Multilingual): Supports both voice input and typed queries. Users can speak in Hindi, Hinglish, or English, or type their questions, making the tool accessible to a diverse user base.

Speech-to-Text and Text-to-Speech Integration: Incorporates speech recognition to convert spoken questions into text, and uses text-to-speech to read out the assistant’s replies. This hands-free interaction allows farmers to use the app in the field without needing to read or type.

Multi-Turn Conversation with Memory: Powered by GPT and LangChain’s memory capabilities, the assistant engages in a natural back-and-forth dialogue. It retains conversation context and previously provided information (persistent memory), enabling follow-up questions and continuous consultation without losing track of earlier details.

User-Friendly Interface: Built with Streamlit, the web app provides an intuitive interface with clear prompts, buttons to record audio, and real-time display of the conversation. Farmers with minimal tech experience can easily use AgroSense.

Tech Stack

Streamlit – Used to develop the interactive web application UI for easy deployment and sharing.

LangChain & LangGraph – Libraries for structuring the AI agent’s workflow. LangChain handles prompt chaining and conversational memory, while LangGraph provides a framework for building resilient, stateful agent flows.

OpenAI GPT models – Large language models (e.g. GPT-3.5/4) that power the chatbot’s natural language understanding and response generation.

Python – The core programming language used for all backend logic, data processing, and integration of components.

streamlit-mic-recorder – A Streamlit component for capturing microphone input and performing speech-to-text. This enables voice query input directly within the app.

ChromaDB – Employed as a vector store for embeddings. It stores conversation context or reference data (such as encoded crop profiles) to enable quick similarity searches and memory retrieval during the conversation.

Dataset

The assistant’s recommendations are informed by the Crop Recommendation Dataset from Kaggle
github.com
. This dataset contains 2,200 samples of farming observations with labels for 22 different crops (e.g. rice, maize, apple, cotton, etc.)
github.com
. Each sample includes several soil and climate features – Nitrogen content, Phosphorus content, Potassium content, Temperature (°C), Humidity (%), pH of soil, and Rainfall (mm) – along with the optimal crop to grow under those conditions
github.com
github.com
.

For this project, we preprocessed the dataset by grouping the data by crop type and calculating the average value of each feature for that crop. This resulted in a profile of typical soil and weather conditions for each of the 22 crops. These averaged crop profiles are stored in the vector database (ChromaDB) and used by the chatbot to compare against user-provided conditions. When a user asks for a recommendation, the assistant can retrieve the crop profile closest to the given soil/weather parameters and use GPT to formulate a recommendation based on that information.
