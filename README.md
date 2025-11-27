AgroSense: The Curious Farming Assistant

AgroSense is an AI-powered farming assistant that helps farmers make informed decisions. It supports multi-turn conversational interactions with voice input/output, and provides intelligent crop recommendations based on soil and weather conditions. Built with Streamlit and LangChain, it leverages OpenAI's API to engage users in natural dialogue.

Features

Conversational AI: Multi-turn chat interface that retains context across questions for a more natural dialogue.

Voice Input & Output: Accepts voice queries (speech-to-text) and responds with audio (text-to-speech) for hands-free interaction.

Crop Recommendations: Suggests optimal crops for planting based on user-provided soil properties and weather conditions, using a knowledge base derived from agricultural data.

Tech Stack

Streamlit: Web application framework for the interactive UI.

LangChain: Framework for building LLM-powered applications and managing conversational flows.

OpenAI API: Large language model backend (e.g. GPT-3.5) for natural language understanding and response generation.

Python & Libraries: Python for core logic, plus libraries for speech recognition and text-to-speech to enable voice features.

Dataset

This project uses the Crop Recommendation Dataset from Kaggle as the basis for its crop suggestion feature. The data was simplified by averaging the soil and climate parameters for each crop type to inform the recommendation engine.

Setup

Clone the repository: git clone https://github.com/yourusername/AgroSense.git (replace with the actual repo URL).

Install dependencies: Navigate into the project folder and run pip install -r requirements.txt.

Configure API Key: Obtain an OpenAI API key and set it as an environment variable (e.g. OPENAI_API_KEY) or add it to the app configuration.

Run the app: Start the Streamlit app with streamlit run app.py, then open the local URL it provides (usually http://localhost:8501) in your browser.

Usage Example

Once the app is running, you can interact through text or voice:

Ask a question or describe your farming scenario (e.g. "What crop grows well in acidic soil with high rainfall?").

AgroSense will analyze the input and reply with a recommendation. For instance, it might suggest planting rice and explain that rice thrives in waterlogged, acidic conditions.

You can continue the conversation, ask follow-up questions, or clarify details. The assistant maintains context over multiple turns, making the exchange feel natural and informative.
