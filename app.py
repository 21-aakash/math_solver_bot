import os  # Import os to interact with the operating system
import streamlit as st  # Import Streamlit for building the web app
from dotenv import load_dotenv  # Import dotenv to load environment variables
from langchain_groq import ChatGroq  # Import ChatGroq for using Groq API with LangChain
from langchain.chains import LLMMathChain, LLMChain  # Import LangChain modules for math and general chains
from langchain.prompts import PromptTemplate  # Import PromptTemplate to create custom prompts
from langchain_community.utilities import WikipediaAPIWrapper  # Import WikipediaAPIWrapper for Wikipedia integration
from langchain.agents.agent_types import AgentType  # Import AgentType to specify the type of agent
from langchain.agents import Tool, initialize_agent  # Import Tool and initialize_agent to create and initialize tools
from langchain.callbacks import StreamlitCallbackHandler  # Import StreamlitCallbackHandler to handle Streamlit callbacks

# Load environment variables from the .env file
load_dotenv()

# Custom CSS to style the app container with neon border effect
st.markdown("""
    <style>
        .neon-container {
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 10px rgba(0, 255, 0, 0.8), 
                        0 0 20px rgba(0, 255, 0, 0.6), 
                        0 0 30px rgba(0, 255, 0, 0.4);
            background-color: #0f0f0f; /* Dark background to highlight the neon effect */
            margin: 20px;
        }

        .main-title {
            color: #00FF00;  /* Neon green color */
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }

        .subheader {
            color: #98FB98;  /* Light green color */
            font-size: 30px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Wrap the entire app inside the neon container
st.markdown('<div class="neon-container">', unsafe_allow_html=True)

# Title and subheader inside the neon box
st.markdown('<h1 class="main-title">👽SkyMath</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="subheader">Your problem solver assistant Google Gemma 2</h4>', unsafe_allow_html=True)

# Load Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# If API key is not provided, show an info message and stop the app
if not groq_api_key:
    st.info("Please add your Groq API key in the .env file to continue")
    st.stop()

# Initialize the ChatGroq model with the provided API key
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Initialize the Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,  # Set the function for the Wikipedia tool
    description="A tool for searching the Internet to find various information on the topics mentioned"
)

# Initialize the Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,  # Set the function for the Calculator tool
    description="A tool for answering math-related questions. Only input mathematical expressions need to be provided"
)

# Create a custom prompt template for reasoning questions
prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation,
and display it point-wise for the question below.
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],  # Define input variables for the prompt
    template=prompt  # Set the prompt template
)

# Create a chain for the reasoning tool
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,  # Set the function for the Reasoning tool
    description="A tool for answering logic-based and reasoning questions."
)

# Combine all tools into an agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Set the agent type
    verbose=False,  # Verbose logging off
    handle_parsing_errors=True  # Handle parsing errors
)

# Initialize session state if messages do not exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a SkyMath chatbot who can answer all your maths questions"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Get user input and start interaction
question = st.text_area("Enter your question:", "")

# Generate response when button is pressed
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):  # Show a spinner while generating response
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            try:
                # Generate and display response from the agent
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.write('### Response:')
                st.success(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")  # Display error if something goes wrong

    else:
        st.warning("Please enter the question")  # Show a warning if no question is entered

# Close the neon container
st.markdown('</div>', unsafe_allow_html=True)
