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

# .env file se environment variables ko load karne ke liye
load_dotenv()

# Custom CSS to style the title and headings
st.markdown("""
    <style>
    .main-title {
        color: #FF5733;  /* Bright orange-red */
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .subheader {
        color: #1E90FF;  /* Dodger blue */
        font-size: 30px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app ki settings set karenge
st.markdown('<h1 class="main-title">👽SkyMath</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="subheader">Your problem solver assistant Google Gemma 2</h4>', unsafe_allow_html=True)

# Groq API key ko environment variables se load karenge
groq_api_key = os.getenv("GROQ_API_KEY")

# Agar API key nahi mili, toh user ko message dikhayenge aur app stop karenge
if not groq_api_key:
    st.info("Please add your Groq API key in the .env file to continue")
    st.stop()

# ChatGroq model ko initialize karenge Groq API key ke sath
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Wikipedia tool initialize karenge
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,  # Wikipedia tool ke liye function set karenge
    description="A tool for searching the Internet to find various information on the topics mentioned"
)

# Math tool initialize karenge
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,  # Math tool ke liye function set karenge
    description="A tool for answering math-related questions. Only input mathematical expressions need to be provided"
)

# Custom prompt template banayenge reasoning questions ke liye
prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation,
and display it point-wise for the question below.
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],  # Prompt ke input variables define karenge
    template=prompt  # Prompt ko template ke sath set karenge
)

# Reasoning tool ke liye chain banayenge
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,  # Reasoning tool ke liye function set karenge
    description="A tool for answering logic-based and reasoning questions."
)

# Sabhi tools ko agent ke andar combine karenge
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent type set karenge
    verbose=False,  # Verbose logging off rakhenge
    handle_parsing_errors=True  # Parsing errors ko handle karenge
)

# Session state initialize karenge agar messages pehle se nahi hain
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questions"}
    ]

# Pichle messages ko display karenge
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Interaction start karne ke liye user se question lenge
question = st.text_area("Enter your question:", "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

# Agar user "find my answer" button press kare, toh response generate karenge
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):  # Spinner dikhayenge jab tak response generate ho raha hai
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Response generate karenge agent se aur display karenge
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")  # Agar question nahi diya gaya toh warning dikhayenge
