import streamlit as st
from crewai import LLM, Agent, Task, Crew
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Homework Helper AI",
    page_icon="📚",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    color: #000000;
}
    </style>
""", unsafe_allow_html=True)

# Initialize LLM with Gemini 2.5 Flash
@st.cache_resource
def get_llm():
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2000,
        temperature=0.3
    )

# Initialize search tool
@st.cache_resource
def get_search_tool():
    return SerperDevTool()

# Title and description
st.title("📚 Homework Helper AI")
st.markdown("### Your personal AI tutor for all subjects!")
st.markdown("---")

# Sidebar for API key input
with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Enter your Groq API key here"
    )
    serper_api_key = st.text_input(
        "Serper API Key",
        type="password",
        value=os.getenv("SERPER_API_KEY", ""),
        help="Enter your Serper API key for web search"
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    
    st.markdown("---")
    st.markdown("💡 **Tip:** You can also set these in your `.env` file")

# Main form for user input
with st.form("homework_form"):
    st.subheader("📝 Enter Your Homework Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input(
            "Your Name",
            placeholder="Enter your name",
            help="We'll personalize the response for you"
        )
    
    with col2:
        subject = st.selectbox(
            "Subject",
            ["Mathematics", "Science", "History", "English", "Geography", 
             "Physics", "Chemistry", "Biology", "Computer Science", "Other"],
            help="Select the subject of your question"
        )
    
    question = st.text_area(
        "Your Homework Question",
        placeholder="Type your homework question here...",
        height=150,
        help="Be as specific as possible for better results"
    )
    
    submitted = st.form_submit_button("🚀 Get Help!")

# Process the request when form is submitted
if submitted:
    if not name:
        st.error("Please enter your name")
    elif not question:
        st.error("Please enter your homework question")
    elif not groq_api_key and not os.getenv("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar")
    else:
        # Show progress
        with st.spinner(f"🔍 Hi {name}! Researching your {subject} question..."):
            try:
                # Get tools
                llm = get_llm()
                search_tool = get_search_tool()
                
                # Agent 1: Researcher
                researcher = Agent(
                    role='Research Expert',
                    goal='Find accurate information for homework',
                    backstory='You are an expert researcher who helps students with accurate and comprehensive homework answers. You use web search to find the most up-to-date information.',
                    llm=llm,
                    tools=[search_tool],
                    verbose=False
                )
                
                # Agent 2: Teacher
                teacher = Agent(
                    role='Friendly Teacher',
                    goal='Explain things simply and clearly',
                    backstory='You are a patient and friendly teacher who explains complex topics in simple, easy-to-understand language suitable for students.',
                    llm=llm,
                    verbose=False
                )
                
                # Tasks
                research_task = Task(
                    description=f'Research and answer this {subject} question: {question}. Provide accurate and comprehensive information.',
                    expected_output='A detailed, accurate answer with relevant facts and information',
                    agent=researcher
                )
                
                explain_task = Task(
                    description=f'Explain the answer to "{question}" to {name} in a simple, friendly, and easy-to-understand way. Make it engaging and educational.',
                    expected_output='A clear, simple explanation that a student can easily understand',
                    agent=teacher,
                    context=[research_task]
                )
                
                # Run Crew
                crew = Crew(
                    agents=[researcher, teacher],
                    tasks=[research_task, explain_task],
                    verbose=False,
                    max_rpm=2
                )
                
                result = crew.kickoff()
                
                # Display result
                st.success("Answer ready!")
                st.markdown("---")
                st.subheader(f"Answer for {name}")
                st.markdown(f"**Subject:** {subject}")
                st.markdown(f"**Question:** {question}")
                st.markdown("---")
                
                # Result box
                st.markdown("### Explanation:")
                st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
                
                # Download button
                result_text = f"Helper AI FOR {name.upper()}\nSubject: {subject}\nQuestion: {question}\n\n{result}"
                st.download_button(
                    label="📥 Download Answer",
                    data=result_text,
                    file_name=f"{name}_helper_ai.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your API keys and try again")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by CrewAI & Groq (Llama 3.3 70B)</p>", unsafe_allow_html=True)
