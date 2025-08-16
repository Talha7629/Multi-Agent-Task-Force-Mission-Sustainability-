# app.py
import streamlit as st
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import random

from agno.agent import Agent
from agno.models.groq import Groq
from agno.team.team import Team
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.hackernews import HackerNewsTools

# --- Load environment variables ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- Optional: path to example dataset ---
data_path = Path(__file__).parent / "sample_air_quality.csv"
if not data_path.exists():
    df_mock = pd.DataFrame({
        "Year": [2021, 2022, 2023],
        "PM2.5": [55, 48, 42],
        "CO2": [400, 395, 390]
    })
    df_mock.to_csv(data_path, index=False)

# --- Agent Definitions ---
news_analyst = Agent(
    name="📰 News Analyst",
    role="Track recent news on sustainability initiatives",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[GoogleSearchTools()],
    instructions="Find the most relevant city-level green projects from the past year and summarize key findings.",
    show_tool_calls=True,
    markdown=True,
)

def analyze_dataset():
    df = pd.read_csv(data_path)
    summary = df.describe(include='all').to_string()
    trends = []
    if "PM2.5" in df.columns:
        trends.append(f"🌫️ **Average PM2.5:** {df['PM2.5'].mean():.2f} μg/m³")
    if "CO2" in df.columns:
        trends.append(f"🌍 **Average CO2:** {df['CO2'].mean():.2f} ppm")
    trends_text = "\n".join(trends)
    return f"📊 **Dataset Summary:**\n```\n{summary}\n```\n\n**Environmental Trends:**\n{trends_text}"

data_analyst = Agent(
    name="📊 Data Analyst",
    role="Analyze environmental datasets",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[],
    instructions="Read the provided air quality dataset and summarize trends.",
    show_tool_calls=False,
    markdown=True,
)

policy_reviewer = Agent(
    name="📜 Policy Reviewer",
    role="Summarize government sustainability policies",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[GoogleSearchTools()],
    instructions="Find official city government sources and summarize their recent sustainability policy changes.",
    show_tool_calls=True,
    markdown=True,
)

innovation_scout = Agent(
    name="💡 Innovations Scout",
    role="Discover cutting-edge green tech ideas",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[HackerNewsTools(), GoogleSearchTools()],
    instructions="Search for innovative urban sustainability technologies and describe them in detail.",
    show_tool_calls=True,
    markdown=True,
)

# --- Task Force Team ---
sustainability_team = Team(
    name="🌐 Sustainability Task Force",
    mode="collaborate",
    model=Groq(id="qwen/qwen3-32b"),
    members=[news_analyst, data_analyst, policy_reviewer, innovation_scout],
    instructions=["Collaborate to produce a comprehensive sustainability proposal for the city."],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit UI Config ---
st.set_page_config(page_title="🌱 Mission Sustainability", page_icon="🌎", layout="wide")

# --- Full Dark Theme Styling ---
page_bg_color = """
<style>
    [data-testid="stAppViewContainer"]{
        background-color:#121212 !important;
        color: #f5f5f5 !important;
    }
    aside[data-testid="stSidebar"],
    section[data-testid="stSidebar"]{
        background: #000000 !important;
        border-right: 2px solid #222 !important;
        color: #f5f5f5 !important;
    }
    [data-testid="stSidebar"] *{
        color: #f5f5f5 !important;
    }
    .agent-box {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        margin-top: 10px;
        color: #f5f5f5;
    }
    .banner {
        padding: 12px 16px;
        border-radius: 10px 10px 0 0;
        font-size: 18px;
        font-weight: 700;
        color: #fff;
        display: flex; align-items: center; gap: 10px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.5);
    }
    .news-analyst-banner { background-color: #1976d2; }
    .data-analyst-banner { background-color: #fb8c00; }
    .policy-reviewer-banner { background-color: #388e3c; }
    .innovation-scout-banner { background-color: #8e24aa; }
    .sustainability-team-banner { background-color: #d32f2f; }
    textarea, input, select {
        background-color: #1e1e1e !important;
        color: #f5f5f5 !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        border-radius: 10px; font-weight: 700;
        background: linear-gradient(90deg,#43a047,#2e7d32);
        color: #fff; border: none; padding: .6em 1.2em;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg,#2e7d32,#1b5e20);
        transition: .3s;
    }
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

# --- Title ---
st.markdown(
    "<h1 style='text-align:center; color:#4caf50; text-shadow: 0 0 10px #4caf50;'>🌱 "
    "<span style='color:white; text-shadow: 0 0 8px #ffffff;'>Multi-Agent Task Force</span>"
    ": Mission Sustainability 🌍</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("🛠️ Mission Control")

# Live Mission Clock
st.sidebar.markdown("### ⏳ Mission Time")
st.sidebar.markdown(f"**{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

# Divider
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Choose Your Operative")

# Agent Choice
agent_choice = st.sidebar.radio(
    "Select your operative:",
    ("📰 News Analyst", "📊 Data Analyst", "📜 Policy Reviewer", "💡 Innovations Scout", "🌐 Full Task Force")
)

# Sustainability Fact
facts = [
    "🌍 Every ton of recycled paper saves 17 trees.",
    "💡 Renewable energy creates 5x more jobs than fossil fuels.",
    "🌱 Urban forests can cool cities by up to 5°C.",
    "🚰 Saving one liter of water also saves energy used to pump it.",
    "♻️ Recycling aluminum saves 95% of the energy needed to make new aluminum."
]
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌟 Sustainability Fact")
st.sidebar.info(random.choice(facts))

# --- Map agent choice ---
if agent_choice == "📰 News Analyst":
    selected_agent = news_analyst
    banner_class = "news-analyst-banner"
    banner_icon = "📰"
elif agent_choice == "📊 Data Analyst":
    selected_agent = data_analyst
    banner_class = "data-analyst-banner"
    banner_icon = "📊"
elif agent_choice == "📜 Policy Reviewer":
    selected_agent = policy_reviewer
    banner_class = "policy-reviewer-banner"
    banner_icon = "📜"
elif agent_choice == "💡 Innovations Scout":
    selected_agent = innovation_scout
    banner_class = "innovation-scout-banner"
    banner_icon = "💡"
else:
    selected_agent = sustainability_team
    banner_class = "sustainability-team-banner"
    banner_icon = "🌐"

# --- Main Input ---
topic = st.text_area(
    "🎯 Enter mission target (topic/city):",
    placeholder="Example: Renewable energy transition in Karachi"
)

# --- Launch Button ---
if st.button("🚀 Launch Mission", use_container_width=True):
    if topic.strip():
        mission_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"📅 Mission Start Time: **{mission_time}**")
        with st.spinner(f"🔍 Deploying {agent_choice}... please stand by..."):
            try:
                if agent_choice == "📊 Data Analyst":
                    result_text = analyze_dataset()
                    st.markdown(
                        f"<div class='banner {banner_class}'>{banner_icon} {agent_choice} Report</div>"
                        f"<div class='agent-box'>{result_text}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    result = selected_agent.run(topic)
                    if result and hasattr(result, "content"):
                        st.markdown(
                            f"<div class='banner {banner_class}'>{banner_icon} {agent_choice} Report</div>"
                            f"<div class='agent-box'>{result.content}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("⚠️ No content returned from the agent.")
            except Exception as e:
                st.error(f"💥 Mission Error: {e}")
    else:
        st.warning("✏️ Please specify your mission target first.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:white;'>🚀 Developed for <b>Mission Sustainability</b> Lab | Powered by Multi-Agent AI</p>",
    unsafe_allow_html=True
)
