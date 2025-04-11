from collections import defaultdict
from dataclasses import dataclass
import datetime
import difflib
from typing import List, Literal, TypeAlias, Dict, Any
import streamlit as st
import nbformat
from pathlib import Path
from google import genai
from google.genai import types as genai_types
import subprocess
import tomli_w
import toml

#st.set_page_config(layout="wide")

GENAI_MODEL = 'gemini-2.0-flash'

ChatRole: TypeAlias = Literal["user", "assistant"]

@dataclass
class TimedMessage:
    """A message with a timestamp."""
    role: ChatRole
    content: str
    timestamp: datetime.datetime

    def __init__(self, role: ChatRole, content: str, timestamp:datetime.datetime):
        self.role = role
        self.content = content
        self.timestamp = timestamp

    def as_genai_message(self) -> genai_types.Content:
        """Convert to Gemini GenAI message format."""
        if self.role == "user":
            return genai_types.UserContent(self.content)
        elif self.role == "assistant":
            return genai_types.ModelContent(self.content)
        else:
            raise ValueError(f"Invalid role: {self.role}")


@dataclass
class Diff:
    src_name: str
    dst_name: str
    diff: str
    n_changed_lines: int


@st.cache_resource
def genai_client():
    """Initialize OpenAI client."""
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
    )


DATA_DIR = Path("data")
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)
REPO_DIR = DATA_DIR / "cs-375-376-public"

def get_updated_repo():
    if not REPO_DIR.exists():
        print("Cloning the cs-375-376-public repo...")
        # clone https://github.com/Calvin-Data-Science/cs375-376-public
        subprocess.run(["git", "clone", "https://github.com/Calvin-Data-Science/cs375-376-public", str(REPO_DIR)], check=True)
    else:
        # update the repo
        print("Updating the cs-375-376-public repo...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=False)

STARTERS_DIR = REPO_DIR / "notebooks"

@st.cache_resource
def all_starters():
    """Read all starter notebooks, in Quarto format."""
    get_updated_repo()
    all_starters = {}
    for starter in STARTERS_DIR.glob("*.ipynb"):
        with open(starter, 'r') as f:
            nb = nbformat.read(f, as_version=4)
            all_starters[starter.stem] = notebook_to_quarto(nb)
    return all_starters


@st.cache_resource
def system_prompt():
    return (DATA_DIR / "system_prompt.md").read_text()

@st.cache_resource
def get_first_followup_prompt():
    return (DATA_DIR / "first_followup_prompt.md").read_text()


def notebook_to_quarto(nb):
    """Convert notebook to Quarto markdown format."""
    chunks = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            chunks.append(cell.source)
        elif cell.cell_type == 'code':
            chunks.append(f"```{{python}}\n{cell.source}\n```")
    return '\n\n'.join(chunks)


def unified_diff(notebook: str, starter: str, n_context_lines: int = 3) -> str:
    return ''.join(
        difflib.unified_diff(
            starter.splitlines(keepends=True),
            notebook.splitlines(keepends=True),
            fromfile="starter.ipynb",
            tofile="your_notebook.ipynb",
            n=n_context_lines,
        )
    )


@st.cache_resource
def get_starter_and_diff(notebook_quarto, n_context_lines: int) -> Diff:
    # Find a starter notebook that most closely matches the uploaded notebook
    starter_notebooks = all_starters()
    diffs = [
        (len(unified_diff(notebook_quarto, starter_quarto, n_context_lines=2).split('\n')), starter)
        for starter, starter_quarto in starter_notebooks.items()
    ]

    # Sort by size of the diff
    diffs.sort()
    num_diff_lines, closest_starter = diffs[0]

    # Redo the diff with the desired number of context lines.
    starter_quarto = starter_notebooks[closest_starter]
    new_diff = unified_diff(notebook_quarto, starter_quarto, n_context_lines=n_context_lines)

    return Diff(
        src_name=closest_starter,
        dst_name="your_notebook.ipynb",
        diff=new_diff,
        n_changed_lines=num_diff_lines,
    )


def parse_conversation_file(file_content) -> Dict[str, Any]:
    """Parse a downloaded conversation file"""
    try:
        data = toml.loads(file_content)
        # Convert message dicts back to TimedMessage objects
        if "message" in data:
            data["messages"] = [
                TimedMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=datetime.datetime.fromisoformat(msg["timestamp"])
                )
                for msg in data["message"]
            ]
        return data
    except Exception as e:
        st.error(f"Error parsing conversation file: {e}")
        return {}


def conversation_viewer():
    st.title("Conversation Viewer")
    
    uploaded_file = st.file_uploader("Upload a conversation file", type=["txt"])
    if uploaded_file is None:
        st.stop()
        
    # Read and parse the conversation file
    conversation_data = parse_conversation_file(uploaded_file.read().decode())
    
    if not conversation_data:
        st.error("Failed to parse conversation file.")
        st.stop()
    
    # Display conversation metadata
    st.subheader("Conversation Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Lab notebook:**", conversation_data.get("is_lab", "Unknown"))
    with col2:
        st.write("**Starter notebook:**", conversation_data.get("starter", "N/A"))
    
    # Display reflection if available
    if "reflection" in conversation_data and conversation_data["reflection"]:
        with st.expander("Student Reflection", expanded=True):
            st.markdown(conversation_data["reflection"])
    
    # Display the conversation
    st.subheader("Conversation")
    counts_so_far = defaultdict(int)
    if "messages" in conversation_data:
        messages = conversation_data["messages"]
        for i, message in enumerate(messages):
            is_first_of_type = counts_so_far[message.role] == 0
            counts_so_far[message.role] += 1
            if is_first_of_type and message.role == "user":
                # This is the initial uploaded notebook/diff - show in collapsed expander
                with st.expander("Initial Notebook/Diff", expanded=False):
                    st.markdown(message.content)
                continue
            if is_first_of_type and message.role == "system":
                # This is the system prompt - show in collapsed expander
                with st.expander("System Prompt", expanded=False):
                    st.markdown(message.content)
                continue
                
            with st.chat_message(message.role):
                st.markdown(message.content)
                st.caption(f"Timestamp: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("No conversation messages found in the file.")


def notebook_feedback():
    st.title("Notebook Feedback Assistant")

    uploaded_notebook = st.file_uploader("Upload your notebook", type=["ipynb"])
    if uploaded_notebook is None:
        st.stop()

    # Read the uploaded notebook
    notebook = nbformat.read(uploaded_notebook, as_version=4)
    notebook_quarto = notebook_to_quarto(notebook)
    num_lines = len(notebook_quarto.split('\n'))

    diff = get_starter_and_diff(notebook_quarto, n_context_lines=9999)
    #st.write(f"Debug: {diff.n_changed_lines} lines changed out of {num_lines} lines in the notebook.")
    is_likely_based_on_starter = diff.n_changed_lines < num_lines * .75
    is_lab = st.checkbox("Is this a lab notebook? (i.e., based on a starter notebook)?", value=is_likely_based_on_starter)
    if is_lab:
        st.write(f"It looks like this notebook was based on {diff.src_name}. If this is not correct, please uncheck the box above.")

    with st.expander("Show your notebook", expanded=False):
        if is_lab:
            st.write(f"Diff between your notebook and the starter notebook, `{diff.src_name}`:")
            st.code(diff.diff, language="diff")

            starting_prompt = f"""
<document title="Diff with Starter Notebook">
{diff.diff}
</document>
"""
        else:
            starting_prompt = f"""
<document title="Your Notebook">
{notebook_quarto}
</document>
"""
            
    with st.expander("Show prompt (debug only)", expanded=False):
        st.write("system prompt:")
        st.code(system_prompt(), language="markdown")
        st.write("user prompt:")
        st.code(starting_prompt, language="markdown")
    
    now = datetime.datetime.now()

    # Reset if a new notebook gets uploaded
    messages: List[TimedMessage] = st.session_state.get("messages", [])
    should_restart = (
        'messages' not in st.session_state
          or messages[0].content != starting_prompt
          or (len(messages) > 2 and st.button("Restart conversation")))
    if should_restart:
        st.session_state['system_prompt'] = system_prompt()
        st.session_state.messages = messages = [
            TimedMessage("user", starting_prompt, timestamp=now),
            TimedMessage("assistant", get_first_followup_prompt(), timestamp=now),
        ]

    for message in messages[1:]:
        with st.chat_message(message.role):
            st.markdown(message.content)

    client = genai_client()
    
    if prompt := st.chat_input("Type your message here (in your own words, no AI please). Use Shift-Enter to add a new line."):
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append(TimedMessage("user", prompt, timestamp=now))


        with st.chat_message("assistant"):
            stream = client.models.generate_content_stream(
                    model=GENAI_MODEL,
                    contents=[message.as_genai_message() for message in messages],
                    config=genai_types.GenerateContentConfig(
                        system_instruction=st.session_state['system_prompt'] or '',
                        temperature=0.7,
                        max_output_tokens=1500,
                        top_k=40,
                    )
                )
            response = st.write_stream((chunk.text for chunk in stream))
            if not isinstance(response, str):
                # There's some situations where the response is not a string
                # Hack: convert the response to a string
                response = str(response)
            messages.append(TimedMessage("assistant", response, timestamp=now))
            st.session_state.messages = messages

    with st.expander("Ready to wrap up?", expanded=False):
        st.write("""Please write, very briefly:

1. Your best and worst moments
2. the chatbot's best and worst moments, and
3. any takeaways you have.""")
        reflection = st.text_area("Reflection")

        # Make a downloadable version of the conversation in TOML format
        downloadable = tomli_w.dumps(dict(message=[
                {
                    "role": message.role,
                    "timestamp": message.timestamp.isoformat(),
                    "content": message.content
                }
                for message in messages
            ],
            reflection=reflection,
            is_lab=is_lab,
            starter=diff.src_name if is_lab else '',
        ), multiline_strings=True)
        
        st.download_button(
            label="Download conversation",
            data=downloadable,
            file_name="conversation.txt",
            mime="text/plain"
        )

feedback_page = st.Page(notebook_feedback, title="Notebook Feedback")
viewer_page = st.Page(conversation_viewer, title="Conversation Viewer")

pg = st.navigation([feedback_page, viewer_page])
pg.run()
