from collections import defaultdict
from dataclasses import dataclass
import datetime
import difflib
import json
from typing import List, Literal, TypeAlias, Dict, Any
import streamlit as st
import nbformat
from pathlib import Path
from google import genai
from google.genai import types as genai_types
import subprocess
import tomli_w
import toml
from pydantic import BaseModel

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
    ratio: float
    revision: str


@st.cache_resource
def genai_client():
    """Initialize OpenAI client."""
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
    )


class RubricResponse(BaseModel):
    """A response to a rubric check."""
    item: str
    status: Literal["pass", "not yet", "not applicable"]
    comment: str

DATA_DIR = Path("data")
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)
REPO_DIR = DATA_DIR / "cs-375-376-public"

def get_updated_repo():
    if not REPO_DIR.exists():
        print("Cloning the cs-375-376-public repo...")
        subprocess.run(["git", "clone", "https://github.com/Calvin-Data-Science/cs375-376-public", str(REPO_DIR)], check=True)
    else:
        print("Updating the cs-375-376-public repo...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=False)

STARTERS_DIR = REPO_DIR / "notebooks"
OBJECTIVES_FILE = REPO_DIR / "course_objectives.yaml"

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
    base_system_prompt = (DATA_DIR / "system_prompt.md").read_text()

    # Add a compact version of the course objectives
    objectives = get_course_objectives()
    fields = ["id", "pillar", "description"]
    objectives_str = '\n'.join(
        f'{objective["id"]}: {objective["description"]}'
        for objective in objectives
        if objective['class'] != '375' # Don't include 375 objectives
    )
    return base_system_prompt + "\n\nFor reference, the course objectives are:\n\n" + objectives_str



@st.cache_resource
def get_first_followup_prompt():
    return (DATA_DIR / "first_followup_prompt.md").read_text()

@st.cache_resource
def get_course_objectives():
    """Read course objectives from YAML file."""
    # Each objective is a dict: {"id", "pillar", "class", "section", "description"}
    import yaml
    with open(OBJECTIVES_FILE, 'r') as f:
        return yaml.safe_load(f)

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


def get_file_history(path: Path) -> List[str]:
    """Return list of commit SHAs that touched this file."""
    rel = str(path.relative_to(REPO_DIR))
    out = subprocess.run(
        ["git", "-C", str(REPO_DIR), "log", "--pretty=format:%H", "--", rel],
        check=True, stdout=subprocess.PIPE, text=True
    )
    return out.stdout.strip().splitlines()


def get_file_content_at_rev(path: Path, rev: str) -> nbformat.NotebookNode:
    """Read the notebook file at a specific revision."""
    rel = str(path.relative_to(REPO_DIR))
    out = subprocess.run(
        ["git", "-C", str(REPO_DIR), "show", f"{rev}:{rel}"],
        check=True, stdout=subprocess.PIPE, text=True
    )
    return nbformat.reads(out.stdout, as_version=4)


@st.cache_resource
def all_starter_versions() -> Dict[str, Dict[str, str]]:
    """
    For each starter notebook (by stem), map every commit SHA to its quarto text.
    """
    get_updated_repo()
    versions: Dict[str, Dict[str, str]] = {}
    for ipynb in STARTERS_DIR.glob("*.ipynb"):
        name = ipynb.stem
        versions[name] = {}
        for rev in get_file_history(ipynb):
            nb = get_file_content_at_rev(ipynb, rev)
            versions[name][rev] = notebook_to_quarto(nb)
    return versions


@st.cache_resource
def get_starter_and_diff(notebook_quarto: str, n_context_lines: int) -> Diff:
    """
    Find the (starter, revision) whose version has maximal similarity to `notebook_quarto`.
    """
    versions = all_starter_versions()
    best: Diff | None = None
    best_ratio = -1.0
    for name, rev_map in versions.items():
        for rev, starter_quarto in rev_map.items():
            # compute similarity ratio
            ratio = difflib.SequenceMatcher(None, notebook_quarto, starter_quarto).ratio()
            if ratio <= best_ratio:
                continue
            # only compute the diff for the new best match
            diff_text = unified_diff(notebook_quarto, starter_quarto, n_context_lines)
            nlines = len(diff_text.splitlines())
            best_ratio = ratio
            best = Diff(
                src_name=name,
                dst_name="your_notebook.ipynb",
                diff=diff_text,
                n_changed_lines=nlines,
                ratio=ratio,
                revision=rev
            )
    assert best is not None, "No starter notebooks found"
    return best


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


def do_rubric_check(rubric, starting_prompt):
    prompt = f"""
{starting_prompt}

<document title="Rubric">
{rubric}
</document>

Check the notebook against the rubric."""

    client = genai_client()
    response = client.models.generate_content(
        model=GENAI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[RubricResponse]
        )
    )

    # try to parse
    try:
        if not isinstance(response.text, str):
            raise ValueError("Response is not a string")
        list_of_rubric_responses = json.loads(response.text)
        if not isinstance(list_of_rubric_responses, list):
            raise ValueError("Response is not a list")
        rubric_responses = [RubricResponse.model_validate(item) for item in list_of_rubric_responses]
    except json.JSONDecodeError as e:
        st.error(f"Error parsing rubric response: {e}")
    except Exception as e:
        st.error(f"Error parsing rubric response: {e}")
        st.write("Raw response:")
        st.code(response.text)
        return

    status_to_emoji = {
        "pass": "✅",
        "not yet": "❌",
        "not applicable": "⏳"
    }
    md = ''
    for rubric_item in rubric_responses:
        emoji = status_to_emoji.get(rubric_item.status, "❓")
        md += f"- {emoji} **{rubric_item.item}**\n"
        if rubric_item.comment:
            md += f"  - {rubric_item.comment}\n"
    st.markdown(md)
        


def notebook_feedback():
    st.title("Notebook Feedback Assistant")

    uploaded_notebook = st.file_uploader("Upload your notebook", type=["ipynb"])
    if uploaded_notebook is None:
        st.stop()

    client = genai_client()
    
    # Read the uploaded notebook
    notebook = nbformat.read(uploaded_notebook, as_version=4)
    notebook_quarto = notebook_to_quarto(notebook)
    num_lines = len(notebook_quarto.split('\n'))

    diff = get_starter_and_diff(notebook_quarto, n_context_lines=9999)
    is_likely_based_on_starter = diff.ratio > 0.5
    is_lab = st.checkbox("Is this a lab notebook? (i.e., based on a starter notebook)?", value=is_likely_based_on_starter)
    if is_lab:
        st.write(f"It looks like this notebook was based on `{diff.src_name}` at `{diff.revision[:6]}`. If this is not correct, please uncheck the box above.")

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
            
    if st.checkbox("Rubric-check mode?", value=False):
        rubric = st.text_area("Rubric", height=200)
        if st.button("Check against rubric"):
            do_rubric_check(rubric, starting_prompt)
            
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
            st.session_state.messages = messages[:]

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
