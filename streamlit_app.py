from dataclasses import dataclass
import datetime
import difflib
from typing import List, Literal, TypeAlias
import streamlit as st
import nbformat
from pathlib import Path
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam
)
import subprocess
import tomli_w

#st.set_page_config(layout="wide")

ChatRole: TypeAlias = Literal["user", "assistant", "system"]

@dataclass
class TimedMessage:
    """A message with a timestamp."""
    role: ChatRole
    content: str
    timestamp: datetime.datetime

    def __init__(self, role: ChatRole, content: str):
        self.role = role
        self.content = content
        self.timestamp = datetime.datetime.now()

    def as_openai_message(self) -> ChatCompletionMessageParam:
        """Convert to OpenAI message format."""
        if self.role == "user":
            return ChatCompletionUserMessageParam(
                role="user",
                content=self.content
            )
        elif self.role == "assistant":
            return ChatCompletionAssistantMessageParam(role="assistant", content=self.content)
        elif self.role == "system":
            return ChatCompletionSystemMessageParam(role="system", content=self.content)
        else:
            raise ValueError(f"Invalid role: {self.role}")

@st.cache_resource
def openai_client():
    """Initialize OpenAI client."""
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai_client = OpenAI(api_key=openai_api_key)
    return openai_client


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
def get_starter_and_diff(notebook_quarto, n_context_lines: int) -> tuple[str, str]:
    # Find a starter notebook that most closely matches the uploaded notebook
    starter_notebooks = all_starters()
    diffs = [
        (starter, unified_diff(notebook_quarto, starter_quarto, n_context_lines=2))
        for starter, starter_quarto in starter_notebooks.items()
    ]

    # Sort by size of the diff
    diffs.sort(key=lambda x: len(x[1]))
    closest_starter, closest_starter_diff = diffs[0]

    # Redo the diff with the desired number of context lines.
    starter_quarto = starter_notebooks[closest_starter]
    new_diff = unified_diff(notebook_quarto, starter_quarto, n_context_lines=n_context_lines)

    return closest_starter, new_diff

def main():
    st.title("Notebook Feedback Assistant")

    uploaded_notebook = st.file_uploader("Upload your notebook", type=["ipynb"])
    if uploaded_notebook is None:
        st.stop()

    # Read the uploaded notebook
    uploaded_notebook.seek(0)
    notebook = nbformat.read(uploaded_notebook, as_version=4)
    notebook_quarto = notebook_to_quarto(notebook)

    closest_starter, closest_starter_diff = get_starter_and_diff(notebook_quarto, n_context_lines=9999)

    with st.expander("Show your notebook", expanded=False):
        st.write(f"Diff between your notebook and the starter notebook, `{closest_starter}`:")
        st.code(closest_starter_diff, language="diff")

    starting_prompt = f"""
<document title="Diff with Starter Notebook">
{closest_starter_diff}
</document>
"""
    with st.expander("Show prompt (debug only)", expanded=False):
        st.write("system prompt:")
        st.code(system_prompt(), language="markdown")
        st.write("user prompt:")
        st.code(starting_prompt, language="markdown")
    
    # Reset if a new notebook gets uploaded
    messages: List[TimedMessage] = st.session_state.get("messages", [])
    should_restart = (
        'messages' not in st.session_state
          or messages[1].content != starting_prompt
          or (len(messages) > 2 and st.button("Restart conversation")))
    if should_restart:
        st.session_state.messages = messages = [
            TimedMessage("system", system_prompt()),
            TimedMessage("user", starting_prompt),
            TimedMessage("assistant", get_first_followup_prompt()),
        ]

    for message in messages[2:]:
        with st.chat_message(message.role):
            st.markdown(message.content)

    client = openai_client()
    
    if prompt := st.chat_input("Type your message here (in your own words, no AI please)"):
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append(TimedMessage("user", prompt))


        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[message.as_openai_message() for message in messages],
                    temperature=0.7,
                    max_tokens=1500,
                    stream=True
                )
            response = st.write_stream(stream)
            if not isinstance(response, str):
                # There's some situations where the response is not a string
                # Hack: convert the response to a string
                response = str(response)
            messages.append(TimedMessage("assistant", response))
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
            starter=closest_starter,
        ), multiline_strings=True)
        
        st.download_button(
            label="Download conversation",
            data=downloadable,
            file_name="conversation.txt",
            mime="text/plain"
        )


main()
