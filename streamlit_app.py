import difflib
import streamlit as st
import nbformat
from pathlib import Path
from openai import OpenAI
import subprocess

#st.set_page_config(layout="wide")


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


def notebook_to_quarto(nb):
    """Convert notebook to Quarto markdown format."""
    chunks = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            chunks.append(cell.source)
        elif cell.cell_type == 'code':
            chunks.append(f"```{{python}}\n{cell.source}\n```")
    return '\n\n'.join(chunks)


def unified_diff(notebook: str, starter: str) -> str:
    return ''.join(
        difflib.unified_diff(
            starter.splitlines(keepends=True),
            notebook.splitlines(keepends=True),
            fromfile="starter.ipynb",
            tofile="your_notebook.ipynb",
        )
    )


@st.cache_resource
def get_starter_and_diff(notebook_quarto) -> tuple[str, str]:
    # Find a starter notebook that most closely matches the uploaded notebook
    diffs = [
        (starter, unified_diff(notebook_quarto, starter_quarto))
        for starter, starter_quarto in all_starters().items()
    ]

    # Sort by size of the diff
    diffs.sort(key=lambda x: len(x[1]))
    closest_starter, closest_starter_diff = diffs[0]
    return closest_starter, closest_starter_diff

def main():
    st.title("Notebook Diff Helper")

    uploaded_notebook = st.file_uploader("Upload your notebook", type=["ipynb"])
    if uploaded_notebook is None:
        st.stop()

    # Read the uploaded notebook
    uploaded_notebook.seek(0)
    notebook = nbformat.read(uploaded_notebook, as_version=4)
    notebook_quarto = notebook_to_quarto(notebook)

    closest_starter, closest_starter_diff = get_starter_and_diff(notebook_quarto)

    with st.expander("Show your notebook", expanded=False):
        st.write(f"Diff between your notebook and the starter notebook, `{closest_starter}`:")
        st.code(closest_starter_diff, language="diff")

    starting_prompt = f"""
<document title="My Notebook">
{notebook_quarto}
</document>

<document title="Diff with Starter Notebook">
{closest_starter_diff}
</document>
"""
    with st.expander("Show prompt", expanded=False):
        st.write("system prompt:")
        st.code(system_prompt(), language="markdown")
        st.write("user prompt:")
        st.code(starting_prompt, language="markdown")
    
    # Reset if a new notebook gets uploaded
    messages = st.session_state.get("messages", [])
    if 'messages' not in st.session_state or messages[1]['content'] != starting_prompt:
        st.session_state.messages = messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": starting_prompt}
        ]

    if len(messages) > 2 and st.button("Restart conversation"):
        messages[2:] = []

    for message in messages[2:]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    client = openai_client()
    

    if len(messages) == 2:
        if not st.button("Generate initial feedback"):
            st.stop()
    else:
        prompt = st.chat_input("Type your message here...")
        if not prompt:
            st.stop()
        messages.append({"role": "user", "content": prompt})


    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                stream=True
            )
        response = st.write_stream(stream)
        messages.append({"role": "assistant", "content": response})
        st.session_state.messages = messages
        st.rerun() # ask for a prompt again


main()
