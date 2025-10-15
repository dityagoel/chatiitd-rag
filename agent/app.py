# app.py

import gradio as gr
import uuid
from agent.agent import invoke_memory_agent

# --- 1. Define the Chatbot Logic for the 'messages' format ---

def add_message_and_clear(message: str, history: list):
    """
    Adds the user's message to the chat history and clears the textbox.
    This function is compatible with type='messages'.
    """
    # Append the new user message as a dictionary to the history
    history.append({"role": "user", "content": message})
    # Return the updated history and a Gradio update to clear the textbox
    return history, gr.update(value="")

def process_and_stream_response(history: list, session_id: str):
    """
    Processes the latest user message and streams the bot's response.
    This function is compatible with type='messages'.
    """
    # Get the user's message from the last dictionary in the history
    user_message = history[-1]["content"]

    # Invoke the agent to get the complete response
    response = invoke_memory_agent(
        {"input": user_message},
        session_id=session_id
    )
    bot_message_full = response["output"]

    # Add a new dictionary for the assistant's response with empty content
    history.append({"role": "assistant", "content": ""})

    # Stream the response by updating the 'content' of the last message
    for char in bot_message_full:
        history[-1]["content"] += char
        yield history

# --- 2. Create the Gradio Interface with gr.Blocks ---

with gr.Blocks(theme=gr.themes.Base(), css="footer {display: none !important}") as demo:
    
    session_id = gr.State(lambda: str(uuid.uuid4()))
    
    gr.Markdown(
        """
        # 🎓 IIT Delhi Academic AI Agent
        Ask me about courses, program structures, or institute rules. 
        The agent uses a local reranker for improved retrieval accuracy.
        """
    )
    
    # Initialize the chatbot with the required type='messages'
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        label="IITD AI Agent",
        bubble_full_width=False,
        height=600,
        avatar_images=(None, "https://i.imgur.com/gIIA43m.png"),
        type='messages' # This is the new, required format
    )
    
    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="e.g., What are the requirements for a minor in CS?",
            container=False,
        )

    # --- 3. Wire Up the Components ---
    # This event chain is now robust for the new message format.
    txt.submit(
        add_message_and_clear, 
        [txt, chatbot], 
        [chatbot, txt], 
        queue=False
    ).then(
        process_and_stream_response, 
        [chatbot, session_id], 
        [chatbot]
    )

# --- 4. Launch the Application ---

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=False)