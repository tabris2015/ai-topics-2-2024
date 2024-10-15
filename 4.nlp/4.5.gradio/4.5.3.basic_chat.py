import gradio as gr
import random

def random_reponse(message, history):
    return random.choice(["SI", "NO", "TAL VEZ"])

demo = gr.ChatInterface(random_reponse, type="messages")

demo.launch()