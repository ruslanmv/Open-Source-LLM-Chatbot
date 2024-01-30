from huggingface_hub import hf_hub_download
import logging
import sys
import gradio as gr
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

MODELS_PATH = "./models"

mistral_model_path = hf_hub_download(
    repo_id= "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    resume_download=True,
    cache_dir=MODELS_PATH,)



"""Step 3 : if you use GPU then make sure ( n_gpu_layers":1) at least 1, you can increase or decrease it based on your GPU performance"""

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=mistral_model_path,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


def model_initialization(model):
    if(model !=""):
       gr.Info("model downloading and configuration process has been started, please wait...")
    MODELS_PATH = "./models"
    repo_id=""
    filename=""
    if(model=="Llama-2-13B-chat"):
      repo_id="TheBloke/Llama-2-13B-chat-GGUF"
      filename="llama-2-13b-chat.Q4_K_M.gguf"
    elif(model=="Mistral-7B-Instruct-v0.2") :
      repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
      filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    elif(model=="zephyr-7B-beta"):
      repo_id="TheBloke/zephyr-7B-beta-GGUF "
      filename="zephyr-7b-beta.Q4_K_M.gguf"
    elif(model=="vicuna-7B-v1.5"):
      repo_id="TheBloke/vicuna-7B-v1.5-GGUF"
      filename="vicuna-7b-v1.5.Q4_K_M.gguf"
    elif(model=="Falcon-7B-Instruct"):
      repo_id="TheBloke/Falcon-7B-Instruct-GGML"
      filename="falcon-7b-instruct.ggccv1.q4_1.bin"
    elif(model=="CodeLlama-7B"):
      repo_id="TheBloke/CodeLlama-7B-GGUF"
      filename="codellama-7b.Q4_K_M.gguf"
    else:
      gr.Warning("please select at least one model")


    mistral_model_path = hf_hub_download(
    repo_id= repo_id,
    filename= filename,
    resume_download=True,
    cache_dir=MODELS_PATH,)

    llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=mistral_model_path,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
    gr.Info("model has been configured and ready to chat")
    return "model has been configured and ready to chat, your current model is "+model

def predict(message, history):
    messages = []
    answer = []
    response = llm.stream_complete(message)
    for bot_response in response:
        token = bot_response.delta
        answer.append(token)
        final_answer = " ".join(answer)
        yield final_answer

with gr.Blocks() as UI:

         models=gr.Dropdown(["CodeLlama-7B","Llama-2-13B-chat","Falcon-7B-Instruct" "Mistral-7B-Instruct-v0.2", "zephyr-7B-beta",
                       "vicuna-7B-v1.5"],value=["CodeLlama-7B","Llama-2-13B-chat","Falcon-7B-Instruct" "Mistral-7B-Instruct-v0.2", "zephyr-7B-beta",
                       "vicuna-7B-v1.5"], label="please select at least one model", info="default model is Mistral-7B-Instruct-v0.2")
         textInfo = gr.Textbox(value="current model is Mistral-7B-Instruct-v0.2",label="Model Status");
          # Chatbot interface
         chatUI= gr.ChatInterface(
                            predict,
                            title="Open Source LLM ChatBot",
                            description="Ask any question",
                            theme="soft",
                            examples=["Hello", "are you LLM model?", "how can i finetune a pre-trained LLM model?","How can i build a chatbot using local open-souce LLM ?"],
                            cache_examples=False,
                            submit_btn="Send Message",
                            retry_btn=None,
                            undo_btn="Delete Previous",
                            clear_btn="Clear",
                        )

         models.change(fn=model_initialization,inputs=[models],outputs=[textInfo])

if __name__ == "__main__":
    UI.launch(debug=True) # 