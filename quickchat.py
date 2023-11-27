from dataclasses import dataclass, field
import os
import shlex
import signal
import sys
import subprocess
import time
from typing import Generic, Iterable, List, Tuple, Type, TypeVar, Union
import uuid
import gradio as gr
from huggingface_hub import InferenceClient, get_session
import requests
from transformers import HfArgumentParser, AutoTokenizer
DataclassT = TypeVar("DataclassT")


@dataclass
class Args:
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    """the name of the HF model or path to use"""
    revision: str = "main"
    """the revision of the model to use"""
    stop_sequences: List[str] = field(default_factory=lambda: ["User:", "###", "<|endoftext|>", "</s>"])
    """the stop sequences to use"""
    slurm_template_path: str = "tgi_template.slurm"
    """Slurm template file path"""
    manage_tgi_instances: bool = True
    """Spin up and terminate TGI instances when the generation is done"""
    endpoint: str = ""
    """TGI endpoint"""


class MyHfArgumentParser(HfArgumentParser, Generic[DataclassT]):
    """A custom HfArgumentParser that supports type annotation overrides."""
    def __init__(self, dataclass_types: Union[Type[DataclassT], Iterable[Type[DataclassT]]], **kwargs):
        super().__init__(dataclass_types, **kwargs)
    def parse_args_into_dataclasses(self, **kwargs) -> Tuple[DataclassT, ...]:
        return super().parse_args_into_dataclasses(**kwargs)


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = fd.communicate()
    return_code = fd.returncode
    if return_code != 0:
        raise ValueError(f"Command failed with error: {errors.decode('utf-8')}")
    return output.decode("utf-8").strip()


def load_endpoints(endpoint_val: Union[str, List[str]], num_instances: int = 1) -> List[str]:
    """ Return list of endpoints from either a file or a comma separated string.
    It also checks if the endpoints are reachable.
    
    Args:
        endpoint_val (Union[str, List[str]]): either a file path or a comma separated string
        num_instances (int, optional): number of instances. Defaults to 1.
    
    Returns:
        List[str]: list of endpoints (e.g. ["http://26.0.154.245:13120"])
    """
    endpoints = None
    while endpoints is None:
        try:
            if endpoint_val.endswith(".txt"):
                endpoints = open(endpoint_val).read().splitlines()
            else:
                endpoints = endpoint_val.split(",")
            assert len(endpoints) == num_instances # could read an empty file
            # due to race condition (slurm writing & us reading)
        except Exception as e:
            print(f"Attempting to load endpoints... error: {e}")
            time.sleep(10)
    print("obtained endpoints", endpoints)
    for endpoint in endpoints:
        connected = False
        while not connected:
            try:
                response = get_session().get(f"{endpoint}/health")
                print(f"Connected to {endpoint}")
                connected = True
            except requests.exceptions.ConnectionError:
                print(f"Attempting to reconnect to {endpoint}...")
                time.sleep(10)
    return endpoints
        

if __name__ == "__main__":
    parser = MyHfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    assert isinstance(args, Args)

    if args.manage_tgi_instances:
        os.makedirs("slurm/logs", exist_ok=True)
        with open(args.slurm_template_path) as f:
            slurm_template = f.read()
        filename = str(uuid.uuid4())
        slurm_path = os.path.join("slurm", f"{filename}.slurm")
        slurm_host_path = os.path.join("slurm", f"{filename}_host.txt")
        slurm_template = slurm_template.replace(r"{{slurm_hosts_path}}", slurm_host_path)
        slurm_template = slurm_template.replace(r"{{model}}", args.model)
        slurm_template = slurm_template.replace(r"{{revision}}", args.revision)
        with open(os.path.join("slurm", f"{filename}.slurm"), "w") as f:
            f.write(slurm_template)
        job_id = run_command(f"sbatch --parsable {slurm_path}")
        print(f"Slurm Job ID: {job_id}")
        def cleanup_function(signum, frame):
            run_command(f"scancel {job_id}")
            print(f"TGI instances terminated")
            sys.exit(0)
        signal.signal(signal.SIGINT, cleanup_function)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, cleanup_function) # Handle termination requests
        endpoints = load_endpoints(slurm_host_path, 1)
    else:
        endpoints = load_endpoints(args.endpoint, 1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    client = InferenceClient(model=endpoints[0])
    STOP_SEQ = ["<|endoftext|>"]

    def completion_inference(message, _):
        partial_message = ""
        
        for token in client.text_generation(
            message,
            max_new_tokens=200,
            stream=True,
            stop_sequences=STOP_SEQ,
        ):
            partial_message += token
            yield partial_message

    def chat_infernece(message, history):
        chat = []
        if len(history) > 0:
            history = history[0]
            print("history", history)
            for i in range(len(history)):
                if i % 2 == 0:
                    chat.append({"role": "user", "content": history[i]})
                else:
                    chat.append({"role": "assistant", "content": history[i]})
        chat.append({"role": "user", "content": message})
        print(chat)
        partial_message = ""
        for token in client.text_generation(
            tokenizer.apply_chat_template(chat, tokenize=False),
            max_new_tokens=200,
            stream=True,
            stop_sequences=STOP_SEQ,
        ):
            partial_message += token
            yield partial_message


    demo = gr.TabbedInterface(
        [
            gr.Interface(
                completion_inference,
                inputs=gr.Textbox(placeholder="Type something...", lines=4),
                outputs=gr.Textbox(placeholder="", container=False, lines=10),
                title="Gradio 🤝 TGI",
                examples=["Are tomatoes vegetables?"],
            ),
            gr.ChatInterface(
                chat_infernece,
                chatbot=gr.Chatbot(height=300),
                textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
                title="Gradio 🤝 TGI",
                examples=["Are tomatoes vegetables?"],
                retry_btn="Retry",
                undo_btn="Undo",
                clear_btn="Clear",
            )
        ],
        tab_names=["Completion", "Chat"],
    ).queue().launch(share=True)
