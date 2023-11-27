# quickchat

A simple cli to chat with HF models, using [TGI](https://github.com/huggingface/text-generation-inference.git), gradio, and slurm. The CLI automatically spins up a tgi instance via slurm and then opens a gradio interface to chat with the model.

### Chat with models

Modify `tgi_template.slurm` to use your own slurm account. Then run the following command to chat with a model:

```shell
pip install -e . # or `poetry install`

# use the public tgi instance
python quickchat.py  --endpoint https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1

# spin up your own tgi instance via slurm
python quickchat.py --manage_tgi_instances --model mistralai/Mistral-7B-Instruct-v0.1 --revision main
```

https://github.com/vwxyzjn/quickchat/assets/5555347/8b561a52-1bdc-48f0-bb36-6d39836c061b

### Why

There are already pretty good solutions like [FastChat](https://github.com/lm-sys/FastChat). This repo is a simpler alternative with minimal code allowing researchers to customize and quickly spin up their own tgi instances and chat with them.
