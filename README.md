# quickchat

A simple cli to chat with HF models, using TGI, gradio, and slurm. The CLI automatically spins up a tgi instance via slurm and then opens a gradio interface to chat with the model.

### Chat with models

Modify `tgi_template.slurm` to use your own slurm account. Then run the following command to chat with a model:

```shell
pip install -e . # or `poetry install`
python quickchat.py --model mistralai/Mistral-7B-Instruct-v0.1 --revision main
```




https://github.com/vwxyzjn/quickchat/assets/5555347/8b561a52-1bdc-48f0-bb36-6d39836c061b


