## how to set up

this is a [uv](https://github.com/astral-sh/uv) project so you need to have uv installed, so do that if you haven't.

Otherwise you just `uv sync` to get all packages installed and your venv set up.

If this is your first time running, you need to download the tokenizer and the model so just run `python3 download_model.py` and then you're ready.

Rn all the code is configured to use CPU but we will change that once we do the real stuff. 


you have to do pip install flash-attn --no-build-isolation 