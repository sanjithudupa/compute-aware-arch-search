## how to set up

this is a [uv](https://github.com/astral-sh/uv) project so you need to have uv installed, so do that if you haven't.

Otherwise you just `uv sync` to get all packages installed and your venv set up.

If this is your first time running, you need to download the tokenizer and the model so just run `python3 download_model.py` and then you're ready.

Rn all the code is configured to use CPU but we will change that once we do the real stuff. 


I have updated the pyproejct.toml - make sure to use the given pytorch and python versions.

Download the flash attention version: 
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl


SANJITH AND VINNY:
1. I have finished doing the layer 1...28 training (hopefully we get some results soon)
2. Please do research on all the FLA linear attention modules because i lowkey don't know which one is the best in terms of throughput and accuracy. RN im just using delta former but i doubt thats the most optimal. 

2. Next thing is that now that we know what layer we are going to replacing, we need to create a way to adaptively create the architecture. Look at config.json.
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],

We need a way to check if "linear_attention" then instead of a MHA block, we put the respective linear attention block.
If you don't see this config.json file, make sure to run download_model.py

3. Then we have to load the weights of each of the blocks respectively but thats not very important right now. (I can work on this)


4. More importantly though, now that we have this self attention/linear attention hybrid, we need to run knolwedge distillation. Best way to usually do this is to run two forward passes hrough the student and teacher and minimize the KL divergence or the MSE between the vocab dsitribution in the final layer. However, if we want to be more scientific, we can find a better way to do such knowledge distillation from prev literature.

i think thats good enoguh. lmk if you have any questions. training lowk fast. 2 hours/layer means that shi will finish in around 2-3 days.