# DSBD
Code for AAAI'25 "[Dynamic-Width Speculative Beam Decoding for Efficient LLM Inference](https://arxiv.org/abs/2409.16560)"

# Environment

See `requirements.txt`. 

Alternatively, you can use docker file `zongyueq/llmss:0.0.2`, then with command `source ~/miniconda3/bin/activate myenv; conda activate myenv;`

Your GPU needs to support `nvidia-smi` to measure GPU energy consumption.

# Data

The SQUAD dataset will be downloaded automatically. To use Spider dataset, download the data from `https://yale-lily.github.io/spider` and uncompress it under the `DSBD` directory. **Make sure the file path in `execution\_accuracy` of `sampling/utils.py` is correct.**

# Example Run

`python evaluation.py --approx_model_name meta-llama/Llama-3.2-1B --target_model_name meta-llama/Llama-3.1-8B --max_tokens 200 --max_seconds 10000 --log_file /llmss/DSBD/logs/tmp.log --dataset squad --top_k=10 --top_p=0.9 --num_inputs=10`

- appox\_model\_name: path of the draft model
- target\_model\_name: path of the target model
- max\_tokens: the number of tokens to generate (values we used: 100, 200)
- max\_seconds: the time limit for each method
- log\_file: path to the log file
- dataset: squad or spider
- top\_k: k for top k sampling (values we used: 10, 20)
- top\_p: p for top p sampling (values we used: 0.8, 0.9)
- num\_inputs: the number of inputs to test (values we used: 100, 200)

To run experiments with MT-Bench, please download its repo and replace the decoding function in "gen\_model\_answer.py" with our "beam\_speculative\_sampling".

# Acknowledgement

The code is forked from "https://github.com/feifeibear/LLMSpeculativeSampling"
