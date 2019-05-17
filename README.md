# DeepLearningComparison
Comparison of multiple optimisers in Deep Learning

# Installation instructions
1. First make a new Virtual Environment
`python3 -m venv venv`
2. Activate this virtual environment `source venv/bin/activate`
3. Install the required packages `pip3 install -r requirements.txt`

# Run instructions
`script.py` is the most important script in this repository. For the parameters and their meanings run `python3 script.py -h`.

`plot.py` and `earlystopping.py` generate the plots and calculate the point at which early stopping should occur (and outputs this to a table together with the accuracies). These scripts require (Peregrine) log files.