# The Constituency Tree Labeling
## Overview

This is an implementation of an LSTM + CRF model that predicts a tag for each node in the constituency tree. It was completed as part of the 10-618 course at CMU. 
We assume that for each example, the branching structure of the tree is known, but the tags are not. The model is trained on a Penn Tree Bank dataset. 
- Input: An input sentence and the associated skeleton of its constituency parse tree
- Output: The labels of the non terminals in the parse tree

## Usage
1. Clone and navigate to the repository:
    
   ```
   git clone https://github.com/alako/constituency-tree.git
   cd constituency-tree
   ```  
2. Create a Virtual Environment with `venv` or a Conda Environment:
   ```
   conda create --name myenv python=3.10 pip
   conda activate myenv
   ```
3. Run the following command to install dependencies from requirements.txt:
   ```
   pip install -r requirements.txt
   ```
4. Start the training:
   ```
   python tree_bp.py
   ```
