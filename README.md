
# Python Rock Paper Scissors AI

This project uses PyTorch to build a Deep Q network that can predict the next move of an opponent. Both the opponent and player are built as Feed Forward networks with one hidden layer. The player is trained with the DQNagent based on the opponents last move and the opponent network is left initialized to random states.



## Installation
Begin by cloning the github \
\
First, intall the dependencies

```bash
  pip install -r requirements.txt
```


## Usage
To run from the command line: \
Run the eval script
```bash
  python eval.py
```
 *Ctrl+c if you need to cancel*

Now the results will be stored in **RockPaperScissorsEval.PNG** in the clone folder
## Future Improvements


Currently, there is only an evaluation mode which shows the DQN playing against the randomized AI opponent and outputs a showing the score as it changes per episode. I plan to create an implementation where you can play against a fresh AI and train it to predict your moves over time. 

If you implement this on your own, ensure that when your calling to the AI you are passing it your *last* move, otherwise it will just learn the rules of Rock paper scissors and just return the winning move. 

