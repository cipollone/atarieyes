# AtariEyes

The main purpose of this software is to train features extractors for the Atari games. More precisely, after we have defined a set of propositional symbols for a game, we can train a set Boolean functions that compute the appropriate truth value for those symbols. For example, we could train boolean symbols such as: `has_5_lives`, `door_open`, `enemy_dead`, and train a network that actually makes those symbols true whenever the conditions happen to be true (just looking at a single frame).

A second part of the software trains a Reinforcement Learning agent (based on Double-QN). These two parts interact. For example, to train the boolean symbols (fluents) we need to run an agent previously train, because the features extractor must receive the observations produced by the agent. Also, once the fluents have been trained, they could be sent back to train a new version of the agent.

## How

For help on all commands and options call this module as a script with the `--help` flag.

The main commands are:
```
python3 -m atarieyes agent train
python3 -m atarieyes agent play
python3 -m atarieyes features select
python3 -m atarieyes features train
python3 -m atarieyes features rb
```
When starting with a new environment, first run a `select` command. `agent train` and `agent play` train and run a reinforcement learning agent, respectively. `features train` trains the features extractor (requires an instance of `agent play` running). `features rb` sends feedback to an agent that is being trained (see <https://www.dis.uniroma1.it/~degiacom/papers/2019/icaps19dfip.pdf> about training with a Restraining Bolt).
