# Atari Eyes

This project is composed of two parts:
- Feature extraction: extract relevant fluents from each frame of the game.
- Reinforcement learning agent: an agent learns the optimal strategy.

Initially, the RL agent learns directly from each frame of the game, receiving standard rewards. Meanwhile, a feature extractor is learned. The purpose is to use these features to reward additional behaviours, later on.

### Commands

The RL agent can be trained with:
```
python3 -m atarieyes agent train -e environment_name
```
To train the feature extractor, instead:
```
python3 -m atarieyes features train -e environment_name
```
