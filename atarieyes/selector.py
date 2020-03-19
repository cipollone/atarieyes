"""Graphical tool for feature selection."""

import gym
import json
import os
import numpy as np
import cv2 as cv

# Directory where selections are saved
info_dir = "envs_data"


def selection_tool(args):
    """Starts a graphical tool for feature selection.
    
    This allows to manually select:
        - The image box, that is the large region where the game is displayed.
            This allows to ignore the useless borders and numbers.
        - A set of local features. A local feature is a region of the game
            which can be in one of two states.

    :param args: namespace of arguments. See --help.
    """

    # Make env and show initial image
    env = gym.make(args.env)
    image0 = env.reset()
    h, w = image0.shape[0:2]

    # Prepare
    f = 4
    image0 = cv.resize(image0, (f*w, f*h))
    selections = []

    # Select outer box
    print("> Select game region")
    box = cv.selectROI("frame", image0)

    # Return if cancelled
    if box[2] == 0 or box[3] == 0:
        return

    # Select features
    print("> Select small local features")
    selection = [0, 0, 1, 1]
    while selection[2] > 0 and selection[3] > 0:
        selection = cv.selectROI("frame", image0)
        selections.append(selection)
    selections.pop()

    # Scale down
    box = np.round(np.array(box) / f).tolist()
    selections = np.round(np.array(selections) / f).tolist()

    # Format data
    data = {}
    data["box"] = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    regions = []
    for selection in selections:
        regions.append(
            [selection[0], selection[1], selection[0] + selection[2],
                selection[1] + selection[3]])
    data["regions"] = regions

    # Save
    env_file = os.path.join(info_dir, env.spec.id + ".json")
    os.makedirs(info_dir, exist_ok=True)
    with open(env_file, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

