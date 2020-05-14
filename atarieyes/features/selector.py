"""Graphical tool for feature selection.

The selected local features are saved in a json file named after the
environment. This is an example that could be generated:
{
    "_frame": [ 8, 32, 152, 197 ],
    "green_bar": {
        "region": [ 8, 81, 152, 87 ],
        "fluents": {
            "symbol": "temporal formula",
            ...
        }
    },
    ...
}
"_frame" is a special name. The coordinates are the large crop of the entire
frame (possibly hiding unrelevant parts).
"green_bar" is the name chosen for a region (could be any). "region" holds
the coordinates for its crop. "fluents" holds a dict of symbols. Each of them
is a propositional atom with an associated LDLf temporal formula.
Symbols are not added by this selection but you can manually define them.
"""

import gym
import json
import os
import numpy as np
import cv2 as cv

# Directory where selections are saved
info_dir = "definitions"


def selection_tool(args):
    """Starts a graphical tool for feature selection.

    This allows to manually select:
        - The image box, that is the large region where the game is displayed.
            This allows to ignore the useless borders and numbers.
        - A set of local features. A local feature is an interesting region of
          the game where to extract fluents.

    :param args: namespace of arguments. See --help.
    """

    # Make env and show initial image
    env = gym.make(args.env)
    image0 = env.reset()
    h, w = image0.shape[0:2]

    # Prepare
    f = 4
    image0 = cv.resize(image0, (f*w, f*h))
    image0 = np.flip(image0, 2)  # opencv uses BGR
    selections = []
    selection_names = []

    # Select outer box
    print("> Select game region")
    box = cv.selectROI("frame", image0)

    # Return if cancelled
    if box[2] == 0 or box[3] == 0:
        return

    # Select features
    print("> Select small local features")
    selection = [0, 0, 1, 1]
    while True:
        selection = cv.selectROI("frame", image0)

        if not (selection[2] > 0 and selection[3] > 0):
            break
        selections.append(selection)
        selection_names.append(input("> Name: "))
    print("")

    # Scale down
    box = np.round(np.array(box) / f).astype(int).tolist()
    selections = np.round(np.array(selections) / f).astype(int).tolist()

    # Selections as coordinates
    selections_ = []
    for selection in selections:
        selections_.append(
            [selection[0], selection[1], selection[0] + selection[2],
                selection[1] + selection[3]])
    selections = selections_

    # Json format
    data = {
        name: {
            "region": selection,
            "fluents": {}
        } for name, selection in zip(selection_names, selections)
    }
    data["_frame"] = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

    # Save
    env_file = os.path.join(info_dir, env.spec.id + ".json")
    os.makedirs(info_dir, exist_ok=True)
    with open(env_file, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def read_back(env_name):
    """Read the json file for the given environment.

    :param env_name: name of a gym environment.
    :return: the loaded json structure.
    :raises: ValueError: if unknown env.
    """

    env_json = os.path.join(info_dir, env_name + ".json")
    if not os.path.exists(env_json):
        raise ValueError(
            "No definitions for " + str(env_name) + ". "
            "Run:\n  atarieyes features select -e " + str(env_name))

    with open(env_json) as f:
        env_data = json.load(f)
    return env_data
