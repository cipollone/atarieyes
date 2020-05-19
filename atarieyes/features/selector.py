"""Graphical tool for feature selection.

The selected local features are saved in a json file named after the
environment. This is an example:

    {
      "_frame": [
        8,
        32,
        152,
        197
      ],
      "blue": {
        "abbrev": "b",
        "fluents": {
          "b_fluent1": "<b_fluent1*>end",
          "b_fluent2": "<(b_fluent1; b_fluent2)*>end"
        },
        "region": [
          8,
          88,
          152,
          94
        ]
      }
    }

"_frame" is a special name: these are the coordinates of a large crop of the
entire frame (possibly hiding unrelevant parts).
"blue" is the name chosen for a region (could be any). "region" holds
the coordinates for its crop. "fluents" holds a dict of symbols. Each of them
is a propositional atom with an associated LDLf temporal formula that it
respects. "abbrev" is an abbreviation of the region name, to be used in
"fluents. The fluents section is not filled by the selector, but you can
define those manually.
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
          the game where to extract fluents. After each selection, look at the
          terminal, because the region name and abbreviations are asked.
    Terminate the selection with <esc> or <enter> on the selection tool,
    without any selection.

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
    selection_abbrev = []

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
        selection_abbrev.append(input("> Abbreviation: "))
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
            "abbrev": abbrev,
            "region": selection,
            "fluents": {}
        } for name, abbrev, selection in zip(
            selection_names, selection_abbrev, selections)
    }
    data["_frame"] = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

    # Save
    env_file = os.path.join(info_dir, env.spec.id + ".json")
    os.makedirs(info_dir, exist_ok=True)
    with open(env_file, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


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
