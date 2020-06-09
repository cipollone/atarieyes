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
      "regions": {
        "blue": {
          "abbrev": "b",
          "fluents": [
            "b_fluent1",
            "b_fluent2"
          ],
          "region": [
            8,
            88,
            152,
            94
          ]
        }
      },
      "constraints": [],
      "restraining_bolt": []
    }

"_frame" contains the coordinates of a large crop of the entire frame
(possibly hiding unrelevant parts). "regions" is a dict of informations,
each region and it's definition. In this case, "blue" is the name chosen
for the only defined region (it could be any name). "region" holds
the coordinates for its crop. "fluents" holds a list of symbols. Each of them
is a propositional atom that is evaluated on this region. All flents must
be named with a prefix of the region ("abbrev") + underscore + any name.
"constraints" is a list of temporal LDLf formulae that declares how
the fluents you define are expected to change and how are they related with
each other. All the expessions are merged and joined in a single conjunction.
I keep them as a list here just because it might be easier to read them
(use parentheses if uncertain).
"restraining_bolt" is a list of temporal LDLf expressions on fluents, just like
contraints. Their purpose is to specify agent behaviours to be rewarded.
This field is not needed for the pure features extraction.
See features.rb.RestrainingBolt docstring for more help about this field.


The selector tool of this module allows to write all fields except "fluents",
"constraints" and "restraining_bolt" which you can easily add manually to the
json file.
"""

import gym
import json
import os
from collections import OrderedDict
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
    Terminate the selection with <esc> or <enter> on the selector window,
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
    regions = {
        name: {
            "abbrev": abbrev,
            "region": selection,
            "fluents": [],
        } for name, abbrev, selection in zip(
            selection_names, selection_abbrev, selections)
    }
    data = OrderedDict([
        ("_frame", [box[0], box[1], box[0] + box[2], box[1] + box[3]]),
        ("regions", regions),
        ("constraints", []),
        ("restraining_bolt", []),
    ])

    # Save
    env_file = os.path.join(info_dir, env.spec.id + ".json")
    os.makedirs(info_dir, exist_ok=True)
    with open(env_file, "w") as f:
        json.dump(data, f, indent=2)


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
