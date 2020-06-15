"""Play with a trained agent."""

import numpy as np
import gym
import cv2 as cv
import tensorflow as tf
from rl.callbacks import Callback

from atarieyes import tools
from atarieyes.streaming import AtariFramesSender
from atarieyes.agent.training import Trainer, CheckpointSaver


class Player:
    """Play with a trained agent.

    This is useful to visualize the behaviour of a trained agent.
    The agent must be already trained and saved.
    """

    def __init__(self, args):
        """Initialize.

        The agent is reconstructed from a json of saved arguments.
        The weights to restore are loaded from a checkpoint saved by
        Trainer. Usually something like:
            runs/agent/<env_name>/models/weights_<step>.<ext>

        :param args: namespace of arguments; see --help.
        """

        # Store
        self.rendering = args.watch in ("render", "both")
        self.streaming = args.watch in ("stream", "both")

        # Load the arguments
        agent_args = tools.ArgumentSaver.load(args.args_file)

        # Check that this was a trained agent
        if agent_args.what != "agent" or agent_args.op != "train":
            raise RuntimeError(
                "Arguments must represent a `atarieyes agent train ...` "
                "command")

        # Dirs
        model_path, log_path = tools.prepare_directories(
            "agent", agent_args.env, no_create=True)

        # Environment
        self.env = gym.make(agent_args.env)
        self.env_name = agent_args.env

        # Repeatability
        if args.deterministic:
            if "Deterministic" not in self.env_name:
                raise ValueError(
                    "--deterministic only works with deterministic"
                    " environments")
            self.env.seed(30013)
            np.random.seed(30013)
            tf.random.set_seed(30013)

        # Agent
        self.kerasrl_agent, _ = Trainer.build_agent(
            tools.Namespace(
                agent_args, training=False, random_test=args.random_test,
                random_epsilon=args.random_epsilon,
            )
        )

        # Load weights
        saver = CheckpointSaver(
            agent=self.kerasrl_agent, path=model_path,
            interval=agent_args.saves,
        )
        saver.load(args.cont)

        # Callbacks
        self.callbacks = []
        if args.random_epsilon:
            self.callbacks.append(self.kerasrl_agent.test_policy.callback)
        if self.streaming:
            self.callbacks.append(
                Streamer(self.env_name, skip_frames=args.skip, port=args.port))
        if args.record:
            self.callbacks.append(Recorder(self.env_name, args.record))

    def play(self):
        """Play."""

        # Go
        self.kerasrl_agent.test(
            self.env, nb_episodes=10000, visualize=self.rendering,
            callbacks=self.callbacks,
        )


class Streamer(Callback):
    """Send frames through a connection."""

    def __init__(self, env_name, skip_frames=None, port=None):
        """Initialize.

        :param env_name: name of an Atari environment.
        :param skip_frames: skip a random number of frames in [0, skip_frames].
        :param port: if given, overrides the default port.
        """

        # Super
        Callback.__init__(self)

        # Check
        if skip_frames is not None and skip_frames <= 0:
            raise ValueError("skip_frames must be positive")

        # Init
        self.sender = AtariFramesSender(env_name, port=port)
        self.skip_frames = skip_frames
        self._skips_left = 0
        self._last_frame = None

    def on_step_end(self, step, logs={}):
        """Send each frame."""

        # Collect a frame
        if not self.skip_frames or self._skips_left == 0:
            frame = logs["raw_observation"]
            self._last_frame = frame
            self.sender.send(frame, "continue")

        # Update
        if self.skip_frames:
            self._skips_left -= 1
            if self._skips_left <= 0:
                self._skips_left = np.random.randint(0, self.skip_frames+1)

    def on_episode_end(self, episode, logs={}):
        """Singnal the end of an episode."""

        self.sender.send(self._last_frame, "repeated_last")


class Recorder(Callback):
    """Record frames to a video."""

    def __init__(self, env_name, out_file):
        """Initialize.

        :param env_name: name of an Atari environment.
        :param out_file: output file.
        """

        # Super
        Callback.__init__(self)

        # Video Writer
        frame_size = gym.make(env_name).observation_space.shape[:2]
        fourcc = cv.VideoWriter_fourcc(*"HFYU")
        fps = 20.0
        self._writer = cv.VideoWriter(
            filename=out_file, fourcc=fourcc, fps=fps,
            frameSize=(frame_size[1], frame_size[0]), isColor=True,
        )

        # Close properly
        tools.QuitWithResources.add(
            "VideoWriter", lambda: self._writer.release())

    def on_step_end(self, step, logs={}):
        """Write each frame."""

        frame = logs["raw_observation"]
        frame = np.flip(frame, 2)  # opencv uses BGR
        self._writer.write(frame)
