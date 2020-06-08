"""Transmit frames to another instance of this program.

The purpose of this module is to share frames collected during training
with another instance of this program. For example, we could send the
frames collected while training the agent and reuse them to train the
feature extractor. In fact, these two nets will usually be trained in
different machines.

The same port must be used for both ends.
"""

import sys
import queue
import threading
import time
import socket
from socketserver import TCPServer, BaseRequestHandler
import gym
import numpy as np
import cv2

from atarieyes.tools import QuitWithResources


class Sender:
    """Generic sender class.

    The sender is a server that runs asynchronously and sends any data
    that is passed to its send function. This never close. I'm expecting the
    program terminates with a Ctrl-C, as often happens in long trainings.
    """

    QUEUE_SIZE = 20

    def __init__(self, msg_length, port, wait=False):
        """Initialize.

        :param msg_length: the fixed length of messages (bytes).
        :param port: (int) a port to use for incoming requests.
        :param wait: if False, a send() returns immediately; if True,
            send() waits if there are too many messages still to be sent.
        """

        self.MSG_LENGTH = msg_length

        # Create connection
        self.server = Sender.OneRequestTCPServer(
            ("0.0.0.0", port), Sender.RequestHandler)

        # Data to send
        self._data_queue = queue.Queue(self.QUEUE_SIZE if wait else 0)
        self.server._data_queue = self._data_queue

    def start(self):
        """Start sending messages on queue."""

        # Finally close
        def close():
            self.server.server_close()
            print("\nSender closed")
        QuitWithResources.add("sender", close)

        thread = threading.Thread(target=self.server.serve_forever)
        thread.daemon = True
        thread.start()

        while not self.server.is_serving:
            time.sleep(0.1)

    def send(self, data):
        """Send data asynchronously.

        :param data: binary data
        :return: True if the data was correctly pushed to the sending queue
        """

        # Checks
        if not isinstance(data, bytes):
            raise TypeError("Can only send bytes")
        if len(data) != self.MSG_LENGTH:
            raise ValueError("Message with the wrong length")
        if not self.server.is_serving:
            return False

        # Send
        self._data_queue.put(data, block=True)
        return True

    class OneRequestTCPServer(TCPServer):
        """Restrict to only one connection."""

        request_queue_size = 1
        allow_reuse_address = True
        is_serving = False

        def handle_error(self, request, client_address):
            """Stop the server on broken connection."""

            print("Broken connection", file=sys.stderr)
            self.server_close()
            self.is_serving = False

        def serve_forever(self):
            """Forward."""

            self.is_serving = True
            TCPServer.serve_forever(self)

    class RequestHandler(BaseRequestHandler):
        """This actually sends data to the client who requested."""

        def handle(self):
            """Send."""

            while True:
                data = self.server._data_queue.get(block=True, timeout=None)
                try:
                    self.request.sendall(data)

                except ConnectionError:
                    break


class Receiver:
    """Generic receiver class."""

    QUEUE_SIZE = 20

    def __init__(self, msg_length, ip, port, wait=False):
        """Initialize.

        :param msg_length: the fixed length of messages (bytes)
        :param ip: ip address of the sender (str)
        :param port: port of the sender (int)
        :param wait: if True, if receive() is not called often enough,
            it no longer accepts new messages. This is only useful with a
            Sender that also waits.
        """

        self.MSG_LENGTH = msg_length

        # Create connection
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))

        # Received data
        self._data_queue = queue.Queue(self.QUEUE_SIZE if wait else 0)

    def start(self):
        """Start receiving messages to a queue."""

        thread = threading.Thread(target=self._always_receive)
        thread.daemon = True
        thread.start()
        self.receiving_thread = thread

    def _always_receive(self):
        """Continuously receive data; internal use."""

        while True:

            # Receive a complete message
            chunks = []
            remaining_bytes = self.MSG_LENGTH
            while remaining_bytes > 0:

                # Read
                chunk = self.sock.recv(min(remaining_bytes, 2048))
                if chunk == b"":
                    print("Closed", file=sys.stderr)
                    self._data_queue.put(None, block=True)  # Signal EOT
                    return
                chunks.append(chunk)

                remaining_bytes -= len(chunk)

            # Return
            msg = b"".join(chunks)
            self._data_queue.put(msg, block=True)

    def receive(self, wait=False):
        """Return a message received.

        :param wait: if true, waits until a complete message is received
        :return: a bytes object containing the message, or None if there are
            no new messages
        :raises: ConnectionAbortedError at the end of transmission
        """

        if not wait and self._data_queue.empty():
            return None

        # note: this is safe, because i must be the only consumer
        msg = self._data_queue.get(block=wait, timeout=None)

        if msg is None:
            raise ConnectionAbortedError("End Of Transmission")

        return msg


class AtariFramesSender(Sender):
    """Transmitter class for frames of Atari games.

    Each message contains the frame and a termination flag:
        "continue": frames will follow
        "last": this is the last
        "repeated_last": this is a repeated last frame
            (late detection of last frame)
    """

    # Port for this stream
    port = 30013

    # Message protocol
    _termination_flags = {
        "continue": b'\x00',
        "last": b'\x01',
        "repeated_last": b'\x02',
    }

    def __init__(self, env_name):
        """Initialize.

        :param env_name: a gym environment name
        """

        # Discover frame shape
        env = gym.make(env_name)
        frame = env.observation_space.sample()
        assert frame.dtype == np.uint8
        size = len(frame.tobytes()) + 1

        # Super
        Sender.__init__(self, msg_length=size, port=self.port, wait=True)

        # Start
        self.start()
        print("> Serving frames on", self.server.server_address, end="")
        print("    (pause)", end="")
        input()   # Leave some time to connect

    def send(self, frame, termination):
        """Send a frame.

        :param data: a numpy array
        :param termination: termination flag. Must be one of
            "continue", "last", "repeated_last".
        """

        msg = frame.tobytes() + self._termination_flags[termination]
        Sender.send(self, msg)


class AtariFramesReceiver(Receiver):
    """Receiver class for frames of Atari games."""

    def __init__(self, env_name, ip):
        """Initialize.

        :param env_name: a gym environment name
        :param ip: source ip address (str)
        """

        # Discover frame shape
        env = gym.make(env_name)
        frame = env.observation_space.sample()
        size = len(frame.tobytes()) + 1
        self.frame_shape = frame.shape

        # Consistent termination flags
        AtariFramesReceiver._termination_flags = {
            AtariFramesSender._termination_flags[key]: key
            for key in AtariFramesSender._termination_flags.keys()
        }

        # Super
        Receiver.__init__(
            self, msg_length=size,
            ip=ip, port=AtariFramesSender.port, wait=True,
        )

        # Start
        self.start()

    def receive(self, wait=False):
        """Return a received frame of the game.

        See Receiver.receive.

        :return: frame and termination flag
        """

        # Get
        data = Receiver.receive(self, wait)
        if data is None:
            return None

        # Split
        frame_data, termination_data = data[:-1], data[-1:]

        # To frame
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = np.reshape(frame, self.frame_shape)

        # Termination
        termination = self._termination_flags[termination_data]

        return frame, termination


class StateRewardSender(Sender):
    """Sends an integer state and a reward.

    These informations are usually provided by the Restraining Bolt to the
    RL agent.
    """

    # Port for this stream
    port = 30014

    # Store static information
    _format = [
        np.array(0, dtype=np.int32),
        np.array(0.0, dtype=np.float32),
    ]
    _msg_len = sum((len(d.tobytes()) for d in _format))

    def __init__(self):
        """Initialize."""

        # Super
        Sender.__init__(
            self, msg_length=self._msg_len, port=self.port, wait=True)

        # Start
        self.start()
        print(
            "> Serving (state, reward) on", self.server.server_address,
            "   (pause)", end="")
        input()   # Leave some time to connect

    def send(self, state, reward):
        """Send a message.

        :param state: a scalar int
        :param reward: a float reward
        """

        # Serialize
        inputs = [state, reward]
        msg = [
            np.array(data, dtype=field.dtype).tobytes()
            for data, field in zip(inputs, self._format)
        ]
        msg = b''.join(msg)

        # Send
        Sender.send(self, msg)


class StateRewardReceiver(Receiver):
    """Receiver class for StateRewardSender."""

    # Store static
    _fields_len = [len(d.tobytes()) for d in StateRewardSender._format]
    _fields_start = [int(i) for i in np.cumsum(_fields_len) - _fields_len[0]]

    def __init__(self, ip):
        """Initialize.

        :param ip: source ip address (str)
        """

        # Super
        Receiver.__init__(
            self, msg_length=StateRewardSender._msg_len,
            ip=ip, port=StateRewardSender.port, wait=True,
        )

        # Start
        self.start()

    def receive(self):
        """Return a received message.

        If a message was not received, it waits.

        :return: See the relative sender for the message format
        """

        # Get
        data = Receiver.receive(self, wait=True)

        # Parse
        fields = [
            data[start:start + length]
            for start, length in zip(self._fields_start, self._fields_len)
        ]
        data = [
            np.reshape(
                np.frombuffer(field, dtype=fformat.dtype), fformat.shape)
            for field, fformat in zip(fields, StateRewardSender._format)
        ]

        return data


def display_atari_frames(env_name, ip):
    """Display the frames of an Atari games in a window.

    The frames should be produced by AtariFramesSender.
    This is an application that never returns.

    :param env_name: a gym environment name
    :param ip: source ip address (str)
    """

    # Receiver
    receiver = AtariFramesReceiver(env_name, ip)
    name = env_name + " - " + ip

    # Window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    width, height = receiver.frame_shape[0:2]
    scale = 4
    cv2.resizeWindow(name, width*scale, height*scale)

    # Reproduce
    while True:

        # Parse frame
        try:
            frame, _ = receiver.receive(wait=True)
        except ConnectionAbortedError:
            break

        # Show
        frame = np.flip(frame, 2)  # opencv uses BGR
        cv2.imshow(name, frame)
        cv2.waitKey(5)

    # Close
    cv2.destroyAllWindows()
