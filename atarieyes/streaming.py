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

from atarieyes.pytools import QuitWithResources


# Port used for streaming frames
app_port = 30003


class Sender:
    """Generic sender class.

    The sender is a server that runs asynchronously and sends any data
    that is passed to its send function. This never close. I'm expecting the
    program terminates with a Ctrl-C, as often happens in long trainings.
    """

    def __init__(self, msg_length):
        """Initialize.

        :param msg_length: the fixed length of messages (bytes)
        """

        self.MSG_LENGTH = msg_length

        # Create connection
        self.server = Sender.OneRequestTCPServer(
            (socket.gethostname(), app_port), Sender.RequestHandler)

        # Data to send
        self._data_queue = queue.Queue()
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
        self._data_queue.put_nowait(data)
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

    def __init__(self, msg_length, ip):
        """Initialize.

        :param msg_length: the fixed length of messages (bytes)
        :param ip: ip address of the sender
        """

        self.MSG_LENGTH = msg_length

        # Create connection
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, app_port))

        # Received data
        self._data_queue = queue.Queue()

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
                    self._data_queue.put_nowait(None)
                    return
                chunks.append(chunk)

                remaining_bytes -= len(chunk)

            # Return
            msg = b"".join(chunks)
            self._data_queue.put_nowait(msg)

    def get_wait(self):
        """Waits until a complete message is received.

        :return: a bytes object containing the message or None at
            the end of transmission
        """

        return self._data_queue.get(block=True, timeout=None)


class AtariFramesSender(Sender):
    """Transmitter class for frames of Atari games."""

    def __init__(self, env_name):
        """Initialize.

        :param env_name: a gym environment name
        """

        # Discover frame shape
        env = gym.make(env_name)
        frame = env.observation_space.sample()
        assert frame.dtype == np.uint8
        size = len(frame.tobytes())

        # Super
        Sender.__init__(self, size)

        # Start
        self.start()
        print("> Serving frames on", self.server.server_address, end="")
        print("  (pause)", end="")
        input()   # 

    def send(self, frame):
        """Send a frame.

        :param data: a numpy array
        :return: True if the data was correctly pushed to the sending queue
        """

        Sender.send(self, frame.tobytes())


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
        size = len(frame.tobytes())
        self.frame_shape = frame.shape

        # Super
        Receiver.__init__(self, size, ip)

        # Start
        self.start()


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
    window = cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    width, height = receiver.frame_shape[0:2]
    scale = 4
    cv2.resizeWindow(name, width*scale, height*scale)

    # Reproduce
    while True:

        # Parse frame
        buff = receiver.get_wait()
        if buff is None:
            break
        frame = np.frombuffer(buff, dtype=np.uint8)
        frame = np.reshape(frame, receiver.frame_shape)

        # Show
        cv2.imshow(name, frame)
        cv2.waitKey(5)

    # Close
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Just testing here

    if sys.argv[1] == "s":
        sender = AtariFramesSender("Breakout-v4")
        env = gym.make("Breakout-v4")
        run = True
        frame = env.reset()
        while run:
            sender.send(frame)
            input()
            frame, _, _, _ = env.step(env.action_space.sample())

    elif sys.argv[1] == "r":
        display_atari_frames("Breakout-v4", "127.0.1.1")
