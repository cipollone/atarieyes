"""Transmit frames to another instance of this program.

The purpose of this module is to share frames collected during training
with another instance of this program. For example, we could send the
frames collected while training the agent and reuse them to train the
feature extractor. In fact, these two nets will usually be trained in
different machines.

The same port must be used for both ends.
"""

import queue
import socket
import socketserver

# TODO: multithreading
# TODO: use queue to send
# TODO: only push to queue if someone is listening
# TODO: classes for frames


class Sender:
    """Generic sender class.

    The sender is a server that starts when created and sends any data
    that is passed to its send function.
    """

    def __init__(self, address):
        """Initialize.

        :param address: ip address of the stream
        """

        # Data to send
        self.data = queue.Queue()

        # Create connection
        ip, port = address.split(":")
        port = int(port)
        socketserver.TCPServer.allow_reuse_address = True    # socket option
        self.server = socketserver.TCPServer((ip, port), Sender.RequestHandler)

    def start(self):
        """Start sending messages on queue."""

        with self.server:
            self.server.serve_forever()

    class RequestHandler(socketserver.BaseRequestHandler):
        """This actually sends data to the client who requested."""

        def handle(self):
            """Send."""

            with self.request:

                # TODO: placeholder
                while True:
                    self.request.sendall(b"0123456789\n")
                    input()


class Receiver:
    """Generic receiver class."""

    def __init__(self, address, msg_length):
        """Initialize.

        :param address: ip address of the stream
        :param msg_length: the fixed length of messages (bytes)
        """

        self.MSG_LENGTH = msg_length

        # Create connection
        ip, port = address.split(":")
        port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Start
        self.sock.connect((ip, port))

    def read_wait(self):
        """Wait and read for received data.

        :return: Returns a binary message when available.
        """

        # Read an entire message
        chunks = []
        remaining_bytes = self.MSG_LENGTH
        while remaining_bytes > 0:

            # Read
            chunk = self.sock.recv(min(remaining_bytes, 2048))
            if chunk == b"":
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)

            remaining_bytes -= len(chunk)

        # Return
        msg = b"".join(chunks)
        return msg


if __name__ == "__main__":
    # TODO: testing here. Delete when done
    import sys
    if sys.argv[1] == "s":
        sender = Sender("localhost:30003").start()
    elif sys.argv[1] == "r":
        receiver = Receiver("localhost:30003", 3000)
        while True:
            print(receiver.read_wait())
