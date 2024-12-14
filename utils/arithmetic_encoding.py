from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pywt
import numpy as np
import matplotlib.pyplot as plt
import glymur
import cv2

class ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFF
        self.encoded_stream = []

    def encode(self, data, probabilities):
        """
        Encode a sequence of binary data using arithmetic coding.
        """
        for symbol in data:
            range_width = self.high - self.low + 1
            symbol_low, symbol_high = probabilities[symbol]
            self.high = self.low + (range_width * symbol_high // 65536) - 1
            self.low = self.low + (range_width * symbol_low // 65536)

            while True:
                if self.high < 0x8000:  # MSB is 0
                    self.encoded_stream.append(0)
                    self._shift()
                elif self.low >= 0x8000:  # MSB is 1
                    self.encoded_stream.append(1)
                    self._shift()
                    self.low -= 0x8000
                    self.high -= 0x8000
                elif self.low >= 0x4000 and self.high < 0xC000:  # Underflow condition
                    self.low -= 0x4000
                    self.high -= 0x4000
                    self._shift()
                else:
                    break

    def _shift(self):
        """
        Handle the bit shift during encoding.
        """
        self.low = (self.low << 1) & 0xFFFF
        self.high = ((self.high << 1) | 1) & 0xFFFF

    def get_encoded_stream(self):
        """
        Get the encoded stream after completing the encoding process.
        """
        return self.encoded_stream
    

def apply_arithmetic_coding(bit_streams):
    """
    Apply Arithmetic Coding to the given bit streams.

    Parameters:
        bit_streams (list of lists): Bit streams generated from EBCOT.

    Returns:
        list: Compressed streams after arithmetic coding.
    """
    probabilities = {0: (0, 32768), 1: (32768, 65536)}  # Example probabilities for binary data
    encoder = ArithmeticEncoder()
    compressed_streams = []

    for stream in bit_streams:
        # Flatten the stream to ensure it is a 1D list of bits
        if isinstance(stream, (np.ndarray, list)):
            flattened_stream = np.array(stream).flatten().astype(int).tolist()
        else:
            raise ValueError(f"Unsupported stream type: {type(stream)}")

        encoder.encode(flattened_stream, probabilities)
        compressed_streams.append(encoder.get_encoded_stream())
        encoder = ArithmeticEncoder()  # Reset the encoder for the next stream

    return compressed_streams