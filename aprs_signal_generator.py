from typing import List
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile
import argparse


class APRSSignalGenerator:

    preamble_len: int = 8
    """Number of repeats of the start flag."""

    postamble_len: int = 8
    """Number of repeats of the end flag."""

    enable_preemphasis: bool = True
    """Add -6dB/octave preemphasis."""

    f_preemph_3db: float = 2150
    """3dB attenuation point of preemphasis filter, in Hz."""

    f_mark: float = 1200.0
    """Frequency of the mark tone, in Hz."""

    f_space: float = 2200.0
    """Frequency of the space tone, in Hz."""

    f_baud: float = 1200.0
    """Symbol rate, in Hz."""

    f_samp: float = 44100.0
    """Sample rate, in Hz."""

    lf_noise_gaussian_std = 0.03
    """
    Standard deviation of gaussian noise added to audio (CPFSK modulated) signal.
    0.0 to disable. For reference, signal should have <= 1.0 peak to peak (less with pre-emphasis filter).
    This type of noise very, very roughly mirrors Analog-Digital-Converter (ADC) noise.
    It explicitly does not mirror RF noise, due to the way FM works.
    """

    flag: List[int] = [0, 1, 1, 1, 1, 1, 1, 0]
    """AX.25 start/end flag value, as list of bits."""

    @property
    def f_center(self):
        """Frequency of tone centered between mark and space, in Hz."""
        return (self.f_mark + self.f_space) / 2.0

    @property
    def f_delta(self):
        """Frequency deviation from center tone to mark or space tone, in Hz."""
        return abs(self.f_space - self.f_mark) / 2.0

    @property
    def omega_center(self):
        """Angular frequency of tone centered between mark and space, in radians."""
        return 2 * np.pi * self.f_center

    @property
    def omega_delta(self):
        """Angular frequency deviation from center tone to mark or space tone, in radians."""
        return 2 * np.pi * self.f_delta

    def call(self, payload: List[any]) -> List[int]:
        """
        Encode the payload, which is a sequence of bits.
        """
        payload = [int(bool(x)) for x in payload]
        xs = (
            self.preamble_len * self.flag
            + self._stuff_bits(payload)
            + self.postamble_len * self.flag
        )
        xs = self._nrzl_to_nrzi(xs)
        xs = self._cpfsk_modulate(xs)
        if self.enable_preemphasis:
            xs = self._preemphasize(xs)
        xs = np.array(xs)
        xs += np.random.normal(0.0, self.lf_noise_gaussian_std, size=len(xs))
        return xs

    def _preemphasize(self, xs: List[int]) -> List[int]:
        """
        Apply a 6db/octave highpass to the signal, similar to the preemphasis in FM transmitters.
        """
        b, a = signal.butter(
            1, [self.f_preemph_3db], btype="highpass", analog=False, fs=self.f_samp
        )
        return signal.lfilter(b, a, xs)

    def _cpfsk_modulate(self, xs: List[int]) -> List[int]:
        """
        Modulate the signal with Constant Phase Frequency Shift Keying (CPFSK).
        """
        result = []
        t_samp = 1.0 / self.f_samp
        n_samp = int(len(xs) / self.f_baud * self.f_samp)
        integral = 0.0
        for n in range(n_samp):
            t = n * t_samp  # Time at sample in sec
            integral += +1 if xs[int(t * self.f_baud)] else -1
            y = np.cos(
                self.omega_center * t + self.omega_delta * integral / self.f_samp
            )  # Sample value [1]
            result.append(y)
        return result

    def _nrzl_to_nrzi(self, xs: List[int]) -> List[int]:
        """
        Convert the bitvector to NRZI.
        """
        ys = list(range(len(xs)))
        ys[0] = 0
        for i in range(1, len(xs)):
            if xs[i]:
                ys[i] = ys[i - 1]
            else:
                ys[i] = int(not ys[i - 1])
        return ys

    def _stuff_bits(self, xs: List[int]) -> List[int]:
        """
        Apply AX.25 bitstuffing to the bitvector.
        """
        ys = []
        num_ones = 0
        for x in xs:
            ys.append(x)
            if x:
                num_ones += 1
                if num_ones == 5:
                    ys.append(0)
                    num_ones = 0
            else:
                num_ones = 0
        return ys


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--show-graph", dest="show_graph", action="store_true")
    par.add_argument("--graph-out", dest="graph_out", type=str)
    par.add_argument("--audio-out", dest="audio_out", type=str)
    args = par.parse_args()

    gen = APRSSignalGenerator()
    sig = gen.call([0, 0, 0, 0, 0, 0, 0, 1] * 64)

    if args.audio_out:
        soundfile.write(args.audio_out, sig, samplerate=int(gen.f_samp), format="flac")

    if args.graph_out or args.show_graph:
        plt.figure(figsize=(24, 4))
        plt.plot([n for n in range(0, 1000)], sig[0:1000], label="AFSK signal")
        plt.xlabel("Samples")
        plt.xlabel("Amplitude")
        plt.legend()
        if args.show_graph:
            plt.show()
        if args.graph_out:
            plt.savefig(args.graph_out)


if __name__ == "__main__":
    main()
