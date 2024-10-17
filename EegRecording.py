"""
File to simulate multiple LSL streams.
Each stream consists of 64 channels with an Ornstein-Uhlenbeckprocess in each (coloured noise).
Added alpha noise at 10 Hz occurs from time to time in O1, O2 and Oz.
"""
import random
import mne
import numpy as np


class Channel:
    """
    Channel class. Creates EEG.
    A channel has a name, a noise source (random white noise), a list of neighbours, and resulting EEG.
    Each time "readEeg" is called, the current EEG value is handed out, each time "next" is called, the channel is
    advanced to the next time point.
    The EEG is constructed from the noise source (high weight), the noise sources of the channel's neighbours
    (lower weights), and the alpha (a shared sine wave).
    """
    _type = "eeg"
    _name = ""
    _neighbours = []
    _currentNoise = 0.0
    _currentEeg = 0.0
    _currentAlpha = 0.0
    _currentAlphaPhase = 0.0        # current alpha phase, used to calculate the sine value for the alpha
    _currentAlphaThreshold = 0.0    # low-pass filter of the eeg, used to decide whether we have alpha or not
    _currentAlphaAmplitude = 0.0
    _ownNoiseFactor = 2.0     # similarity to neighbouring channels (high means non-similar)
    _eegIntensity = 2.0       # general eeg amplitude
    _alphaIntensity = 0.8     # general alpha amplitude, should be lower for non-occipital channels
    _neighbour_alpha_intensity = 1.0   # sum of alpha intensities of all neighbours, plus my own, used for misc/alpha-readout channels to plot alpha value
    _dt = 0.002

    def __init__(self, name, alphaIntensity, eegIntensity, ownNoiseFactor, channel_type='eeg'):
        """ Constructs the channel with a certain name
        :param name: The name of the channel
        """
        self._type = channel_type
        self._name = name
        self._alphaIntensity = alphaIntensity
        self._neighbour_alpha_intensity = alphaIntensity
        self._neighbour_alpha_intensity = 0.0
        self._eegIntensity = eegIntensity
        self._ownNoiseFactor = ownNoiseFactor
        self.next()

    def set_neighbours(self, neighbours):
        """
        Register the channel's neighbours
        :param neighbours: array with pointers to neighbouring channels
        """
        self._neighbour_alpha_intensity = self._alphaIntensity
        for c in neighbours:
            self._neighbours.append(c)
            self._neighbour_alpha_intensity += c.get_alpha_intensity()

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def get_alpha_intensity(self):
        return self._alphaIntensity

    def read_noise(self):
        """ Get the noise value of the channel, used to calculate EEGs of other channels
        :return: the current noise value
        """
        return self._currentNoise

    def read_alpha_amplitude(self):
        """
        :return: The current alpha threshold of the channel
        """
        return self._currentAlphaAmplitude

    def read_eeg(self):
        """ Calculate/read the current EEG value of the channel
        :return: Calculates a floating point value stating the current EEG of the channel.
        """

        # return value
        sample = 0.0

        # standard channels
        if self._type == "eeg":

            # calculate next point in EEG
            current_noise = self._ownNoiseFactor * self._currentNoise
            for nb in self._neighbours:
                current_noise += nb.read_noise() / len(self._neighbours)
            self._currentEeg += 50.0*(-self._currentEeg + current_noise) * self._dt

            if self._alphaIntensity > 0.0:

                # calculate next alpha sine value
                self._currentAlpha = np.sin(self._currentAlphaPhase)     # current phase

                # calculate alpha amplitude
                alpha_change_speed = 8.0    # how fast the state changes
                # the threshold is a slowly filtered version of our current eeg
                self._currentAlphaThreshold += alpha_change_speed \
                                               * (-self._currentAlphaThreshold + 0.1 * self._currentEeg)\
                                               * self._dt
                alpha_steepness = 5000.0    # how steeply the amplitude changes on flips
                # run a sigmoid over the threshold to limit to 0-1
                alphaFactor = 1.0 / (1.0 + np.exp(alpha_steepness * self._currentAlphaThreshold))

                # factor for alpha ranges from 0 to alpha-intensity, factor for eeg is scaled such that
                # alpha-amplitude + eeg-amplitude = eeg-intensity
                self._currentAlphaAmplitude = alphaFactor * self._alphaIntensity
                eegAmplitude = 5.0 * self._eegIntensity * self._eegIntensity / (self._eegIntensity + self._currentAlphaAmplitude)
            else:
                # if alpha-intensity is zero, eeg amplitude is as defined
                eegAmplitude = 5.0 * self._eegIntensity

            # raw eeg is an interpolation between the noise value and the alpha sine wave
            rawEeg = eegAmplitude * self._currentEeg + self._currentAlphaAmplitude * self._currentAlpha
            sample = rawEeg * 5.0e-5

        # readout channels (helper channel to read overall alpha aamplitudes)
        elif self._type == "misc":
            for nb in self._neighbours:
                sample += nb.read_alpha_amplitude()
            sample /= self._neighbour_alpha_intensity

        # return with eeg value or read alpha value
        return sample

    def next(self):
        """ Advance the channel to the next point in time
        Calculates the next noise value and the next alpha value
        """
        self._currentNoise = random.uniform(-1.0, 1.0)
        self._currentAlphaPhase += 62.0*self._dt


class EegRecording:
    """
    A class representing a person's head, consisting of multiple EEG channels
    """
    _info = None
    _data = []
    _samplingFreq = 500
    _samples = 0
    _channels = []
    _dt = 0.002

    def __init__(self, channels, montage):
        """
        Initialise the class
        :param channels: A list of channels, each row is a dict containing the strings "name", "alpha", "eeg", "uniqueness",
        "channel_type" and a list "neighbours" for channel neighbours.
        """
        self._dt = 0.002
        self._channels = 64
        self._samplingFreq = 500.0
        self._channels = []

        # Create all channels
        c_dict = {}    # translates channel name to index into self._channels
        nc = 0     # counter for channels
        for c in channels:
            name = c['name']
            self._channels += [Channel(name, c['alpha'], c['eeg'], c['uniqueness'], channel_type=c['type'])]
            c_dict[name] = nc
            nc += 1

        # Register neighbours with all channels
        for c in channels:
            name = c['name']
            neighbours = [self._channels[c_dict[name]] for name in c['neighbours']]
            self._channels[c_dict[name]].set_neighbours(neighbours)

        channel_names = [c.get_name() for c in self._channels]
        channel_types = [c.get_type() for c in self._channels]
        self._info = mne.create_info(channel_names, ch_types=channel_types, sfreq=1.0 / self._dt)
        self._info.set_montage(montage)
        print(self._info)


    @staticmethod
    def create_1020_19A():
        """
        A static method to create a standard 19 channel cap in 10-20 system plus channel with alpha information
        :return: An EegRecording object
        """

        return EegRecording([
            # frontal
            {'name': 'Fp1', 'alpha': 0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F7', 'F3', 'Fz', 'Fp2']},
            {'name': 'Fp2', 'alpha':  0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fp1', 'Fz', 'F4', 'F8']},
            {'name': 'F7', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T3', 'C3', 'F3', 'Fp1']},
            {'name': 'F3', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F7', 'Fp1', 'Fz', 'C3']},
            {'name': 'Fz', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fp1', 'F3', 'Cz', 'F4', 'Fp2']},
            {'name': 'F4', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fz', 'C4', 'F8', 'Fp2']},
            {'name': 'F8', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T4', 'C4', 'F4', 'Fp2']},
            # central
            {'name': 'T3', 'alpha':  0.05, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F7', 'C3', 'T5']},
            {'name': 'C3', 'alpha':  0.05, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T3', 'F3', 'Cz', 'P3']},
            {'name': 'Cz', 'alpha':  0.05, 'eeg': 1.3, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['C3', 'Fz', 'C4', 'Pz']},
            {'name': 'C4', 'alpha':  0.05, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Cz', 'F4', 'T4', 'P4']},
            {'name': 'T4', 'alpha':  0.05, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T6', 'C4', 'F8']},
            # occipital
            {'name': 'T5', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T3', 'C3', 'P3', 'O1']},
            {'name': 'P3', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T5', 'C3', 'Pz', 'O1']},
            {'name': 'Pz', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P3', 'Cz', 'P4', 'O1', 'O2']},
            {'name': 'P4', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Pz', 'C4', 'T6', 'O2']},
            {'name': 'T6', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['O2', 'P4', 'C4', 'T4']},
            {'name': 'O1', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T5', 'P3', 'Pz', 'O2']},
            {'name': 'O2', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['O1', 'Pz', 'P4', 'T6']},
            # total alpha
            {'name': 'Alpha Total', 'alpha': 0.5, 'eeg': 0.0, 'uniqueness': 0.0, 'type': 'misc', 'neighbours': ['T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']}
        ], 'standard_1020')

    @staticmethod
    def create_1010_32A():
        """
        A static method to create a standard 32 channel cap in 10-10 system plus channel with alpha information
        :return: An EegRecording object
        """
        return EegRecording([

            {'name': 'Fp1', 'alpha': 0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F7', 'FC5', 'F3', 'AFz', 'Fpz']},
            {'name': 'Fpz', 'alpha': 0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fp1', 'F3', 'AFz', 'F4', 'Fp2']},
            {'name': 'Fp2', 'alpha':  0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F8', 'FC6', 'F4', 'AFz', 'Fpz']},
            {'name': 'AFz', 'alpha':  0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fp1', 'Fz', 'Fp2', 'F4', 'Fz', 'F3']},

            {'name': 'F7', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T7', 'FC5', 'F3', 'Fp1']},
            {'name': 'F3', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F7', 'FC5', 'C3', 'FC1', 'Fz', 'AFz', 'Fpz', 'Fp1']},
            {'name': 'Fz', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F3', 'FC1', 'Cz', 'FC2', 'F4', 'AFz']},
            {'name': 'F4', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F8', 'FC6', 'C4', 'FC2', 'Fz', 'AFz', 'Fpz', 'Fp2']},
            {'name': 'F8', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T8', 'FC6', 'F4', 'Fp2']},

            {'name': 'FC5', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T7', 'C3', 'FC1', 'F3', 'F7']},
            {'name': 'FC1', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F3', 'FC5', 'C3', 'Cz', 'Fp2', 'Fz']},
            {'name': 'FC2', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fz', 'FC1', 'Cz', 'C4', 'FC6', 'F4']},
            {'name': 'FC6', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F8', 'F4', 'FC2', 'C4', 'T8']},

            {'name': 'T7', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'CP5', 'C3', 'FC5', 'F7']},
            {'name': 'C3', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F3', 'FC5', 'T7', 'CP5', 'P3', 'CP1', 'Cz', 'FC1']},
            {'name': 'Cz', 'alpha':  0.0, 'eeg': 1.3, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fz', 'FC1', 'C3', 'CP1', 'Cz', 'CP2', 'C4', 'FC2']},
            {'name': 'C4', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F4', 'FC6', 'T8', 'CP6', 'P4', 'CP2', 'Cz', 'FC2']},
            {'name': 'T8', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'FC6', 'C4', 'CP6', 'F8']},

            {'name': 'CP5', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'O1', 'P3', 'CP1', 'C3', 'FC5', 'T7']},
            {'name': 'CP1', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['FC1', 'C3', 'CP5', 'P3', 'POz', 'Pz', 'CP2', 'Cz']},
            {'name': 'CP2', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['FC2', 'C4', 'CP6', 'P4', 'POz', 'Pz', 'CP1', 'Cz']},
            {'name': 'CP6', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'O2', 'P4', 'CP2', 'C4', 'FC6', 'T8']},

            {'name': 'P7', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T7', 'CP5', 'P3', 'O1']},
            {'name': 'P3', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'CP5', 'C3', 'CP1', 'Fz', 'POz', 'Oz', 'O1']},
            {'name': 'Pz', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P3', 'CP1', 'Cz', 'CP2', 'P4', 'POz']},
            {'name': 'P4', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'CP6', 'C4', 'CP2', 'Fz', 'POz', 'Oz', 'O2']},
            {'name': 'P8', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T8', 'CP6', 'P3', 'O2']},

            {'name': 'POz', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['O1', 'P3', 'CP1', 'Fz', 'CP2', 'P4', 'O2', 'Oz']},
            {'name': 'O1', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'CP5', 'P3', 'POz', 'Oz']},
            {'name': 'Oz', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['O1', 'P3', 'POz', 'P4', 'O2']},
            {'name': 'O2', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'CP6', 'P4', 'POz', 'Oz']},

            {'name': 'Alpha Total', 'alpha': 0.5, 'eeg': 0.0, 'uniqueness': 0.0, 'type': 'misc', 'neighbours': ['O1', 'Oz', 'O2', 'POz', 'P7', 'P3', 'Pz', 'P4', 'P8']}
        ], 'standard_1005')

    @staticmethod
    def create_1010_32():
        """
        A static method to create a standard 32 channel cap in 10-10 system
        :return: An EegRecording object
        """
        return EegRecording([

            {'name': 'Fp1', 'alpha': 0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F7', 'FC5', 'F3', 'AFz', 'Fpz']},
            {'name': 'Fpz', 'alpha': 0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fp1', 'F3', 'AFz', 'F4', 'Fp2']},
            {'name': 'Fp2', 'alpha':  0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F8', 'FC6', 'F4', 'AFz', 'Fpz']},
            {'name': 'AFz', 'alpha':  0.0, 'eeg': 1.4, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fp1', 'Fz', 'Fp2', 'F4', 'Fz', 'F3']},

            {'name': 'F7', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T7', 'FC5', 'F3', 'Fp1']},
            {'name': 'F3', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F7', 'FC5', 'C3', 'FC1', 'Fz', 'AFz', 'Fpz', 'Fp1']},
            {'name': 'Fz', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F3', 'FC1', 'Cz', 'FC2', 'F4', 'AFz']},
            {'name': 'F4', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F8', 'FC6', 'C4', 'FC2', 'Fz', 'AFz', 'Fpz', 'Fp2']},
            {'name': 'F8', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T8', 'FC6', 'F4', 'Fp2']},

            {'name': 'FC5', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T7', 'C3', 'FC1', 'F3', 'F7']},
            {'name': 'FC1', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F3', 'FC5', 'C3', 'Cz', 'Fp2', 'Fz']},
            {'name': 'FC2', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fz', 'FC1', 'Cz', 'C4', 'FC6', 'F4']},
            {'name': 'FC6', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F8', 'F4', 'FC2', 'C4', 'T8']},

            {'name': 'T7', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'CP5', 'C3', 'FC5', 'F7']},
            {'name': 'C3', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F3', 'FC5', 'T7', 'CP5', 'P3', 'CP1', 'Cz', 'FC1']},
            {'name': 'Cz', 'alpha':  0.0, 'eeg': 1.3, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['Fz', 'FC1', 'C3', 'CP1', 'Cz', 'CP2', 'C4', 'FC2']},
            {'name': 'C4', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['F4', 'FC6', 'T8', 'CP6', 'P4', 'CP2', 'Cz', 'FC2']},
            {'name': 'T8', 'alpha':  0.0, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'FC6', 'C4', 'CP6', 'F8']},

            {'name': 'CP5', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'O1', 'P3', 'CP1', 'C3', 'FC5', 'T7']},
            {'name': 'CP1', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['FC1', 'C3', 'CP5', 'P3', 'POz', 'Pz', 'CP2', 'Cz']},
            {'name': 'CP2', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['FC2', 'C4', 'CP6', 'P4', 'POz', 'Pz', 'CP1', 'Cz']},
            {'name': 'CP6', 'alpha':  0.1, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'O2', 'P4', 'CP2', 'C4', 'FC6', 'T8']},

            {'name': 'P7', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T7', 'CP5', 'P3', 'O1']},
            {'name': 'P3', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'CP5', 'C3', 'CP1', 'Fz', 'POz', 'Oz', 'O1']},
            {'name': 'Pz', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P3', 'CP1', 'Cz', 'CP2', 'P4', 'POz']},
            {'name': 'P4', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'CP6', 'C4', 'CP2', 'Fz', 'POz', 'Oz', 'O2']},
            {'name': 'P8', 'alpha':  0.2, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['T8', 'CP6', 'P3', 'O2']},

            {'name': 'POz', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['O1', 'P3', 'CP1', 'Fz', 'CP2', 'P4', 'O2', 'Oz']},
            {'name': 'O1', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P7', 'CP5', 'P3', 'POz', 'Oz']},
            {'name': 'Oz', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['O1', 'P3', 'POz', 'P4', 'O2']},
            {'name': 'O2', 'alpha':  0.3, 'eeg': 1.0, 'uniqueness': 0.3, 'type': 'eeg', 'neighbours': ['P8', 'CP6', 'P4', 'POz', 'Oz']},
        ], 'standard_1005')

    def get_info(self):
        return self._info

    def get_dt(self):
        return self._dt

    def run(self, samples):
        """
        Run the recording for a certain amount of samples
        :return: an array with data
        """

        # create eeg data if necessary
        if not len(self._data) or samples != self._samples:
            self._data = np.zeros((samples, len(self._channels)), dtype="float32")     # lsl cannot  deal with 64 bit
            self._samples = samples

        # read eeg data from channels, sample by sample
        for n in range(0, samples):

            # first advance all noise sources
            for c in range(0, len(self._channels)):
                self._channels[c].next()

            # then calculate all eeg and alpha values
            for c in range(0, len(self._channels)):
                self._data[n, c] = self._channels[c].read_eeg()

        # return the mne object of the recording
        return self._data

