"""
File to simulate multiple LSL streams.
Each stream consists of 64 channels with an Ornstein-Uhlenbeckprocess in each (coloured noise).
Added alpha noise at 10 Hz occurs from time to time in O1, O2 and Oz.
"""
import datetime
import random
import mne
import numpy as np
import matplotlib.pyplot as plt


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
                eegAmplitude = self._eegIntensity * self._eegIntensity / (self._eegIntensity + self._currentAlphaAmplitude)
            else:
                # if alpha-intensity is zero, eeg amplitude is as defined
                eegAmplitude = self._eegIntensity

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
    _data = None
    _samplingFreq = 500
    _samples = 0
    _channels = []
    _dt = 0.002

    def __init__(self):
        self._dt = 0.002
        self._channels = 64
        self._samplingFreq = 500.0
        self._channels = []

        # define all channels
        # frontal
        fp1 = Channel('Fp1', 0.0, 1.4, 0.3)
        fp2 = Channel('Fp2', 0.0, 1.4, 0.3)
        f7 = Channel('F7', 0.0, 1.0, 0.3)
        f3 = Channel('F3', 0.0, 1.0, 0.3)
        fz = Channel('Fz', 0.0, 1.0, 0.3)
        f4 = Channel('F4', 0.0, 1.0, 0.3)
        f8 = Channel('F8', 0.0, 1.0, 0.3)

        # central
        t3 = Channel('T3', 0.05, 1.0, 0.3)
        c3 = Channel('C3', 0.05, 1.0, 0.3)
        cz = Channel('Cz', 0.05, 1.3, 0.3)
        c4 = Channel('C4', 0.05, 1.0, 0.3)
        t4 = Channel('T4', 0.05, 1.0, 0.3)

        # occipital
        t5 = Channel('T5', 0.1, 1.0, 0.3)
        p3 = Channel('P3', 0.1, 1.0, 0.3)
        pz = Channel('Pz', 0.1, 1.0, 0.3)
        p4 = Channel('P4', 0.1, 1.0, 0.3)
        t6 = Channel('T6', 0.1, 1.0, 0.3)
        o1 = Channel('O1', 0.3, 1.0, 0.3)
        o2 = Channel('O2', 0.3, 1.0, 0.3)

        # inverse alpha
        m = Channel('Alpha Total', 0.5, 0.0, 0.0, channel_type='misc')

        # set neighbours
        fp1.set_neighbours([f7, f3, fz, fp2])
        fp2.set_neighbours([fp1, fz, f4, f8])
        f7.set_neighbours([t3, c3, f3, fp1])
        f3.set_neighbours([f7, fp1, fz, c3])
        fz.set_neighbours([fp1, f3, cz, f4, fp2])
        f4.set_neighbours([fz, c4, f8, fp2])
        f8.set_neighbours([t4, c4, f4, fp2])

        t3.set_neighbours([f7, c3, t5])
        c3.set_neighbours([t3, f3, cz, p3])
        cz.set_neighbours([c3, fz, c4, pz])
        c4.set_neighbours([cz, f4, t4, p4])
        t4.set_neighbours([t6, c4, f8])

        t5.set_neighbours([t3, c3, p3, o1])
        p3.set_neighbours([t5, c3, pz, o1])
        pz.set_neighbours([p3, cz, p4, o1, o2])
        p4.set_neighbours([pz, c4, t6, o2])
        t6.set_neighbours([o2, p4, c4, t4])
        o1.set_neighbours([t5, p3, pz, o2])
        o2.set_neighbours([o1, pz, p4, t6])

        m.set_neighbours([t5, p3, pz, p4, t6, o1, o2])

        # save
        self._channels.append(fp1)
        self._channels.append(fp2)
        self._channels.append(f7)
        self._channels.append(f3)
        self._channels.append(fz)
        self._channels.append(f4)
        self._channels.append(f8)
        self._channels.append(t3)
        self._channels.append(c3)
        self._channels.append(cz)
        self._channels.append(c4)
        self._channels.append(t4)
        self._channels.append(t5)
        self._channels.append(p3)
        self._channels.append(pz)
        self._channels.append(p4)
        self._channels.append(t6)
        self._channels.append(o1)
        self._channels.append(o2)
        self._channels.append(m)

        channel_names = [c.get_name() for c in self._channels]
        channel_types = [c.get_type() for c in self._channels]
        self._info = mne.create_info(channel_names, ch_types=channel_types, sfreq=1.0 / self._dt)
        self._info.set_montage("standard_1020")
        print(self._info)

    def get_info(self):
        return self._info

    def run(self, samples):
        """
        Run the recording for a certain amount of samples
        :return: an array with data
        """

        # create eeg data if necessary
        if not self._data or samples != self._samples:
            self._data = np.zeros((len(self._channels), samples))
            self._samples = samples

        # read eeg data from channels, sample by sample
        for n in range(0, samples):

            # first advance all noise sources
            for c in range(0, len(self._channels)):
                self._channels[c].next()

            # then calculate all eeg and alpha values
            for c in range(0, len(self._channels)):
                self._data[c, n] = self._channels[c].read_eeg()

        # return the mne object of the recording
        return self._data


# init EEG class
recording = EegRecording()

# get 10 s of data
time_s = 50     # seconds of data
start_time = datetime.datetime.now()
raw_data = recording.run(time_s * 500)     # run
end_time = datetime.datetime.now()
print("Simulated {} seconds of data in {}.".format(time_s, end_time - start_time))

# plot curves
eeg = mne.io.RawArray(raw_data, recording.get_info())
curves = eeg.plot(show_scrollbars=False, show_scalebars=True,
                  remove_dc=False, scalings={'eeg': 20e-6, 'misc': 5.0})
curves.savefig('simulated_curves.png')
splot = eeg.plot_sensors(show=True, show_names=True)
splot.savefig('simulated_sensors.png')

# export
eeg.export('simulated_eeg.edf', overwrite=True)

# plot spectrum
spectrum = eeg.compute_psd(fmin=0, fmax=50)
splot = spectrum.plot(show=True, average=True, picks=['O1', 'O2', 'T5', 'P3', 'Pz', 'P4', 'T6'], amplitude=False)
splot.savefig('simulated_spectrum.png')
tmap = spectrum.plot_topomap(show=True, cmap='coolwarm')
tmap.savefig('simulated_topomap.png')

# report
report = mne.Report(title="Simulated Data")
report.add_raw(raw=eeg, title="Info", butterfly=False, psd=False)  # omit PSD plot
report.add_image('simulated_curves.png', title="EEG Static (10 s)")
report.add_image('simulated_spectrum.png', title="Spectrum of Channels O1, O2, T5, P3, Pz, P4, T6")
report.add_image('simulated_sensors.png', title="Sensor Layout")
report.add_image('simulated_topomap.png', title="Tological Plot")
report.save("simulated_report.html", overwrite=True)
