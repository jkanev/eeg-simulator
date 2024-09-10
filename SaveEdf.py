"""
Simple script saving 10 seconds of EEG recording as EDF, along with pictures and an HTML report:
simulated_eeg.edf
simulated_spectrum.png
simulated_curves.png
simulated_sensors.png
simulated_report.html
"""

import EegRecording
import datetime
import mne

# init EEG class
recording = EegRecording.EegRecording.create_1010_32A()

# get 10 s of data
time_s = 10     # seconds of data
start_time = datetime.datetime.now()
raw_data = recording.run(time_s * 500)     # run
end_time = datetime.datetime.now()
print("Simulated {} seconds of data in {}.".format(time_s, end_time - start_time))

# plot curves
eeg = mne.io.RawArray(raw_data, recording.get_info())
curves = eeg.plot(show_scrollbars=False, show_scalebars=True, n_channels=32,
                  remove_dc=False, scalings={'eeg': 20e-6, 'misc': 5.0})
curves.savefig('simulated_curves.png')
splot = eeg.plot_sensors(show=True, show_names=True)
splot.savefig('simulated_sensors.png')

# export
eeg.export('simulated_eeg.edf', overwrite=True)

# plot spectrum
spectrum = eeg.compute_psd(fmin=0, fmax=50)
splot = spectrum.plot(show=True, average=True, picks=['O1', 'Oz', 'O2', 'POz', 'P7', 'P3', 'Pz', 'P4', 'P8'], amplitude=False)
splot.savefig('simulated_spectrum.png')
tmap = spectrum.plot_topomap(show=True, cmap='coolwarm')
tmap.savefig('simulated_topomap.png')

# report
report = mne.Report(title="Simulated Data")
report.add_raw(raw=eeg, title="Info", butterfly=False, psd=False)  # omit PSD plot
report.add_image('simulated_curves.png', title="EEG Static (10 s)")
report.add_image('simulated_spectrum.png', title="Spectrum of parietal and occipital Channels")
report.add_image('simulated_sensors.png', title="Sensor Layout")
report.add_image('simulated_topomap.png', title="Tological Plot")
report.save("simulated_report.html", overwrite=True)
