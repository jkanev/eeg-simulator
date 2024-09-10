"""
Simple script that starts an LSL outlet and continuosly streams EEG at 500 Hz.
"""

from EegRecording import EegRecording
from pylsl import StreamOutlet, StreamInfo
import random
import datetime

# init EEG class
recording = EegRecording.create_1010_32A()
chunk_size = 250  # samples of data
dt = recording.get_dt()

# init LSL stream
nme_info = recording.get_info()
sname = 'EEG'
stype = 'EEG'
ssn = '{}—{}—{}'.format(random.randint(10000, 99999), random.randint(10000, 99999), random.randint(10000, 99999))
snchannels = len(nme_info['ch_names'])
sfreq = nme_info['sfreq']
lsl_info = StreamInfo(sname, stype, snchannels, sfreq, 'float32', ssn)
lsl_out = StreamOutlet(lsl_info)
print("Created LSL stream. "
      "Name: {}, type: {}, #channels: {}, frequency: {}, serial: {}".format(sname, stype, snchannels, sfreq, ssn))

# start
pushed_seconds = 0.0
while True:
    raw_data = recording.run(chunk_size)
    pushed_seconds += chunk_size*dt
    lsl_out.push_chunk(raw_data, pushed_seconds)
    print("{} -- sent {} seconds and {} samples of data, total of {} seconds, to LSL stream {}/{}."
          "".format(datetime.datetime.now(), chunk_size*dt, chunk_size, pushed_seconds, sname, ssn))
