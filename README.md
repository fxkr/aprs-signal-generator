This is a simple configurable signal generator for APRS packets.

This is useful for people developing APRS decoders.

APRS is the Automatic Packet Reporting System, an amateur radio protocol for digital beaconing and messaging.

License: AGPLv3  (Note that, as with any software, the license applies to this tool and its source code, but not to the output produced by this tool.)


# Usage

Generate a signal and show the beginning of it visually:

```
python aprs_signal_generator.py --show-graph
```

Generate a signal and write it to a .flac file:

```
python aprs_signal_generator.py --audio-out generated.flac
```

It is also written such that it can be used like a library. However there is zero expectation of stability/compatibility at this time.