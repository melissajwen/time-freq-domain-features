from scipy import signal, fft
from scipy.io import wavfile
import numpy as np

GROUND_TRUTH_FILE = "music_speech.mf"
RESULT_FILE = "results.arff"

BUFFER_LEN = 1024
HOP_SIZE = 512
L = 0.85  # used for SRO

PRECISION = "%.6f"

HEADER = "@RELATION music_speech\n" \
         "@ATTRIBUTE RMS_MEAN NUMERIC\n" \
         "@ATTRIBUTE ZCR_MEAN NUMERIC\n" \
         "@ATTRIBUTE SC_MEAN NUMERIC\n" \
         "@ATTRIBUTE SRO_MEAN NUMERIC\n" \
         "@ATTRIBUTE SFM_MEAN NUMERIC\n" \
         "@ATTRIBUTE RMS_STD NUMERIC\n" \
         "@ATTRIBUTE ZCR_STD NUMERIC\n" \
         "@ATTRIBUTE SC_STD NUMERIC\n" \
         "@ATTRIBUTE SRO_STD NUMERIC\n" \
         "@ATTRIBUTE SFM_STD NUMERIC\n" \
         "@ATTRIBUTE class {music,speech}\n" \
         "@DATA\n"


def main():
    # Open the necessary files
    ground_truth = open(GROUND_TRUTH_FILE, "r")
    result = open(RESULT_FILE, "w")

    # Write header to output file
    result.write(HEADER)

    # Main loop to perform wav calculations
    for line in ground_truth:
        line_arr = line.split("\t")
        wav_file_name = "music_speech/" + line_arr[0].strip()
        wav_file_type = line_arr[1].strip()

        # Split up wav file into buffers
        buffers = get_buffers_from_wav(wav_file_name)

        # Calculate time domain features
        rms_arr = calc_rms_arr(buffers)
        rms_mean = np.mean(rms_arr)
        rms_std = np.std(rms_arr)

        zcr_arr = calc_zcr_arr(buffers)
        zcr_mean = np.mean(zcr_arr)
        zcr_std = np.std(zcr_arr)

        # Convert from time domain to frequency domain
        windows = get_windows_from_buffers(buffers)

        # Calculate frequency domain features
        sc_arr = calc_sc_arr(windows)
        sc_mean = np.mean(sc_arr)
        sc_std = np.std(sc_arr)

        sro_arr = calc_sro_arr(windows)
        sro_mean = np.mean(sro_arr)
        sro_std = np.std(sro_arr)

        sfm_arr = calc_sfm_arr(windows)
        sfm_mean = np.mean(sfm_arr)
        sfm_std = np.std(sfm_arr)

        result.write(PRECISION % rms_mean + "," +
                     PRECISION % zcr_mean + "," +
                     PRECISION % sc_mean + "," +
                     PRECISION % sro_mean + "," +
                     PRECISION % sfm_mean + "," +
                     PRECISION % rms_std + "," +
                     PRECISION % zcr_std + "," +
                     PRECISION % sc_std + "," +
                     PRECISION % sro_std + "," +
                     PRECISION % sfm_std + "," +
                     wav_file_type + "\n")


# Function to calculate buffers
def get_buffers_from_wav(wav_file_name):
    freq, file_data = wavfile.read(wav_file_name)
    data = file_data / 32768.0  # convert to samples

    buffers = []
    start = 0
    end = BUFFER_LEN
    num_buffers = int(len(file_data) / HOP_SIZE - 1)

    for i in range(num_buffers):
        buffer_data = data[start:end]
        start += HOP_SIZE
        end += HOP_SIZE

        if len(buffer_data) == BUFFER_LEN:
            buffers.append(buffer_data)

    return buffers


# Convert buffers from time domain to frequency domain
def get_windows_from_buffers(buffers):
    windows = []

    for buf in buffers:
        win = buf * signal.hamming(len(buf))
        win = fft(win)

        # Only keep the first half of the array
        win = win[:int(len(win) / 2 + 1)]
        windows.append(win)

    return windows


# Calculation of time domain features
# Root mean squared
def calc_rms_arr(buffers):
    rms_arr = []
    for buf in buffers:
        rms = calc_rms(buf)
        rms_arr.append(rms)
    return rms_arr


def calc_rms(buffer):
    rms = np.mean(buffer ** 2)
    rms = np.sqrt(rms)
    return rms


# Zero crossings rate
def calc_zcr_arr(buffers):
    zcr_arr = []
    for buf in buffers:
        zcr = calc_zcr(buf)
        zcr_arr.append(zcr)
    return zcr_arr


def calc_zcr(buffer):
    sign = np.sign(buffer)
    diff = np.diff(sign)
    zcr = len(np.where(np.abs(diff) == 2)[0]) / (len(buffer) - 1)
    return zcr


# Calculation of frequency domain features
# Spectral centroid
def calc_sc_arr(windows):
    sc_arr = []
    for win in windows:
        sc = calc_sc(win)
        sc_arr.append(sc)
    return sc_arr


def calc_sc(buffer):
    num = 0
    for k in range(len(buffer)):
        num += k * np.abs(buffer[k])
    den = np.sum(np.abs(buffer))
    sc = num / den
    return sc


# Spectral roll-off
def calc_sro_arr(windows):
    sro_arr = []
    for win in windows:
        sro = calc_sro(win)
        sro_arr.append(sro)
    return sro_arr


def calc_sro(buffer):
    running_sum = 0
    total = np.sum(np.abs(buffer))
    i = 0

    while 1:
        running_sum += np.abs(buffer[i])
        if running_sum >= L * total:
            return i
        i += 1
    return -1


# Spectral flatness measure
def calc_sfm_arr(windows):
    sfm_arr = []
    for win in windows:
        sfm = calc_sfm(win)
        sfm_arr.append(sfm)
    return sfm_arr


def calc_sfm(buffer):
    num = np.exp(np.mean(np.log(np.abs(buffer))))
    den = np.mean(np.abs(buffer))
    sfm = num / den
    return sfm


main()
