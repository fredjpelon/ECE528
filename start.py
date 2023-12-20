#! /usr/bin/env python

# This is the module for running inference on the Raspberry Pi
# This is it, that's all there is.
# Running this module was added as a @reboot line to the cron table
# with '&' post-pended of course to free the boot process

import os
import pathlib
# The interpreter and inference is run entirely through the Coral.AI tools
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
#from PIL import Image - necessary for the Coral example
import numpy as np
import scipy.io.wavfile
from scipy import signal
from datetime import date
import datetime


norm_factor = 30000.0 # for gaining up the samples
clip_limit = 8000     # noise floor

# same STFT calculator as training, but, see pdc5_train J Notebook for
# discussion comparing scipy and tf STFT
def get_spectrogram(waveform):
    ff = 0.7
    # Convert the waveform to a spectrogram via a STFT.
    # spectrogram = tf.signal.stft(
    #    waveform, frame_length=255, frame_step=1024)
    _, _, spectrogram = signal.stft(
        waveform, nperseg=1024, noverlap=768)
    win = signal.get_window('hann', 1024)
    scale = np.sqrt(1.0 / win.sum()**2)
    #print(type(spectrogram))
    # Obtain the magnitude of the STFT.
    spectrogram = ff*np.absolute(spectrogram.transpose())/scale
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[:-1,:]
    spectrogram = spectrogram[..., np.newaxis]
    return spectrogram

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
con_model_file = os.path.join(script_dir, 'conc5.tflite')        # voice recog trained model
con_label_file = os.path.join(script_dir, 'conc5labels.txt')     # classes
#pd_model_file = os.path.join(script_dir, 'pdqc4_edgetpu.tflite')
#pd_model_file = os.path.join(script_dir, 'pdc4.tflite')
pd_model_file = os.path.join(script_dir, 'pdc5.tflite')          # prairie dog trained model
pd_label_file = os.path.join(script_dir, 'pdc4labels.txt')
pd_label_file = os.path.join(script_dir, 'pdc5labels.txt')       # labels
con_wav_file = os.path.join(script_dir, 'contest.wav')           # voice rec sample repository
pd_wav_file = os.path.join(script_dir, 'pdtest.wav')             # pd detect sample repo
data_dir = os.path.join(script_dir, 'data/')                     # save dir
#print(wav_file)
#os.system('aplay ' + wav_file) # play for a test

# infinite loop unless quit command
while 1:
    # Initialize the TF interpreter
    # according to the docs, if the model is not quantized
    # fully, the model will load to the CPU
    interpreter_con = edgetpu.make_interpreter(con_model_file)
    interpreter_con.allocate_tensors()
    os.system('aplay ./pdog/readyforcommand.wav')
    CON_DETECTED = 0
    while CON_DETECTED == 0:
        # Resize the image
        size = common.input_size(interpreter_con)

        DETECT_AUDIO = 0
        while DETECT_AUDIO == 0: # loop until valid attention
            os.system('arecord -f S16_LE -c1 -r44100 --duration=4 ./pdog/contest.wav')
            # convert the wav file to a numpy spectrogram
            sample_rate, data = scipy.io.wavfile.read(con_wav_file)
            if data.max() > clip_limit:
                DETECT_AUDIO = 1

        # normalize data
        #print(data[:5])
        data = data.astype(np.float64)
        data *= norm_factor / max(abs(data))
        data = data.astype(np.int16)

        # samples in waveform
        samplen = len(data)/sample_rate
        # 3 s chunks
        chunks = round(samplen/3)
        # force each chunk to exactly 3 s
        chunkval = 3*sample_rate
        temp = data[0:chunkval]
        #temp = np.pad(temp, (0, chunkval-len(temp)), 'constant') # zero-pad to constant 3 s
        if abs(abs(temp).max()) > clip_limit:
            image=get_spectrogram(temp.astype(float))

        # Run an inference
        common.set_input(interpreter_con, image)
        interpreter_con.invoke()
        classes = classify.get_classes(interpreter_con, top_k=1)

        # Print the result
        labels = dataset.read_label_file(con_label_file)
        for c in classes:
            string = labels.get(c.id, c.id)
            if string == 'OKCon':
                CON_DETECTED = 1
                os.system('aplay ./pdog/listening.wav')
            else:
                os.system('aplay ./pdog/heardother.wav')
                print('%s: %.5f' % (string, c.score))

    RECORD_DETECTED = 0
    while RECORD_DETECTED == 0: # loop until valid command
        # Resize the image
        size = common.input_size(interpreter_con)

        DETECT_AUDIO = 0
        while DETECT_AUDIO == 0:
            os.system('arecord -f S16_LE -c1 -r44100 --duration=4 ./pdog/contest.wav')
            # convert the wav file to a numpy spectrogram
            sample_rate, data = scipy.io.wavfile.read(con_wav_file)
            if data.max() > clip_limit:
                DETECT_AUDIO = 1

        # normalize data
        data = data.astype(np.float64)
        data *= norm_factor / max(abs(data))
        data = data.astype(np.int16)

        # samples in waveform
        samplen = len(data)/sample_rate
        # 3 s chunks
        chunks = round(samplen/3)
        # force each chunk to exactly 3 s
        chunkval = 3*sample_rate
        temp = data[0:chunkval]
        #temp = np.pad(temp, (0, chunkval-len(temp)), 'constant') # zero-pad to constant 3 s
        if abs(abs(temp).max()) > clip_limit:
            image=get_spectrogram(temp.astype(float))

            # Run an inference
            common.set_input(interpreter_con, image)
            interpreter_con.invoke()
            classes = classify.get_classes(interpreter_con, top_k=1)

            # Print the result
            labels = dataset.read_label_file(con_label_file)
            for c in classes:
                string = labels.get(c.id, c.id)
                if string == 'Record': # command to start pd detect
                    RECORD_DETECTED = 1
                    print('COMMAND: %s' % (string))
                    print('Proceeding to run inference on PDs')
                    os.system('aplay ./pdog/inferring.wav')
                elif string == 'Save': # command to quit
                    os.system('aplay ./pdog/quitting.wav')
                    quit()
                else:
                    os.system('aplay ./pdog/heardother.wav')
                    print('%s: %.5f' % (string, c.score))

    RECORD_LOOP = 1
    while RECORD_LOOP == 1: # pd record infinite loop
        # Initialize the TF interpreter
        interpreter_pd = edgetpu.make_interpreter(pd_model_file)
        interpreter_pd.allocate_tensors()

        # Resize the image
        size = common.input_size(interpreter_pd)

        os.system('aplay ./pdog/running.wav')
        os.system('arecord -f S16_LE -c1 -r44100 --duration=30 ./pdog/pdtest.wav')
        # convert the wav file to a numpy spectrogram
        sample_rate, data = scipy.io.wavfile.read(pd_wav_file)

        NOTSILENCE = 1
        print('data max = ',data.max())
        if data.max() < clip_limit: # return to control if no sounds (mute mic for hard stop here)
            NOTSILENCE = 0
            RECORD_LOOP = 0
            os.system('aplay ./pdog/onlysilence.wav')
            print('Silence detected, returning control to Con')

        if NOTSILENCE == 1:
            # normalize data
            data = data.astype(np.float64)
            data *= norm_factor / max(abs(data))
            data = data.astype(np.int16)

            # samples in waveform
            samplen = len(data)/sample_rate
            # 3 s chunks
            chunks = round(samplen/3)
            #chunkval = floor(len(data)/chunks)
            # force each chunk to exactly 3 s
            chunkval = 3*sample_rate

            # date object of today's date
            today = date.today() 
            y=str(today.year)
            m=str(today.month)
            d=str(today.day)
            print("Current year:", y)
            print("Current month:", m)
            print("Current day:", d)

            now = datetime.datetime.now()
            h=str(now.hour)
            mi=str(now.minute)
            s=str(now.second)
            print("Current hour:",h)
            print("Current minute:",mi)
            print("Current second:",s)

            PD_DETECTED = 0
            for i in range(chunks): # infer prairie dog barks
                temp = data[0+i*chunkval:chunkval+i*chunkval]
                temp = np.pad(temp, (0, chunkval-len(temp)), 'constant') # zero-pad to constant 3 s


                if abs(abs(temp).max()) > clip_limit:
                    image=get_spectrogram(temp.astype(float))
                    # Run an inference
                    common.set_input(interpreter_pd, image)
                    interpreter_pd.invoke()
                    classes = classify.get_classes(interpreter_pd, top_k=1)

                    # Print the result
                    labels = dataset.read_label_file(pd_label_file)
                    print('chunk ',i)
                    for c in classes:
                        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
                        if labels.get(c.id, c.id) != 'Other': # od detected, save it
                            PD_DETECTED = 1
                            print('collecting sample')
                            # serialized filename
                            fname = 'JF' + labels.get(c.id, c.id) + '_' + y + m + d + "_" + h + mi + s + "_" + str(i) + ".wav"
                            data_file = os.path.join(data_dir, fname)
                            print('data file:', data_file)
                            scipy.io.wavfile.write(data_file, sample_rate, temp)
                else:
                    print('low chunk')

            if PD_DETECTED == 0:  # no pds detected in sample: return to control
                RECORD_LOOP = 0
                os.system('aplay ./pdog/nopdogsdetected.wav')
                print('No prairie dogs detected, returning control to Con')
