import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import librosa
from lyon.calc import LyonCalc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import re
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet


def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def _parse_rescale_arg(rescale):
  """Parse the rescaling argument to a standard form.

  Args:
    rescale ({'normalize', 'standardize', None}): Determines how rescaling
      will be performed.

  Returns:
    (str or None): A valid rescaling argument, for use with wav_to_array or
      similar.

  Raises:
    ValueError: Throws an error if rescale value is unrecognized.
  """
  if rescale is not None:
    rescale = rescale.lower()
  if rescale == 'normalize':
    out_rescale = 'normalize'
  elif rescale == 'standardize':
    out_rescale = 'standardize'
  elif rescale is None:
    out_rescale = None
  else:
    raise ValueError('Unrecognized rescale value: %s' % rescale)
  return out_rescale

def rescale_sound(snd_array, rescale):
  """Rescale the sound with the provided rescaling method (if supported).

  Args:
    snd_array (array): The array containing the sound data.
    rescale ({'standardize', 'normalize', None}): Determines type of
      rescaling to perform. 'standardize' will divide by the max value
      allowed by the numerical precision of the input. 'normalize' will
      rescale to the interval [-1, 1]. None will not perform rescaling (NOTE:
      be careful with this as this can be *very* loud if playedback!).

  Returns:
    array:
    **rescaled_snd**: The sound array after rescaling.
  """
  rescale = _parse_rescale_arg(rescale)
  if rescale == 'standardize':
    if issubclass(snd_array.dtype.type, np.integer):
      snd_array = snd_array / float(np.iinfo(snd_array.dtype).max)  # rescale so max value allowed by precision has value 1
    elif issubclass(snd_array.dtype.type, np.floating):
      snd_array = snd_array / float(np.finfo(snd_array.dtype).max)  # rescale so max value allowed by precision has value 1
    else:
      raise ValueError('rescale is undefined for input type: %s' % snd_array.dtype)
  elif rescale == 'normalize':
    snd_array = snd_array / float(snd_array.max())  # rescale to [-1, 1]
  # do nothing if rescale is None
  return snd_array

def wav_to_array(fn, rescale='standardize'):
  """ Reads wav file data into a numpy array.

    Args:
      fn (str): The file path to .wav file.
      rescale ({'standardize', 'normalize', None}): Determines type of
        rescaling to perform. 'standardize' will divide by the max value
        allowed by the numerical precision of the input. 'normalize' will
        rescale to the interval [-1, 1]. None will not perform rescaling (NOTE:
        be careful with this as this can be *very* loud if playedback!).

    Returns:
      tuple:
        **snd** (int): The sound in the .wav file as a numpy array.
        **samp_freq** (array): Sampling frequency of the input sound.
  """
  samp_freq, snd = wavfile.read(fn)
  snd = rescale_sound(snd, rescale)
  return snd, samp_freq

def search_audio_files(folder_path):
    """Parse the folder for audio files

    Args:
        folder_path ([string]): [parse the path for .wave files]

    Returns:
        [list]: [list of audio file names]
    """   

    file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and '.wav' in f]
    return file_names

def parse_filename(fn):
    """parse_filename function which parses the filename to split the filename delimited by _

    Args:
        fn ([string]): [pass the filename here]

    Returns:
        [list]: [returns the list of split strings.]
    """    
    parsed_fn= re.split('_+', fn)
    parsed_fn[2]=parsed_fn[2][0:-4]
    parsed_fn.append(fn)
    return parsed_fn

def convert_to_cochleagram(
    audio_path,
    decimation_factor=77,
    n=None,
    nonlinearity=None,
    maxLength=None,
):  
    """
     Creates a spectrogram of a wav file. Cochleagram dimension = Nf(channel)xNt(time steps)
    :param audio_path: path of wav file
    :param n: (int) Number of filters to use in the filterbank.
    :param nonlinearity:    None applies no nonlinearity. 
                            'db' will convert output to decibels (truncated at -60). 
                            'power' will apply 3/10 power compression.
    :return:
        None
    """

    #print("Processing %s..." % audio_path)
    
    # audio_path=src_dir+audio_fn
    signal,sample_rate = wav_to_array(audio_path)
    
    #plt.plot(signal); plt.show()
    signal = signal[50001: 125000]
    
    fs=12E3 #resample frequency
    data = librosa.core.resample(y=signal.astype(np.float64), orig_sr=sample_rate, target_sr=fs, res_type="scipy")
        
    # zero padding
    if maxLength is not None and len(data) > maxLength:
       
        err_msg=f"datalenght={len(data)}, maxlength={maxLength},data length cannot exceed padding length."
        print(err_msg)
        # raise ValueError(err_msg)
        return (None,-1)
    elif maxLength is not None and len(data) < maxLength:
        embedded_data = np.zeros(maxLength)
        offset = np.random.randint(low = 0, high = maxLength - len(data))
        embedded_data[offset:offset+len(data)] = data
    elif maxLength is not None and len(data) == maxLength:
        # nothing to do here
        embedded_data = data
        pass
    
    if maxLength is not None:
        data = embedded_data

    calc = LyonCalc()

    #using resampled data
    coch = calc.lyon_passive_ear(
        data, fs, decimation_factor=decimation_factor, ear_q=8,
    ) #, step_factor=0.35)

    coch=np.array(coch)

    return (coch,0)

def create_coch_cache(audio_dir, cache_fn, n=None, nonlinearity=None,cochLib=1):
    """
     Creates Cochleagrams cache parsing all the audio files in audio_dir.
 
    :param audio_dir: path of directory with audio files
    :param spectrogram_dir: path to save spectrograms
    :param n: (int) Number of filters to use in the filterbank.
    :param nonlinearity:    None applies no nonlinearity. 
                            'db' will convert output to decibels (truncated at -60). 
                            'power' will apply 3/10 power compression.
    :return:
        None
    """
    file_names = search_audio_files(audio_dir)
    print(f"total files={len(file_names)}")
    cache={}
    for file_name in file_names:
        audio_path = os.path.join(audio_dir, file_name)
        coch,status=convert_to_cochleagram(audio_path)
        if(status>=0):
          cache[file_name]=coch
    np.save(cache_fn, cache)
    

def create_dataset(data_path,train_fn,test_fn,num_speaker=5, num_utternace=10, train_ratio=0.8, seed=1):
    '''
    Function to generate isolate spoken digit dataset for give training ratio
    :param data_path: (string) path of directory with .npy files
    :param trainingRatio: (float) ratio of training size to (training +testing size)
    :return:
        return the dataset dictionary with keys "train_data" for list of training data and
        "test_data" for for list of testing data .
    '''
#------------------------------------------------------------------------------ 
    #modifcation to resolve ValueError: Object arrays cannot be loaded when allow_pickle=False
    # save np.load
#     np_load_old = np.load    
    # modify the default parameters of np.load
    load = lambda *a,**k: np.load(*a, allow_pickle=True, **k)
#------------------------------------------------------------------------------ 
    
    dataset=load(data_path).item()    
    file_names=dataset.keys()
    
    parsed_files=[]
    for file_name in file_names:
        print(file_name)
        parsed_files.append(parse_filename(file_name))
    parsed_files=np.array(parsed_files)
    
    column_names=["digit","speaker","utterance","filename"]
    df=pd.DataFrame(parsed_files,columns=column_names)
    #convert the string to integer
    df[column_names[0]] = df[column_names[0]].astype(int)
    df[column_names[2]] = df[column_names[2]].astype(int)

    #get the unique speaker details, shuffle the speaker and pick the selected number of speaker
    speakers=df["speaker"].unique()
    np.random.shuffle(speakers)
    sel_speakers=speakers[0:num_speaker]

    #calculate the training and testing size for given speaker and digit
    train_size=int(num_utternace*train_ratio)
    test_size=num_utternace-train_size
    #create a dataframe for training and testing filenames
    train_df=pd.DataFrame(columns=column_names)
    test_df=pd.DataFrame(columns=column_names)
    for sel_speaker in sel_speakers:
       df_speaker=df.loc[df['speaker']==sel_speaker]
       for digit in np.arange(10):
          df_speaker_digit=df_speaker.loc[df_speaker['digit']==digit]
          df_shuffled = df_speaker_digit.sample(frac=1)
          train_df=train_df._append(df_shuffled.head(train_size), ignore_index = True)
          test_df=test_df._append(df_shuffled.tail(test_size), ignore_index = True)

    train_cache={}
    for index, row in train_df.iterrows():
        fn=row[column_names[3]]
        train_cache[fn]={}
        for i in np.arange(3):
          train_cache[fn][column_names[i]]=row[column_names[i]]
        train_cache[fn]["label"]=to_categorical(row[column_names[0]],num_classes=10)
        train_cache[fn]["coch"]=dataset[fn]

    test_cache={}
    for index, row in test_df.iterrows():
        fn=row[column_names[3]]
        test_cache[fn]={}
        for i in np.arange(3):
          test_cache[fn][column_names[i]]=row[column_names[i]]
        test_cache[fn]["label"]=to_categorical(row[column_names[0]],num_classes=10)
        test_cache[fn]["coch"]=dataset[fn]

    np.save(train_fn,train_cache)
    np.save(test_fn,test_cache)

def gen_features_targets(cache):
    cache_keys=list(cache.keys())
    features=[]
    targets=[]
    for key in cache_keys:
      label=cache[key]["label"]*2-1
      coch =cache[key]["coch"]
      labels=np.repeat(label.reshape(-1,1), coch.shape[0], axis=1).T
      features.append(coch)
      targets.append(labels)
      # print(labels.shape)
    n_feature=coch.shape[1]
    n_digits  =labels.shape[1]
    features=np.array(features).reshape(-1,n_feature)
    targets       =np.array(targets).reshape(-1,n_digits)
    return (features,targets)
   

def linreg_classifier(train_cache_fn,test_cache_fn,seed=1):
    '''
    this function runs machine learning using cochleagram files and test the performance.
    '''

    train_cache=np.load(train_cache_fn,allow_pickle=True).item()
    test_cache=np.load(test_cache_fn,allow_pickle=True).item()
    train_feature,train_targets=gen_features_targets(train_cache)
    # test_feature,test_targets=gen_features_targets(test_cache)

    clf = LinearRegression(fit_intercept=True)
    clf.fit(train_feature, train_targets)
    score=clf.score(train_feature, train_targets)
    print("Regression Score=%f"%(score))

    weights=clf.coef_
    bias=clf.intercept_
    bias=np.array(bias).reshape(10,1)
    # print(weights.shape)
    # estimatedTargets=[]
    # actualTarget=[]
    test_cache_keys=list(test_cache.keys())
    test_est_lst=[]
    test_target_lst=[]
    for cache_key in test_cache_keys:
        test_feature=test_cache[cache_key]['coch']
        test_target =test_cache[cache_key]["label"]*2-1
        test_est=np.dot(weights,test_feature.T)+bias
        test_est_mean=test_est.mean(axis=1)
        test_est_mean=test_est_mean.reshape(1,10)
        test_target=test_target.reshape(1,10)
        test_est_lst.append(test_est_mean)
        test_target_lst.append(test_target)

    
    print("WSR=%f"%(WSR_MSE( test_target_lst,test_est_lst)))
    
def WSR_MSE(target_lst,est_lst):
    '''
    
    :param target: this is list of targets 
    :param estimate: this is list of estimate
    '''
    
    sucess=0
    count=0
    for estimate,target in zip(est_lst,target_lst):
        estimate_winner=(estimate==np.amax(estimate))*2-1
        count=count+1
        if((estimate_winner==target).all()):
            sucess=sucess+1
    
    wsr=sucess/count
    return wsr
    


if __name__ == '__main__':      
    # path= "free-spoken-digit-dataset/recordings/"
    # audio_file = "0_jackson_1.wav"
    # signal,sample_rate = wav_to_array(audio_file)
    # print("original sampling rate=%d"%(sample_rate))
    # fs=int(12E3) #resample frequency
    # data = librosa.core.resample(y=signal.astype(np.float64), orig_sr=sample_rate, target_sr=fs, res_type="scipy")
    # plt.figure()
    # plt.plot(data)
    # plt.show()
    # parsed_fn=parse_filename(audio_file)
    # print(parsed_fn)
    # coch=convert_to_cochleagram(path,audio_file)
    # plt.imshow(coch)
    # plt.show()
    # audio_dir= "free-spoken-digit-dataset/recordings/"
    # create_coch_cache(audio_dir,"coch_cache.npy")

    # create_dataset("coch_cache.npy","train_set.npy","test_set.npy")
    linreg_classifier("train_set.npy","test_set.npy")




