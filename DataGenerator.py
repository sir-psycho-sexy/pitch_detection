import sys, os, time, datetime, copy
import numpy as np
import librosa, pickle, keras.utils.data_utils
import scipy.signal

try:
  import DataLoader
except:
  print('Custom Dataloader not found in directory!')


# Class to generate and feed intput to the various models
# using Keras base to ensure safe multiprocessing
class DataGenerator(keras.utils.data_utils.Sequence):

  def __init__(self, spectra_path=None,                 # path to the dataset of spectra (already preprocessed by a data loader)
              analytic_mode=False,                      # determines if the model should be treated as analytic (batch_size=1 full sample)
              use_aug_transposed=True,                  # determines if samples augmented by transposition should be used
              use_aug_noised=True,                      # determines if samples augmented by added noise should be used
              use_comp_samples=True,                    # determines if samples based on chordal play (focus on multiple simultaneous pitches) should be included
              use_solo_samples=True,                    # determines if samples based on solo play (focus on fewer simultaneous pitches) should be included
              use_aug_validation=True,                  # determines if augmented samples should be used within the validation set
              batch_size=90,                            # batch size that is passed with every __getitem__ call
              window_size=9,                            # size of the window centered around the current frame to be generated
              valid_test_split=[0.1, 0.1],              # ratio of the full dataset to split into validation ([0]) and test ([1]) sets
              full_test_samples=True,                   # whether the test set should contain full samples (no shuffled frame chunks)
              cross_validate=False,                     # whether to use k-fold cross validation on the training/validation sets (overrides use_aug_validation)
              binarize_labels=True,                     # determines if labels should be converted to a binary representation for the semitone classes
              lowest_note='E2',                         # lowest note to be considered for labels (for number of classes when binarizing labels)
              highest_note='C6',                        # highest note to be considered for labels (for number of classes when binarizing labels)
              include_empty_class=False,                # determines if labels should include a class for no note ringing (may be useful for 6-note polyphony detection)
              shuffle_after_epochs=True,                # determines if samples should be shuffled between epochs
              shuffle_consecutive_frames=27,             # determines how many consecutive frames should be shuffled together within batches (0->only full samples are shuffled)
              include_analytic_results=None,            # include predictions of the analytic model at the given path in the input
              include_rms=None,                         # moving RMS values at the given path in the input (helps with detecting empty frames)
              include_mfcc=None,                        # include MFCCs from the given directory in the sample inputs as well
              filter_type='bessel',                     # IIR filter type to use when preprocessing filters (includes 'bessel', 'butter', 'cheby1', 'cheby2', 'ellip', 'random')
              lowpass_cutoff=librosa.note_to_hz('E2'),  # cut-off frequency of the optional low-pass filter
              highpass_cutoff=librosa.note_to_hz('C6'), # cut-off frequency of the optional high-pass filter
              normalize_spectra='framewise',            # determines if and how spectra should be normalized ('framewise', 'freqwise', 'full', or None)
              spectra_scale='db',                       # determines if the spectra should be scaled to the Decibel scale, or left as linear amplitudes
              include_empty_frames=True,                # determines if frames with no note playing should be included
              pad_frames=True,                          # determines if the samples should be padded on both sides
              reorder_cqt_bins=None,                    # optionally reorder CQT/VQT bins into 'semitone' 'circle' of fifth semitone groupings
              load_generator_setup=None,                # path to optionally load fully saved generators into memory (contains all data in the structure)
              random_seed=0,                            # seed for random number generator
              ):
    
    
    np.random.seed(random_seed)
    
    
    self.path_data = ((os.path.dirname(os.path.abspath(__file__)) + os.path.sep) if '__file__' in globals() 
                            else 'gdrive/MyDrive/MSc Diploma/') + 'data' + os.path.sep # colab support for mounted Drive
    
    self.path_generators = self.path_data + 'generators' + os.path.sep
    
    if not os.path.exists(self.path_generators):
      os.makedirs(self.path_generators)

    if load_generator_setup is not None:
      if os.path.exists(self.path_generators + load_generator_setup):
        with open(self.path_generators+load_generator-setup, 'rb') as file:
          self.__dict__ = pickle.load(file)
        self.on_epoch_end() 
        print(f'Successfully loaded prepared data generator!')
      else:
        print('Unable to load data generator. File does not exist!')
      return


    if spectra_path is None:
      spectra_path = np.random.choice(os.listdir(self.path_data + 'preprocessed' + os.path.sep))
    self.spectra_path = self.path_data + 'preprocessed' + os.path.sep + spectra_path + os.path.sep

    self.analytic_mode      = analytic_mode
    self.batch_size         = batch_size if not analytic_mode else 1
    self.window_size        = window_size if not analytic_mode else 0
    self.valid_test_split   = valid_test_split
    self.full_test_samples  = full_test_samples
    self.cross_validate     = cross_validate
    
    self.use_aug_transposed = use_aug_transposed
    self.use_aug_noised     = use_aug_noised    
    self.use_comp_samples   = use_comp_samples  
    self.use_solo_samples   = use_solo_samples  
    self.use_aug_validation = use_aug_validation

    self.shuffle_after_epochs       = shuffle_after_epochs
    self.shuffle_consecutive_frames = shuffle_consecutive_frames

    self.include_analytic_results = include_analytic_results
    self.include_mfcc             = include_mfcc
    self.include_rms              = include_rms
    self.binarize_labels          = binarize_labels
    self.highest_note             = highest_note
    self.include_empty_class      = include_empty_class
    
    self.filter_type          = filter_type
    self.lowpass_cutoff       = lowpass_cutoff
    self.highpass_cutoff      = highpass_cutoff
    self.normalize_spectra    = normalize_spectra
    self.spectra_scale        = spectra_scale
    self.include_empty_frames = include_empty_frames
    self.pad_size             = int(window_size//2) if (pad_frames and not analytic_mode) else 0
    self.reorder_cqt_bins     = reorder_cqt_bins

    
    
    self.samples = os.listdir(self.spectra_path)
    if 'DataLoader.pkl' in self.samples:
      self.samples.remove('DataLoader.pkl')
      
      with open(self.spectra_path+'DataLoader.pkl', 'rb') as file:
        self.dataLoaderDict = pickle.load(file)
        self.sr = self.dataLoaderDict['sampling_rate']
        self.class_num = ((librosa.note_to_midi(highest_note)    # shape of the labels
                          - librosa.note_to_midi(lowest_note))
                          + (1 if include_empty_class else 0)) if binarize_labels else 6
                          
        self.spectrum_type = self.dataLoaderDict['preproc_type']
        self.data_scale = 'db' if self.dataLoaderDict['amplitude_to_db'] else 'amplitude'
        self.cqt_bins_per_octave = self.dataLoaderDict['cqt_bins_per_oct']
        self.fmin = librosa.note_to_hz(self.dataLoaderDict['lowest_note'])
    else:
      print('Data Loader not found for the dataset! Please calibrate data generator manually!')
    
    
    self.configure_sample_set()
    self.samples.sort()
    
    if not self.analytic_mode:
      self.initialize_dataset()     # load dataset and preprocess it as necessary
      if valid_test_split is not None:
        self.split_samples()        # then split the dataset into training, validation and test sets
      
      
    self.on_epoch_end() # shuffle samples/frames for the first epock on creation



  # function to decide which samples should be used from the dataset
  def configure_sample_set(self):
        
    self.samples = np.array(self.samples) 
    
    if not (self.use_aug_noised and self.use_aug_transposed):
      self.noised_samples = self.samples[np.char.startswith(self.samples, 'AUG_NOISED')]
      self.transposed_samples = self.samples[np.char.startswith(self.samples, 'AUG_TRAN')]
      self.samples = self.samples[np.logical_not(np.char.startswith(self.samples, 'AUG'))]

      if self.use_aug_noised:
        self.samples = np.concatenate((self.samples, self.noised_samples))
      if self.use_aug_transposed:
        self.samples = np.concatenate((self.samples, self.transposed_samples))
    
    
    # filter out solo/comp type samples if required
    if not (self.use_solo_samples and self.use_comp_samples):
      if self.use_solo_samples:
        self.samples = self.samples[np.char.endswith(self.samples, 'solo', end=-4)]
      if self.use_comp_samples:
        self.samples = self.samples[np.char.endswith(self.samples, 'comp', end=-4)]
  
    return
  
  

    
  # function to return the generator's configuration dict, excluding data loaded into memory
  def get_generator_setup(self):
    data_keys = ['spectra', 'labels', 'frame_indices', 'chunk_sizes', 'chunks', 'test_chunks', 'validation_chunks']
    return {k:v for k,v in self.__dict__.items() if k not in data_keys}
    
  # function that returns the DataLoader of the preprocessed dataset
  def get_data_loader(self):
    return self.dataLoaderDict
    
  # function that returns the name of the preprocessed dataset directory
  def get_spectra_path(self):
    return self.spectra_path.split(os.path.sep)[-2]
  
  # function to save full generator setup
  def save_generator_setup(self):
    if not hasattr(self, 'generator_name'):
      self.generator_name = (f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")}_{self.spectrum_type}_'
                          + f'{np.sum(self.chunk_sizes)}:{np.sum(self.validation_chunk_sizes)}_{np.sum(self.test_chunk_sizes)}'
                          + f'{self.normalize_spectra if self.normalize_spectra is not None else ""}_{self.filter_type}_')
    with open(self.path_generators + self.generator_name + '.pkl', 'wb') as file:
      pickle.dump(self.__dict__, file)
    return 
    

  # function to split available samples into training, validation and test sets
  # the class instance running the function becomes the training generator
  # retrieve the corresponding validation and test data generators
  # with get_validation_generator() and get_test_generator() methods
  def split_samples(self):
  
    # generate the test set from the unaugmented samples
    sample_multiplicity = 1 + self.use_aug_noised + self.use_aug_transposed
    unaugmented_size = len(self.chunks) // (sample_multiplicity)
    if not self.full_test_samples:
      test_size = int(np.ceil(unaugmented_size * self.valid_test_split[1]))
      self.test_chunks = sorted(np.random.choice(self.chunks[:unaugmented_size], test_size, replace=False))
    else:
      unaug_sample_size = len(self.samples) // sample_multiplicity
      test_sample_size  = int(np.ceil(unaug_sample_size* self.valid_test_split[1]))
      self.test_samples = sorted(np.random.choice(np.arange(unaug_sample_size), test_sample_size, replace=False))

      test_chunk_amounts = np.ceil(self.sample_lengths[self.test_samples] / self.shuffle_consecutive_frames).astype(int)
      self.test_chunks, chunk_idx = np.empty(np.sum(test_chunk_amounts), dtype=int), 0
      
      for i, test_sample in enumerate(self.test_samples):
        sample_chunk_mask = np.where(np.logical_and(self.chunks >= self.sample_start_ind[test_sample],
                                                    self.chunks < self.sample_start_ind[test_sample]+self.sample_lengths[test_sample]))
        self.test_chunks[chunk_idx:chunk_idx+test_chunk_amounts[i]] = self.chunks[sample_chunk_mask]
        chunk_idx += test_chunk_amounts[i]
        



    idx_mask = np.ones(len(self.chunks))
    # remove the test chunks and augmented versions of the test set chunks from the training set to avoid overfitting
    for chunk in self.test_chunks:
      for i in range(sample_multiplicity):
        idx_mask = np.logical_and(idx_mask, np.roll((self.chunks != chunk), i * len(self.chunks)//sample_multiplicity))
    
    # remove chunks corresponding to the test set from chunk set
    self.chunks = self.chunks[idx_mask]
    self.test_chunk_sizes = self.chunk_sizes[np.logical_not(idx_mask)][:len(self.test_chunks)]
    self.chunk_sizes = self.chunk_sizes[idx_mask]
    

    
    # generate the validation set with the configured size ratio
    if not self.use_aug_validation and not self.cross_validate:  # if no augmentation is used for the validation set split the remaining chunks the same way as for the test set
      validation_size = int(np.ceil(unaugmented_size * self.valid_test_split[0])) 
      self.validation_chunks = np.random.choice(self.chunks[:len(self.chunks)//(sample_multiplicity)], validation_size, replace=False)
    else:
      # shuffle remaining chunks before splitting into folds
      perm = np.random.permutation(len(self.chunks))
      self.chunks, self.chunk_sizes = self.chunks[perm], self.chunk_sizes[perm]

      # split the remaining chunks into folds for possible cross-validation
      fold_cuts = np.arange(0, len(self.chunks), int(np.ceil(len(self.chunks)*self.valid_test_split[0])), dtype=int)[1:]
      self.folds = np.split(self.chunks, fold_cuts)
      
      self.current_fold = np.random.choice(len(self.folds))
      self.validation_chunks = self.folds[self.current_fold]
      sample_multiplicity = 1
    
      
    # remove chunks corresponding to the validation set from chunk set
    idx_mask = np.ones(len(self.chunks))
    for chunk in self.validation_chunks:
      for i in range(sample_multiplicity):    # remove augmented versions as well if they are not used for validation
        idx_mask = np.logical_and(idx_mask, np.roll((self.chunks != chunk), i * len(self.chunks)//sample_multiplicity))
    
    self.chunks = self.chunks[idx_mask]
    self.validation_chunk_sizes = self.chunk_sizes[np.logical_not(idx_mask)][:len(self.validation_chunks)]
    self.chunk_sizes = self.chunk_sizes[idx_mask]
    
    # create separate data generators for the validation and test sets that reference the same loaded data (shallow copies)
    self.validationGenerator = copy.copy(self)
    self.validationGenerator.chunks, self.validationGenerator.chunk_sizes = self.validation_chunks, self.validation_chunk_sizes
    self.validationGenerator.on_epoch_end()
    
    self.testGenerator = copy.copy(self)
    self.testGenerator.shuffle_after_epochs = False
    self.testGenerator.shuffle_consecutive_frames = 0
    self.testGenerator.chunks, self.testGenerator.chunk_sizes = self.test_chunks, self.test_chunk_sizes
    self.testGenerator.on_epoch_end()
    

    print(f'\tSamples successfully split!\n\tTraining set:\t{np.sum(self.chunk_sizes)} samples'.ljust(60)
            + f'\n\t\tValidation set:\t{np.sum(self.validation_chunk_sizes)} samples'
            + f'\n\t\tTest set:\t{np.sum(self.test_chunk_sizes)} samples')
    return
    
    

  # function to retrieve the validation set data generator for this instance's training set
  def get_validation_generator(self):
    return self.validationGenerator
    
  # function to retrieve the test set data generator for this instance's training set
  def get_test_generator(self):
    return self.testGenerator
    
  # function to reset all generators so they can be reused
  def reset_generators(self):
    self.on_epoch_end()
    self.validationGenerator.on_epoch_end()
    self.testGenerator.on_epoch_end()
    return
  
  
  # function that returns the amount of folds available for cross-validation
  def fold_number(self):
    return len(self.folds) if self.cross_validate else 1
    
  # function to repartition the training and validation sets into the next fold for cross-validation
  def next_fold(self):
    if self.fold_number() < 2:
      print('No more folds available for cross-validation!')
      return
    self.current_fold = (self.current_fold + 1) % self.fold_number() # increment fold number
    
    self.chunks = np.concatenate((self.chunks, self.validation_chunks))     # reset the available chunk pool
    self.chunk_sizes = np.concatenate((self.chunk_sizes, self.validation_chunk_sizes))
    
    self.validation_chunks = self.folds[self.current_fold]                # select new validation fold
    
    
    # remove chunks corresponding to the new validation set
    idx_mask = np.ones(len(self.chunks))
    for chunk in self.validation_chunks:
      idx_mask = np.logical_and(idx_mask, (self.chunks != chunk))
    
    # set the new training and validation chunk sets and update the validation generator
    self.chunks = self.chunks[idx_mask]
    self.validation_chunk_sizes = self.chunk_sizes[np.logical_not(idx_mask)][:len(self.validation_chunks)]
    self.chunk_sizes = self.chunk_sizes[idx_mask]
    self.validationGenerator.chunks, self.validationGenerator.chunk_sizes = self.validation_chunks, self.validation_chunk_sizes
    
    self.reset_generators()
    return

  @staticmethod
  # function to binarize labels
  def labels_to_binary(labels, class_num, include_empty_class):
    binarized_labels = np.zeros((labels.shape[0], class_num), dtype=int)
    for frame, label in enumerate(labels):
      label = label if include_empty_class else (label[np.nonzero(label)]-1)
      binarized_labels[frame, label[label < class_num]] = 1
    return binarized_labels

  @staticmethod
  # function to reorder rows of spectra into groupings by semitones
  def spectra_to_semitone_groups(spectra, bins_per_octave=36, fmin=librosa.note_to_hz('C2')):
    freqs = [(librosa.hz_to_midi(a)) for 
             a in librosa.cqt_frequencies(n_bins=spectra.shape[0],bins_per_octave=bins_per_octave,fmin=fmin)]
    semitones = np.rint(np.array(freqs)).astype('int') % 12
    
    reordered_spectra = np.zeros(spectra.shape)
    octaves = len(freqs)//bins_per_octave
    row=0
    for semitone in range(12):
      semitone_rows = np.where(semitones == semitone)[0]
      reordered_spectra[row:row+len(semitone_rows)] = spectra[semitone_rows]
      row = row + len(semitone_rows)
    
    return reordered_spectra
    
  @staticmethod
  # function to reorder rows of spectra into semitone groupings in the order of the circle of fifths
  def spectra_to_circle_fifths(spectra, bins_per_octave=36, fmin=librosa.note_to_hz('C2')):
    freqs = [(librosa.hz_to_midi(a)) for 
             a in librosa.cqt_frequencies(n_bins=spectra.shape[0],bins_per_octave=bins_per_octave,fmin=fmin)]
    semitones = np.rint(np.array(freqs)).astype('int') % 12
    
    reordered_spectra = np.zeros(spectra.shape)
    octaves = len(freqs)//bins_per_octave
    row=0
    for semitone in [7*i % 12 for i in range(8,20)]:
      semitone_rows = np.where(semitones == semitone)[0]
      reordered_spectra[row:row+len(semitone_rows)] = spectra[semitone_rows]
      row = row + len(semitone_rows)
    
    return reordered_spectra


  
  @staticmethod
  # function to create a low-, high-, or bandpass IIR filter and return its frequency response at the freq_bins
  def create_filter(freq_bins, filter_type='bessel', low_cutoff=librosa.note_to_hz('E2'), high_cutoff=librosa.note_to_hz('C6'),
                    filter_order=None, rp=None, rs=None, expand_passband=None):

    # optionally generate a random filter (within usable limits)
    if filter_type == 'random':
      filter_type = np.random.choice(['bessel', 'butter', 'cheby1', 'cheby2', 'ellip'])
      
      # randomize cutoff frequency or optionally set it to None (-> low- or highpass filter)
      low_cutoff = np.random.choice(np.linspace (librosa.note_to_hz('C2'), librosa.note_to_hz('F2')))
      low_cutoff = np.random.choice([low_cutoff, None], p=[0.9, 0.1])
      
      high_cutoff = np.random.choice(np.linspace(librosa.note_to_hz('A5'), librosa.note_to_hz('C8')))
      high_cutoff = np.random.choice([high_cutoff, None], p=[0.9, 0.1]) if low_cutoff is not None else high_cutoff

      # randomize other parameters from an experimentally validated range
      filter_order = np.random.choice({'bessel':3, 'butter':4, 'cheby1':5, 'cheby2':2, 'ellip':4}[filter_type]) + 1
      expand_passband = np.random.choice(np.linspace(*{'bessel':(0.85, 2.0), 'butter':(0.9, 1.75), 'cheby1':(1.0, 1.8), 'cheby2':(1.0, 1.5), 'ellip':(0.9, 1.6)}[filter_type]))
      rp = np.random.choice(np.linspace(*{'bessel':(1,1), 'butter':(1,1), 'cheby1':(0.1, 2.5), 'cheby2':(1,1), 'ellip':(0.1, 2.5)}[filter_type]))
      rs = np.random.choice(np.linspace(*{'bessel':(1,1), 'butter':(1,1), 'cheby1':(1,1), 'cheby2':(8, 20), 'ellip':(40, 60)}[filter_type]))



    # load some default values for filters when not specified
    if filter_order is None:    # order of the IIR filter to be created
      filter_order = {'bessel':1, 'butter':4, 'cheby1':1, 'cheby2':2, 'ellip':2}[filter_type]
    if expand_passband is None: # optionally expand window by factoring cutoff frequencies
      expand_passband = {'bessel':1.1, 'butter':1, 'cheby1':1.4, 'cheby2':1, 'ellip':1}[filter_type]
    if rp is None:              # maximum ripple (dB) for Chebyshev/elliptic filters
      rp = {'bessel':1, 'butter':1, 'cheby1':10, 'cheby2':10, 'ellip':1}[filter_type]
    if rs is None:              # minimum attenuation (dB) in stopband for Chebyshev/elliptic filters
      rs = {'bessel':1, 'butter':1, 'cheby1':10, 'cheby2':10, 'ellip':60}[filter_type]

    btype = 'highpass' if high_cutoff is None else ('lowpass' if low_cutoff is None else 'bandpass')
    cutoff = (low_cutoff/expand_passband if high_cutoff is None else 
            (high_cutoff*expand_passband if low_cutoff is None else 
              np.array([low_cutoff/expand_passband, high_cutoff*expand_passband])))
    

    # create IIR filter of the selected type and calculate frequency response on given frequency bins
    b, a = scipy.signal.iirfilter(filter_order, cutoff*2*np.pi, rp=rp, rs=rs, btype=btype, ftype=filter_type, analog=True)
    _, freq_response = scipy.signal.freqs(b, a, worN=freq_bins * 2*np.pi) # Hz freqs -> rad/s
    
    return abs(freq_response)[:, np.newaxis], (filter_type, filter_order, expand_passband, rp, rs, low_cutoff, high_cutoff)

  
  @staticmethod
  # function to optionally preprocess input spectra by padding, normalizing, filtering and reordering
  def preprocess_spectra(spectra, dataGenerator=None, pad_size=4, normalize='framewise', input_scale='db', output_scale='db', filter_type='bessel',
    lowpass_freq=82, highpass_freq=None, reorder_cqt_bins=None, sr=22050, spectra_type='CQT', cqt_bins_per_octave=36, cqt_fmin='C2'):
    
    # if a data generator is passed as an argument, take parameters from its state
    if dataGenerator is not None:
      pad_size=dataGenerator.pad_size
      normalize=dataGenerator.normalize_spectra
      input_scale=dataGenerator.data_scale
      output_scale=dataGenerator.spectra_scale
      filter_type=dataGenerator.filter_type
      lowpass_freq=dataGenerator.lowpass_cutoff
      highpass_freq=dataGenerator.highpass_cutoff
      reorder_cqt_bins=dataGenerator.reorder_cqt_bins
      sr=dataGenerator.sr
      spectra_type = dataGenerator.spectrum_type
      cqt_bins_per_octave=dataGenerator.cqt_bins_per_octave
      cqt_fmin=dataGenerator.fmin
      if hasattr(dataGenerator, 'preproc_filter'):
        preproc_filter = dataGenerator.preproc_filter

    # return to linear amplitude scale if necessary
    if input_scale == 'db':
      spectra = librosa.db_to_amplitude(spectra)
    
    
    # if filter is not available, create one specific for the spectra parameters
    if filter_type is not None:
      
      if dataGenerator is None or not hasattr(dataGenerator, 'preproc_filter'): # create filter if one is not available
        
        if spectra_type == 'FFT':
          freq_bins = librosa.fft_frequencies(sr=sr, n_fft=(spectra.shape[0]-1)*2)
        if spectra_type in ['CEPS', 'PCEPS']: # create frequency bins corresponding to the cepstral quefrency bins
          freq_bins = librosa.fft_frequencies(sr=((spectra.shape[0]-1)*4)/sr, n_fft=(spectra.shape[0]-1)*2)
          np.true_divide(1.0, freq_bins, where=(freq_bins != 0), out=freq_bins)
          freq_bins[0] = 0.0
        if spectra_type in ['CQT', 'VQT']:
          freq_bins = librosa.cqt_frequencies(n_bins=spectra.shape[0],bins_per_octave=cqt_bins_per_octave,fmin=cqt_fmin)
        if spectra_type == 'AUTOCORR':
          freq_bins = np.ones(spectra.shape[0])*sr
          np.true_divide(freq_bins, np.arange(spectra.shape[0]), where=(np.arange(spectra.shape[0]) != 0), out=freq_bins)
        
        preproc_filter, _ = DataGenerator.create_filter(freq_bins, filter_type, lowpass_freq, highpass_freq)
        
        if dataGenerator is not None and filter_type != 'random':   # save the filter to the data generator (if used through one)
          dataGenerator.preproc_filter = preproc_filter                      # unless random filtering is selected
    
      # perform filtering in the frequency domain using frequency responses
      sepctra = spectra * preproc_filter
    
    
    
    if output_scale == 'db':
      spectra = librosa.amplitude_to_db(spectra, ref=np.max(spectra))
    
    
    # normalize the input spectra
    # either framewise, frequency channelwise or considering the full sample
    if normalize is not None:
      
      if normalize == 'full':
        mean = np.mean(spectra)
        std = np.nanstd(spectra)
      if normalize == 'framewise':
        mean = np.mean(spectra, axis=0)
        std = np.nanstd(spectra, axis=0)
      if normalize == 'freqwise':
        mean = np.mean(spectra, axis=1)[:, np.newaxis]
        std = np.nanstd(spectra, axis=1)[:, np.newaxis]
      
      np.true_divide((spectra - mean), std, out=spectra, where=(std != 0))
    
    
    if reorder_cqt_bins is not None and dataGenerator is not None:
      spectra = (dataGenerator.spectra_to_semitone_groups(spectra, cqt_bins_per_octave, cqt_fmin) if reorder_cqt_bins == 'semitone' 
            else dataGenerator.spectra_to_circle_fifths(spectra, cqt_bins_per_octave, cqt_fmin))
    
    
    # pad the input in the time dimension so all valid frames can be processed equally
    spectra = np.pad(spectra, ((0, 0), (pad_size, pad_size)), mode='edge')
    
    return spectra
  
  
  # function to load/prepare a dataset for generating input to a model
  # includes optional preprocessing steps like filtering and normalization
  def initialize_dataset(self):
    print('Initializing Data Generator!')
    
    # load a random sample to get basic data
    with open(self.spectra_path + np.random.choice(self.samples), 'rb') as file:
      calibration_sample = pickle.load(file)
      self.feature_size = calibration_sample['spectra'].shape[0]

    
    # get RAM capacities of the system and size of uncompress
    if os.name == 'nt':     # Windows OS
      mem_data = os.popen('wmic os get TotalVisibleMemorySize,FreePhysicalMemory -VALUE').readlines()
      for line in mem_data:
        if line.startswith('TotalVisibleMemorySize'):
          total_memory = float(line.split('=')[-1]) / (1024**2)
        if line.startswith('FreePhysicalMemory'):
          free_memory = float(line.split('=')[-1]) / (1024**2)
    if os.name == 'posix':  # UNIX OS
      import psutil
      mem_data = psutil.virtual_memory()
      total_memory, free_memory = mem_data.total/(1024**3), mem_data.available/(1024**3)
        
        
    used_memory = total_memory - free_memory
    dataset_size = sum(os.path.getsize(self.spectra_path + f) for f in self.samples if os.path.isfile(self.spectra_path + f)) / (1024**3)

    print(f'\tRAM: {(used_memory/total_memory)*100:.2f}% used of {total_memory:.2f} GB. {free_memory:.2f} GB of RAM available.')
    print(f'\tSampleset size: {dataset_size:.2f} GB - {(dataset_size/free_memory)*100:.2f}% of available RAM.')
    
    
    if 1.1*dataset_size > free_memory: # account for memory overhead
      print('\tNot enough RAM to safely load entire dataset into memory.')
      quit() # TODO: load only a batch at a time (preprocess samples into smaller chunks first)
    else:
      print('\tLoading dataset into memory!', end='')
      
      loading_start = time.time()
      # count total amount of frames and create an index array for each full sample and one for each valid frames
      self.total_frames, spectra_start_ind, self.frame_indices = 0, np.empty(len(self.samples), dtype=int), np.empty((0), dtype=int)
      self.sample_lengths, self.sample_start_ind = np.empty(len(self.samples), dtype=int), np.empty(len(self.samples), dtype=int)
      aug_samples_start = 0
      for i, filename in enumerate(self.samples):
        with open(self.spectra_path + filename, 'rb') as file:
          data = pickle.load(file)
          spectra_start_ind[i] = int(self.total_frames) + self.pad_size   # account for possible padding
          self.total_frames += data['labels'].shape[0] + 2*self.pad_size
          current_indices = np.arange(spectra_start_ind[i], self.total_frames-self.pad_size, dtype=int)
          
          if not self.include_empty_frames: # optionally discard indices to frames with no notes being played
            current_indices = current_indices[np.where(np.count_nonzero(data['labels'], axis=1))]
          
          self.sample_lengths[i] = len(current_indices)   # save sample lengths and corresponding start indices
          self.sample_start_ind[i] = (self.sample_lengths[i-1] + self.sample_start_ind[i-1]) if i else 0
          self.frame_indices = np.concatenate((self.frame_indices, current_indices), axis=0)
          
          if filename.startswith('AUG') and not aug_samples_start:
            aug_samples_start = self.sample_start_ind[i]
          
      
      
      # optionally load MFCC data, analytic model predictions or movin RMS values to merge with the base input
      if self.include_mfcc is not None:
        with open(self.path_data + 'preprocessed' + os.path.sep + self.include_mfcc + os.path.sep + np.random.choice(self.samples), 'rb') as file:
          calibration_sample = pickle.load(file)
          self.mfcc_size = calibration_sample['spectra'].shape[0]
        self.mfcc = np.empty((self.mfcc_size, self.total_frames))

      if self.include_analytic_results is not None:
        analytic_preds_file = os.listdir(self.path_data + 'models' + os.path.sep + self.include_analytic_results + os.path.sep)[0]
        with open(self.path_data + 'models' + os.path.sep + self.include_analytic_results + os.path.sep + analytic_preds_file, 'rb') as file:
          analytic_data = pickle.load(file)['preds']
        self.analytic_predictions = np.empty((self.total_frames, self.class_num))
      

      if self.include_rms is not None:
        self.rms = np.zeros(self.total_frames)
        
      
      # load main input spectra and labels into memory
      self.spectra, self.labels = np.empty((self.feature_size, self.total_frames)), np.empty((self.total_frames, self.class_num))
      for i, filename in enumerate(self.samples):
        print(f'\r\tLoading dataset into memory! Sample {i+1}/{len(self.samples)}'.ljust(60), end='')
        with open(self.spectra_path + filename, 'rb') as file:
          data = pickle.load(file)
          # preprocess the spectra by optionally padding, normalizing and filtering them
          sample_spectra = self.preprocess_spectra(data['spectra'], dataGenerator=self)
          self.spectra[:, spectra_start_ind[i] -self.pad_size : spectra_start_ind[i]+data['labels'].shape[0] +self.pad_size] = sample_spectra
          
          # preprocess labels by optionally encoding them in a binary vector format
          sample_labels = (self.labels_to_binary(data['labels'], self.class_num, self.include_empty_class) if self.binarize_labels else data['labels'])
          self.labels[spectra_start_ind[i]:spectra_start_ind[i]+data['labels'].shape[0]] = sample_labels
          
          
      
        # optionally load MFCC data and add them to the input
        if self.include_mfcc is not None:
          with open(self.path_data + 'preprocessed' + os.path.sep + self.include_mfcc + os.path.sep + filename, 'rb') as file:
            self.mfcc[:, spectra_start_ind[i] : spectra_start_ind[i]+data['labels'].shape[0]] = pickle.load(file)['spectra']
            
        # optionally load results from an analytic model's predictions
        if self.include_analytic_results is not None:
          sample_analytic_preds = (self.labels_to_binary(analytic_data[i], self.class_num, self.include_empty_class) if self.binarize_labels else analytic_data[i])
          self.analytic_predictions[spectra_start_ind[i]:spectra_start_ind[i]+data['labels'].shape[0]] = sample_analytic_preds
          
          
        # optionally load moving RMS results
        if self.include_rms is not None:
          with open(self.path_data + 'preprocessed' + os.path.sep + self.include_rms + os.path.sep + filename, 'rb') as file:
            self.rms[spectra_start_ind[i] : spectra_start_ind[i]+data['labels'].shape[0]] = pickle.load(file)['spectra']
          
          
          
      self.total_frames = len(self.frame_indices)
      print(f'\r\tDataset loaded and preprocessed in {time.time()-loading_start:.2f} seconds.')
      
    
    
    # create index array according to the chunk sizes corresponding to consecutive frame windows
    # (or the sample start indices if samples frames are only shuffled together)
    if self.shuffle_consecutive_frames:
      # get chunks for unaugmented part of the dataset
      aug_samples_start = aug_samples_start if aug_samples_start else self.total_frames
      unaug_sample_lens = self.sample_lengths[np.where(self.sample_start_ind < aug_samples_start)]
      unaug_chunk_amnts = np.ceil(unaug_sample_lens/self.shuffle_consecutive_frames).astype(int)

      # split samples into chunks and save index arrays and the length of the chunks
      self.chunks, self.chunk_sizes = np.empty(np.sum(unaug_chunk_amnts), dtype=int), np.empty(np.sum(unaug_chunk_amnts), dtype=int)
      chunk_idx, chunk_counter = 0, 0
      for i, length in enumerate(unaug_sample_lens):
        self.chunks[chunk_idx:chunk_idx+unaug_chunk_amnts[i]] = np.arange(chunk_counter, chunk_counter+length, self.shuffle_consecutive_frames)
        self.chunk_sizes[chunk_idx:chunk_idx+unaug_chunk_amnts[i]] = self.shuffle_consecutive_frames * np.ones(unaug_chunk_amnts[i], dtype=int)
        self.chunk_sizes[chunk_idx+unaug_chunk_amnts[i]-1] = length % self.shuffle_consecutive_frames # last chunk size may be shorter
        chunk_idx += unaug_chunk_amnts[i]
        chunk_counter += length
      
      # chunks will be mirrored for the augmented samples
      aug_noised, aug_transposed = np.empty(0, dtype=int), np.empty(0, dtype=int)
      aug_noised_sizes, aug_transposed_sizes = np.empty(0, dtype=int), np.empty(0, dtype=int)
      if self.use_aug_noised:
        aug_noised = self.chunks + aug_samples_start
        aug_noised_sizes = self.chunk_sizes
        aug_samples_start += aug_samples_start
      if self.use_aug_transposed:
        aug_transposed = self.chunks + aug_samples_start
        aug_transposed_sizes = self.chunk_sizes
        aug_samples_start += aug_samples_start
        
      self.chunks = np.concatenate((self.chunks, aug_noised, aug_transposed))
      self.chunk_sizes = np.concatenate((self.chunk_sizes, aug_noised_sizes, aug_transposed_sizes))
      
    else:
      self.chunks = np.array(self.sample_start_ind, copy=True)
      self.chunk_sizes = self.sample_lengths
    



    self.x_shape = (self.feature_size, self.window_size)
    self.y_shape = (self.class_num)
    
    
  # OVERLOADED FUNCTIONS USED BY TENSORFLOW/KERAS MODELS
  # function that returns the number of batches per epoch
  def __len__(self):
    return self.samples.size if self.analytic_mode else int(np.floor((np.sum(self.chunk_sizes)/self.batch_size)))

  # function that generates and returns a batch of samples (#ind)
  def __getitem__(self, ind):
  
    if self.analytic_mode: # in case of analytic models, return full sample at the index
      with open(self.spectra_path + self.samples[ind], 'rb') as file:
        sample = pickle.load(file)
        if self.filter_type is not None or self.normalize_spectra is not None or self.spectra_scale != self.data_scale:
          self.preprocess_spectra(sample['spectra'], dataGenerator=self)
      return (sample['spectra'], sample['labels'])
    
    
    self.curr_batch_idx += 1
    if self.curr_batch_idx == self.__len__():
      self.on_epoch_end()
    
    # if this is the last batch, set the size accordingly
    batch_size = self.batch_size if self.curr_batch_idx < self.__len__() else (np.sum(self.chunk_sizes) % self.batch_size)
    
    indices, count = np.empty(batch_size, dtype=int), 0
    #print('\t\t', ind, self.curr_batch_idx, batch_size, self.curr_chunk_idx, self.chunks.shape, self.chunk_sizes.shape, flush=True)
    
    while batch_size-count:
      chunk_size = self.chunk_sizes[self.curr_chunk_idx]
      
      if batch_size-count >= chunk_size - self.chunk_subidx+1:     # if the rest of the chunk fits into the batch
        indices[count:count + chunk_size-self.chunk_subidx] = self.frame_indices[self.chunks[self.curr_chunk_idx] + np.arange(self.chunk_subidx, chunk_size)]
        count += chunk_size - self.chunk_subidx
        self.curr_chunk_idx, self.chunk_subidx = self.curr_chunk_idx+1, 0
      else:                                                             # if the chunk only partially fits into the rest of the batch
        indices[count:batch_size] = self.frame_indices[self.chunks[self.curr_chunk_idx] + np.arange(self.chunk_subidx, self.chunk_subidx + batch_size-count)]
        self.chunk_subidx += batch_size - count
        count = batch_size
    
    x, y = np.empty((batch_size, *self.x_shape)), self.labels[indices]
    
    # create batch from the gathered frame indexes with the required window size
    for frame in range(batch_size):
      x[frame] = self.spectra[:, indices[frame]-self.pad_size : indices[frame]+self.pad_size+1]
    
    
    
      # append optional data to the input
    if self.include_mfcc or self.include_analytic_results or self.include_rms:
      x = [x]
    
      if self.include_mfcc:
        x.append(self.mfcc[:,indices].T)
      if self.include_analytic_results:
        x.append(self.analytic_predictions[indices])
      if self.include_rms:
        x_rms = np.empty((batch_size, self.window_size))
        for frame in range(batch_size):
          x_rms[frame] = self.rms[indices[frame]-self.pad_size : indices[frame]+self.pad_size+1]
        x.append(x_rms)
        
        
    return x, y


  # function to reset (and optionally shuffle) the samples after each epoch
  def on_epoch_end(self):
    if self.analytic_mode and self.shuffle_after_epochs: # shuffle samples when fitting on analytic models in case of early stopping
      np.random.shuffle(self.samples)
    else:
      self.curr_batch_idx, self.curr_chunk_idx, self.chunk_subidx = 0, 0, 0   # reset the chunk and frame indexes
      
      if self.shuffle_consecutive_frames:
        self.chunk_diddle = np.random.randint(self.shuffle_consecutive_frames) # TODO index to diddle the start indices of chunks
      if self.shuffle_after_epochs:
        new_permutation = np.random.permutation(len(self.chunks))
        self.chunks, self.chunk_sizes = self.chunks[new_permutation], self.chunk_sizes[new_permutation]
        
