import sys, os, time, datetime
import jams, librosa, pickle
import numpy as np


path_data_src = {
  'cln' : {'path' : 'audio_hex-pickup_debleeded', 'suffix' : '_hex_cln'},
  'org' : {'path' : 'audio_hex-pickup_original', 'suffix' : '_hex'},
  'mic' : {'path' : 'audio_mono-mic', 'suffix' : '_mic'},
  'mix' : {'path' : 'audio_mono-pickup_mix', 'suffix' : '_mix'}}

  
# Main data loader class that handles basic preprocessing of the dataset for later handling
class DataLoader:
  
  def __init__(self, data_src='cln',      # audio data source type from the ones available in GuitarSet dataset
                sampling_rate=22050,      # sampling rate for downsampling the audio files
                window_size=2048,         # window size used when 
                window_overlap=4,         # multiplicative inverse of the overlap between subsequent windows (e.g. step_size 4 => 25% overlap)
                file_type='.wav',         # extension for the file type used to generate the data
                max_pitches=6,            # maximum number of pitches that will be available in a frame
                
                # Parameters for the audio preprocessing methods
                preproc_type='CQT',       # type of preprocessing to apply on the input (includes FFT, CQT, VQT, CEPS, PCEPS, MELCEPS, MFCC, AUTOCORR, RMS)
                lowest_note='C2',         # lowest note to consider for the spectra (where relevant, like CQT)
                octave_number=7,          # number of octaves to consider above the lowest note for the spectra (where relevant, like CQT)
                amplitude_to_db=True,     # flag to determine if spectra amplitudes are to be converted to the Decibel scale (logarithmic)
                shift_midi_notes=True,    # flag to determine if midi notes should be shifted to account for lowest note possible in the labels
                smooth_label_width=1,     # width of the median filter used to smooth the labels
                win_type='hann',          # window function to apply on the audio frames before processing
                                          #   (includes hann, hamming, blakcman, blackmanharris, nuttall, boxcar, triang)
                cqt_bins_per_oct=36,      # number of bins per octave when preprocessing via CQT
                cqt_filter_scale=1.0,     # factor by which to scale CQT/VQT filters (<1 -> shorter windows -> better time resolution)
                vqt_gamma=None,           # bandwidth offset for VQT filter lengths
                                          #   (=0 -> VQT=CQT;    >0 -> smaller Q at lower freqs -> better time resolution at lower freqs,
                                          #   None defaults so filter bwidths are proportional to ERBs, so gamma~=4.4 at 36 semitones per octave)
                mfcc_banks=40,            # number of mel-frequency cepstral coefficients (corresponding to mel-scale filter banks) when extracting MFCC features
                rms_size=2048,            # size of the window used for calculating moving RMS values
                aug_base=None,            # preprocessed directory for which the augmented data should match
                
                # Parameters for optional data augmentation
                augment_samples=True,     # flag for whether to create additional samples using data augmentation methods
                transpose_semitones_rng=3,# maximum number of semitones to transpose the data during preprocessing
                transpose_down=False,     # whether to transpose audio downwards or only upwards
                white_noise_std=0.002,    # standard deviation of the white noise added during preprocessing
                salt_pepper_prb=0.0001,   # probability of salt and pepper noise added during preprocessing
                
                ):
  
    self.data_src       = path_data_src[data_src]
    self.sampling_rate  = sampling_rate
    self.file_type      = file_type
    self.max_pitches    = max_pitches
    
    self.preproc_type        = preproc_type
    self.lowest_note         = lowest_note
    self.octave_number       = octave_number
    self.amplitude_to_db     = amplitude_to_db
    self.window_size         = window_size
    self.window_overlap      = window_overlap
    self.step_size           = self.window_size // self.window_overlap
    self.win_type            = win_type
    self.cqt_bins_per_oct    = cqt_bins_per_oct
    self.cqt_filter_scale    = cqt_filter_scale
    self.vqt_gamma           = vqt_gamma
    self.mfcc_banks          = mfcc_banks
    self.rms_size            = np.minimum(rms_size, window_size)
    self.smooth_label_width  = smooth_label_width
    
    self.augment_samples          = augment_samples
    self.aug_base                 = aug_base
    self.transpose_down           = transpose_down
    self.shift_midi_notes         = librosa.note_to_midi('E2')-1 - (transpose_semitones_rng if transpose_down and shift_midi_notes else 0)
    self.transpose_semitones_rng  = transpose_semitones_rng
    self.white_noise_std          = white_noise_std
    self.salt_pepper_prb          = salt_pepper_prb
    
    
    self.path_dataset     = ((os.path.dirname(os.path.abspath(__file__)) + os.path.sep) if '__file__' in globals() 
                            else 'gdrive/MyDrive/MSc Diploma/') + 'data' + os.path.sep # colab support for mounted Drive
    self.path_annotations = self.path_dataset + 'annotation' + os.path.sep
    self.path_data        = self.path_dataset + self.data_src['path'] + os.path.sep
    self.path_save        = (self.path_dataset + 'preprocessed' + os.path.sep + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + preproc_type 
                            + (str(cqt_bins_per_oct) if preproc_type in ['CQT', 'VQT'] else '')
                            + (('_' + str(cqt_filter_scale)) if preproc_type in ['CQT', 'VQT'] else '')
                            + (('_' + ('EBR' if vqt_gamma is None else str(vqt_gamma))) if preproc_type=='VQT' else '')
                            + (str(mfcc_banks) if preproc_type=='MFCC' else '')
                            + (str(rms_size) if preproc_type=='RMS' else '')
                            + (('_LOW' + lowest_note) if lowest_note != 'C2' else '')
                            + (('_OCT' + str(octave_number)) if octave_number != 7 else '')
                            + '_' + win_type + str(window_size) + '_' + str(self.step_size) + '_'
                            + ('DBSCALE_' if amplitude_to_db else '') + ('AUG_' if augment_samples else '') + str(sampling_rate) 
                            + self.data_src['suffix'] + os.path.sep)
    
    if aug_base is not None and os.path.exists(self.path_dataset + 'preprocessed' + os.path.sep + aug_base):
      base_samples = np.array(os.listdir(self.path_dataset + 'preprocessed' + os.path.sep + aug_base))
      base_samples = base_samples[np.char.startswith(base_samples, 'AUG_TRAN')]
      self.base_transposed = {}
      for sample in base_samples:
        sample_name = sample.split('_')
        transp_amt, transp_dir = sample_name[3][-2], sample_name[3][-1]
        sample_name[3] = sample_name[3][:-2]
        sample_name[-1] = sample_name[-1][:-4]
        self.base_transposed['_'.join(sample_name[2:])] = (transp_amt, transp_dir)
        
      print('Augmentation base found! Matching augmented data!')
    else:
      print('Augmentation base not found!')
    
    
    if not os.path.exists(self.path_dataset + 'preprocessed' + os.path.sep):
      os.makedirs(self.path_dataset + 'preprocessed' + os.path.sep)
    
  
  # main function to preprocess all data from the designated input dataset
  def preprocess_data(self):
    samples = os.listdir(self.path_annotations)
    sample_size = len(samples)
    
    
    if not os.path.exists(self.path_save):
      os.makedirs(self.path_save)
    
    preprocessing_start = time.time()
    for i in range(sample_size):
      filename = samples[i][:-5]
      print(('\rPreprocessing dataset: ' + str(i+1) + '/' + str(sample_size) + ' - ' + filename).ljust(80), end='')
      
      self.preprocess_sample(filename)
    
    self.processing_time = time.time() - preprocessing_start
    print(f'Preprocessing dataset: COMPLETED in {self.processing_time:.2f} seconds')
    print(f'    Preprocessed samples saved to: {self.path_save}')
    
    # save the data loader specifications for reference
    with open(self.path_save + 'DataLoader.pkl', 'wb') as file:
      pickle.dump(self.__dict__, file)
    return 
    
  
  # function to preprocess a single sample and its labels and save it for later processing
  def preprocess_sample(self, filename):
    
    audio, labels = self.load_sample(filename)
    
    data = {'spectra' : self.preprocess_audio(audio), 'labels' : self.preprocess_labels(labels)}
    self.save_sample(filename, data) 
    
    
    # create augmented samples with the chosen modifications
    if self.augment_samples:
      self.augment_sample(audio, labels, filename)
    
    return
  
  
  # function to load and prepare the audio data and labels belonging to a single sample
  def load_sample(self, filename, file_suffix=None, sampling_rate=None):
    if file_suffix is None:
      file_suffix = self.data_src['suffix']
    if sampling_rate is None:
      sampling_rate = self.sampling_rate
    
    jam_file = jams.load(self.path_annotations + filename + '.jams')
    
    path_audio = self.path_data + jam_file['file_metadata']['title'] + file_suffix + self.file_type
    audio, sr = librosa.load(path_audio, sr=sampling_rate)
    
    # pad the audio signal to fit the window sizes
    if len(audio) % self.step_size:
      audio = np.append(audio, np.ones([self.step_size - len(audio) % self.step_size], dtype=float)*audio[-1])
    
    # We choose the MIDI note annotation as the ground truth
    # and subsample it to fit the windows from our audio sample
    
    subsample_pts = librosa.frames_to_time(range(len(audio)//self.step_size - self.window_overlap + 1),
                                          sr=self.sampling_rate, hop_length=self.step_size)
    subsample_pts = subsample_pts + self.window_size/(2*self.sampling_rate) # adjust time points to window centers
    
    labels = np.empty([self.max_pitches, len(subsample_pts)], dtype=int)
    for string in range(self.max_pitches):
      string_notes = jam_file.annotations['note_midi'][string].to_samples(subsample_pts)
      string_notes = np.array([tmp[0] if tmp else 0 for tmp in string_notes]) # use 0 for frames where the string is not ringing
      labels[string][:] = np.rint(string_notes)                                # round midi notes to integers
    
    return audio, labels
    
  
  def preprocess_audio(self, audio, preproc_type=None):
    if preproc_type is None:
      preproc_type = self.preproc_type
    
    if preproc_type == 'CQT':
      spectra = np.abs(librosa.cqt(audio,
                      sr=self.sampling_rate, 
                      hop_length=self.step_size, 
                      n_bins=self.cqt_bins_per_oct * self.octave_number, 
                      bins_per_octave=self.cqt_bins_per_oct,
                      filter_scale=self.cqt_filter_scale,
                      fmin=librosa.note_to_hz(self.lowest_note),
                      window=self.win_type))
    
    if preproc_type == 'VQT':
      spectra = np.abs(librosa.vqt(audio,
                      sr=self.sampling_rate, 
                      hop_length=self.step_size, 
                      n_bins=self.cqt_bins_per_oct * self.octave_number, 
                      bins_per_octave=self.cqt_bins_per_oct,
                      filter_scale=self.cqt_filter_scale,
                      gamma=self.vqt_gamma,
                      fmin=librosa.note_to_hz(self.lowest_note),
                      window=self.win_type))
    
    
    
    if preproc_type in ['FFT', 'CEPS', 'PCEPS']:
      spectra = np.abs(librosa.stft(audio,
                      hop_length=self.step_size, 
                      n_fft=self.window_size,
                      window=self.win_type))
    
    
    
    if preproc_type in ['CEPS', 'PCEPS']:
      spectra = np.abs(np.fft.rfft(np.log(spectra.T)).T)
      
    if preproc_type == 'PCEPS':
      spectra = 4*np.square(spectra)
      
      
      
    if preproc_type == 'AUTOCORR':
      spectra = np.empty((self.window_size, len(audio)//self.step_size - self.window_overlap+1))
      for frame in range(spectra.shape[1]):
        win_func = {'hann':np.hanning, 'hamming':np.hamming, 'blackman':np.blackman}.get(self.win_type, np.ones)(self.window_size)
        spectra[:,frame] = np.maximum(librosa.autocorrelate(win_func*audio[frame*self.step_size:frame*self.step_size + self.window_size]), 0)
      
      
    if preproc_type =='MFCC':
      spectra = librosa.feature.mfcc(y=audio,
                      sr=self.sampling_rate, 
                      hop_length=self.step_size, 
                      n_fft=self.window_size,
                      n_mfcc=self.mfcc_banks,
                      window=self.win_type)
                      
                      
    
      
    if preproc_type == 'RMS':     # moving RMS
      spectra = np.empty(len(audio)//self.step_size - self.window_overlap+1)
      frame_centers = np.linspace(self.window_size//2, (len(audio)-self.window_size//2), (len(audio)-self.window_size)//self.step_size + 1, dtype=int)
      for i, center in enumerate(frame_centers):   # calculate RMS on for a sliding window (default boxcar windowing)
        win_func = {'hann':np.hanning, 'hamming':np.hamming, 'blackman':np.blackman}.get(self.win_type, np.ones)(self.rms_size)
        spectra[i] = DataLoader.RMS(win_func * audio[center-(self.rms_size//2):center+(self.rms_size//2)])

    
    if preproc_type not in ['AUTOCORR', 'RMS']:
      spectra = spectra[:,self.window_overlap//2:-self.window_overlap//2] # remove outer padded frames with unclear labels
    return ((100 + (5/4) * librosa.amplitude_to_db(spectra, ref=np.max)) if self.amplitude_to_db else spectra)
    
  # function to optionally clean the labels and smooth out edge cases
  def preprocess_labels(self, labels):
    processed_labels = np.array(labels, copy=True)
      
    # smooth the labels via a median filter to account for edge cases occuring usually between notes
    if self.smooth_label_width:
      for i in range(self.max_pitches):
        for j in range(self.smooth_label_width,len(labels[i]) - self.smooth_label_width):
          processed_labels[i, j] = np.median(labels[i,j-self.smooth_label_width:j+self.smooth_label_width+1])
    
    
    # shift midi notes as notes below E2 are not present in the dataset
    if self.shift_midi_notes:
      processed_labels = processed_labels - self.shift_midi_notes
      processed_labels = processed_labels * (processed_labels > 0)
    
    processed_labels  = np.swapaxes(processed_labels, 1, 0)

    # filter out duplicate notes (same note on multiple strings)
    for frame, labels in enumerate(processed_labels):
      uniques = set(labels)
      processed_labels[frame] = np.array(list(uniques) + (self.max_pitches-len(uniques))*[0])

    processed_labels.sort()
    return processed_labels[:,::-1]
  
  
  # function to save the sample data and labels in a serialized object for later processing
  def save_sample(self, filename, data):
    with open(self.path_save + filename + '.pkl', 'wb') as file:
      pickle.dump(data, file)
  
  
  # function to create augmented data from a sample
  def augment_sample(self, audio, labels, filename):
  
    if self.white_noise_std or self.salt_pepper_prb:
      augmented_audio = self.data_add_whitenoise(audio)
      augmented_audio = self.data_add_spnoise(augmented_audio)
      augmented_data = {'spectra' : self.preprocess_audio(augmented_audio), 'labels' : self.preprocess_labels(labels)}
      self.save_sample('AUG_NOISED_' + filename, augmented_data)
    
    if self.transpose_semitones_rng:
      if not hasattr(self, 'base_transposed'):
        transpose_steps = ((1 + np.random.randint(self.transpose_semitones_rng))             # select a random amount of semitones within range to transpose
                         * (-1 if self.transpose_down and np.random.random() < 0.5 else 1))  # transpose half the samples downward if selected
                         
      else:       # or match the amount to another preprocessed set (for merging results within ML network)
        transp_amt, transp_dir = self.base_transposed[filename]
        transpose_steps = (1 if transp_dir == 'A' else -1) * int(transp_amt)
      augmented_audio, augmented_labels = self.data_transpose(audio, labels, steps=transpose_steps)
      augmented_data = {'spectra' : self.preprocess_audio(augmented_audio), 'labels' : self.preprocess_labels(augmented_labels)}
      
      save_name = filename.split('_')
      save_name[1] = save_name[1] + str(np.abs(transpose_steps)) + ('A' if transpose_steps>0 else 'D')
      
      self.save_sample('AUG_TRAN_' + '_'.join(save_name), augmented_data)
    
    return
  
  # function to add white noise to the input signal
  def data_add_whitenoise(self, audio_sample, noise_std=None):
    if noise_std is None:
      noise_std = self.white_noise_std
    noise = np.random.normal(0, noise_std, audio_sample.shape[0])
    
    return audio_sample+noise
  
  # function to add salt-and-pepper noise to the input signal
  def data_add_spnoise(self, audio_sample, noise_prb=0.0002):
    if noise_prb is None:
      noise_prb = self.salt_pepper_prb
      
    noise_base = (np.random.random_sample(audio_sample.shape)* 2) - 1
    noised_audio = np.array([1 if noise_base[i] > (1-noise_prb) else 
                  (-1 if noise_base[i] <-(1-noise_prb) else audio_sample[i]) for i in range(audio_sample.size)])
    return noised_audio
  
  # function to transpose the input signal while keeping the original duration
  def data_transpose(self, audio_sample, labels, steps=0):
    if steps:
      audio_transposed = librosa.effects.pitch_shift(audio_sample, sr=self.sampling_rate, n_steps=steps)
      
      ringing_notes     = labels != 0
      labels_transposed = (labels + steps) * ringing_notes
      
      # rescale the transposed audio to the same loudness level via RMS normalization (if possible without clipping)
      RMS_ratio = self.RMS(audio_sample)/self.RMS(audio_transposed)
      if max(np.abs(audio_transposed)) * RMS_ratio < 1:
        audio_transposed = audio_transposed * RMS_ratio
      else:
        audio_transposed = audio_transposed * (1/max(np.abs(audio_transposed)))
      
    return audio_transposed, labels_transposed
  
  
  @staticmethod
  def RMS(data):
    return np.sqrt(np.mean(np.square(data)))
