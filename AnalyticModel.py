import sys, os, time, datetime
import numpy as np
import librosa, pickle, scipy.signal

try:
  from Metrics import AnalyticMetrics
except:
  print('Custom Metrics not found in directory!')


# Analytic model that uses peak detection within spectra 
# to determine notes being played
# Use with provided custom generator set to analytic_mode=True
# (each batch will consist of the spectra of one full song)  
class AnalyticModel:

  def __init__(self, model_path=None,   # path of an existing model to optionally load parameters
               model_type='FFT',        # type of spectra the model should tune
               hyp_param_search='sequential', # type of hyperparameter tuning (grid, random, sequential, threshold)
               hyp_param_runs=10,       # number of parameter combinations to try during tuning (random search)
               hyp_param_grid=10,       # number of grid points to check when searching tuning (grid/sequential search)
               hyp_param_size=None,     # ratio of the dataset to use when tuning parameters (TODO: None -> successive halving)
               peak_min_note='E2',      # lowest note to consider when selecting peaks
               peak_max_note='C6',      # highest note to consider when selecting peaks
               ot_filter_type='quad',   # filter used for weighing overtone(/undertone) scores, can either be:
                                        #   list of max_overtones floats containing the weights
                                        #   lambda function mapping ot_number -> ot_weight for each overtone (0 = the fundamental note)
                                        #   float in [0,1] by which overtones are exponentially weighted
                                        #   string that uses one of the predefined filters (includes 'exp', 'quad', 'lin')
               peak_prom_ratio=0.9,     # prominence to height ratio used as a threshold for peak selection
               neighbor_weight=0.75,    # ratio by which to exponentially weigh past/future frame amplitudes for peak selection
               ovrl_threshold=3.0,      # threshold for discarding spectrum peaks after scoring them
               max_neighbors=10,        # maximum number of neighboring frames to check on each side
               max_overtones=5,         # maximum number of overtones to weigh when scoring notes (including self)
               early_stop=0.3,          # stop current parameter eval if avg loss is not close to current best after this ratio of samples
               LossMetrics=AnalyticMetrics(),   # loss metrics to which parameters should be tuned
               ):

    self.models_dir = (((os.path.dirname(os.path.abspath(__file__)) + os.path.sep) if '__file__' in globals() 
                      else 'gdrive/MyDrive/MSc Diploma/') + 'data' + os.path.sep + 'models' + os.path.sep) # colab support for mounted Drive
                         
    self.model_type = model_type
    
    self.hyp_param_search = hyp_param_search
    self.hyp_param_runs = hyp_param_runs
    self.hyp_param_grid = hyp_param_grid
    self.hyp_param_size = hyp_param_size
    self.peak_min_note = peak_min_note
    self.peak_max_note = peak_max_note
    
    self.min_midi, self.max_midi = librosa.note_to_midi([peak_min_note, peak_max_note])

    """# limiting overtones/undertone numbers to spectra resolution
    if self.model_type not in ['CEPS', 'PCEPS', 'AUTOCORR']:
      self.max_overtones = min(int(np.floor((self.sr/2)/librosa.note_to_hz(peak_max_note))), max_overtones)
    else:
      self.max_overtones = min(int(np.floor(librosa.note_to_hz(peak_min_note)/(self.sr/2048))), max_overtones)"""

    self.max_overtones = max_overtones
    self.max_neighbors = max_neighbors
    self.early_stop = early_stop
    
    
    self.ot_filter_type = ot_filter_type
    self.overtone_filter = self.create_ot_filter(ot_filter_type)
    
    
    self.params, self.param_ranges = {}, {}
    for overtone in range(len(self.overtone_filter)):
      self.params[f'1-overtone_weight_{overtone}'] = self.overtone_filter[overtone]
      self.param_ranges[f'1-overtone_weight_{overtone}'] = (0, 1)
    self.params["2-peak_prom_ratio"] = peak_prom_ratio
    self.params["3-neighbor_weight"] = neighbor_weight
    self.params["4-ovrl_threshold"] = ovrl_threshold

    self.param_ranges["2-peak_prom_ratio"] = (0.1, 0.99)
    self.param_ranges["3-neighbor_weight"] = (0, 1)
    self.param_ranges["4-ovrl_threshold"] = (0, 15)
    
    
    
    self.LossMetrics = LossMetrics
    self.sr = 22050
    self.shift_labels = 39
    self.scale_spectra = True
    self.bins_per_octave = 36
    self.fmin = librosa.note_to_hz('C2')

    if not os.path.exists(self.models_dir):
      os.makedirs(self.models_dir)

    # load an existing tuned model if specified
    if model_path is not None:
      if os.path.exists(self.models_dir + model_path):
        print('Loading existing model!')
        self.load_model(self.models_dir + model_path)
      else:
        print('Unable to load model. File does not exist!')



  # function to load the parameters of the model from an existing file
  def load_model(self, model_path):
    if os.path.exists(model_path):
      with open(model_path + os.path.sep + 'Model.pkl', 'rb') as file:
        model_dict = pickle.load(file)
      
      for k,v in model_dict.items():
        setattr(self, k, v)
      
      print(f'Loaded model with parameters: {self.params}')
    else:
      print('Unable to load model. File does not exist!')
    return

  # function to manually save the set parameters of the model
  def save_model(self):
    if not hasattr(self, 'model_name'):
      self.model_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + self.model_type + '_MANUAL_UNFITTED/'

    if not os.path.exists(self.models_dir + self.model_name):
      os.makedirs(self.models_dir + self.model_name)
    with open(self.models_dir + self.model_name + 'Model.pkl', 'wb') as file:
      pickle.dump(self.__dict__, file)

  # function to save predictions of a model evaluation
  def save_predictions(self, gtruths, preds, spectra_path):
    with open(self.models_dir + self.model_name + spectra_path + '.pkl', 'wb') as file:
      pickle.dump({'gtruths': gtruths, 'preds':preds}, file)

  # function to load relevant information from the given data loader/generator
  def load_generator_specs(self, dataGenerator):
    dataLoader = dataGenerator.get_data_loader()
    self.model_type = dataLoader["preproc_type"]
    self.sr = dataLoader["sampling_rate"]
    self.shift_labels = dataLoader["shift_midi_notes"]
    self.bins_per_octave = dataLoader["cqt_bins_per_oct"]
    self.fmin = librosa.note_to_hz(dataLoader["lowest_note"])

    self.scale_spectra = False if dataGenerator.normalize_spectra is None else True 
    self.calibrated_to_generator = True
    return


  # function to tune the parameters of the model through a (x=)generator,
  # maximizing the given loss function
  # additional parameters to fit with the prebuilt Keras fit function
  def fit(self, x, load_generator_specs=True,
          y=None, validation_data=None, shuffle=None, epochs=None,
          batch_size=None, callbacks=None, use_multiprocessing=None, verbose=None):
    
    if load_generator_specs:
      self.load_generator_specs(x) 



    parameter_grid = np.zeros((len(self.params), self.hyp_param_grid))
    permutation_grid = np.zeros((len(self.params), self.hyp_param_grid))
    for i, param in enumerate(sorted(self.param_ranges.keys())):
      parameter_grid[i] = np.linspace(*self.param_ranges[param], self.hyp_param_grid)

    

    if self.hyp_param_search == 'grid':
      indexes = np.array([np.arange(self.hyp_param_grid) for i in range(len(self.params))])
      index_grid = np.array(np.meshgrid(*indexes)).T.reshape(-1, len(self.params))
      permutations = index_grid
      
    if self.hyp_param_search == 'sequential':
      # fill all permutations with the center of the grid ranges
      range_centers = parameter_grid[:,parameter_grid.shape[1]//2]
      range_centers[-1] = 0.0             # set overall threshold to 0.0 until last stage
      permutations = np.tile(range_centers, (len(self.params)*self.hyp_param_grid, 1))
      self.LossMetrics.metric = 'recall'  # maximize recall first, then set to f1 when tuning overall threshold

      # then replace with the respective grid for each parameter sequentially
      for i, param in enumerate(sorted(self.param_ranges.keys())):
        permutations[self.hyp_param_grid*i:self.hyp_param_grid*(i+1), i] = parameter_grid[i]
        
    if self.hyp_param_search == 'threshold':    # only fit final thresholding parameter
      permutations = np.empty((self.hyp_param_grid, (len(self.params))))
      for i, param in enumerate(sorted(self.params.keys())):
        permutations[:, i] = self.params[param]
      permutations[:,-1] = np.linspace(*self.param_ranges["4-ovrl_threshold"], self.hyp_param_grid)
      
    if self.hyp_param_search == 'random':
      indexes = np.array([np.arange(self.hyp_param_grid) for i in range(len(self.params))])
      index_grid = np.array(np.meshgrid(*indexes)).T.reshape(-1, len(self.params))
      permutations = index_grid[np.random.choice(index_grid.shape[0], self.hyp_param_runs, replace=False)]
    
    
    log_template = (f'{self.model_type} Model tuning: Permutation PERM_NR - Sample SAMPLE_NR - Frame FRAME_NR '
                   + '- Avg sample processing time PROC_AVG_TIME - Avg sample loss AVG_LOSS. Current best avg loss: MIN_LOSS')
    

    if not hasattr(self, 'model_name'):
      self.model_name = (f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")}_'
              + f'{self.model_type}_{self.hyp_param_search}{len(permutations)}_{self.hyp_param_grid}' 
              + f'{("_" + str(self.hyp_param_size)) if self.hyp_param_size is not None else ""}/')
    

    self.processing_start = time.time()
    
    self.min_loss, self.min_params = np.finfo(float).max, self.params
    try:
      for try_nr, param_permutation in enumerate(permutations):

        if self.hyp_param_search == 'sequential' and try_nr == len(permutations) - self.hyp_param_grid: 
          self.LossMetrics.metric = 'f1' # switch loss metric to f1 for last parameter (overall threshold)
          self.min_loss = np.finfo(float).max
          
        # set parameters to the current permutation
        for i, param in enumerate(sorted(self.param_ranges.keys())):
          self.params[param] = (parameter_grid[i, param_permutation[i]]
                                if self.hyp_param_search not in ['sequential', 'threshold'] else param_permutation[i])
                                
          for overtone in range(len(self.overtone_filter)):
            self.overtone_filter[overtone] = self.params[f'1-overtone_weight_{overtone}']
            
        
        self.log_state = log_template.replace('PERM_NR', f'{try_nr+1}/{len(permutations)}')
        self.log_state = self.log_state.replace('MIN_LOSS', (f'{self.min_loss:.2f}' if self.min_loss < np.finfo(float).max else "NaN"))
        
        loss = self.evaluate(x, early_stop=self.early_stop) # evaluate accuracy loss for current parameter setup

        if loss < self.min_loss:
          self.min_loss, self.min_params = loss, dict(self.params)

          if self.hyp_param_search == 'sequential': # update the remaining permutations to have the current value
            curr_param = try_nr // self.hyp_param_grid   # of the parameter being minimized (=i//grid)
            permutations[self.hyp_param_grid*(curr_param+1):, curr_param] = param_permutation[curr_param]
            
            
      self.processing_time = time.time() - self.processing_start
      print(f'Model tuning: COMPLETED in {self.processing_time:.2f} seconds.')
      print(f'    Best loss: {self.min_loss:.2f} for parameters {self.min_params}')
    except KeyboardInterrupt:
      print(f'\nFitting interrupted. Best parameter setup so far: {self.min_params}')
    print(f'    Best model saved to: {self.models_dir + self.model_name}')
      
    
    self.params = dict(self.min_params)                 # set parameters and filters according
    for overtone in range(len(self.overtone_filter)):   # to the best permutation found
      self.overtone_filter[overtone] = self.params[f'1-overtone_weight_{overtone}']
    
    # save the data loader specifications for reference, including the generator it was fitted to
    self.fitted_generator = x.get_generator_setup()
    self.save_model()
    return
    
  
  
  # function that evaluates the accuracy loss of the model
  # with the current parameter setup for a (x=)generator
  def evaluate(self, x, load_generator_specs=False, early_stop=0, return_preds=False, skip_thresholding=False, save_preds=False):
    
    if not hasattr(self, 'log_state') or return_preds:
      self.log_state = f'Evaluating: Sample SAMPLE_NR - Frame FRAME_NR - Avg sample processing time PROC_AVG_TIME - Avg sample loss AVG_LOSS'
    
    evaluation_start = time.time()
    
    if load_generator_specs:
      self.load_generator_specs(x) 
      
    log_template = self.log_state
    
    if return_preds:
      preds, gtruths = [], []
      
    if skip_thresholding:
      threshold = self.params["4-ovrl_threshold"]
      self.params["4-ovrl_threshold"] = 0.0
    
    full_accuracy_loss, self.dataset_size = 0, x.__len__()
    
    if return_preds or self.hyp_param_size is None:
      self.hyp_param_size = 1
    
    # evaluate the model on the ratio of total samples requested
    sample_subset = np.random.choice(np.arange(self.dataset_size),
              int(np.ceil(self.dataset_size*self.hyp_param_size)), replace=False) if not return_preds else np.arange(self.dataset_size)
    for i, sample in enumerate(sample_subset):
    
      self.log_state = log_template.replace('SAMPLE_NR', f'{i+1}/{int(np.ceil(self.dataset_size*self.hyp_param_size))}')
      self.log_state = self.log_state.replace('PROC_AVG_TIME', (f'{self.avg_proc_time:.2f}') if i else 'NaN')
      self.log_state = self.log_state.replace('AVG_LOSS', (f'{self.avg_loss:.2f}') if i else 'NaN')
      
      if not return_preds:
        full_accuracy_loss += self.evaluate_sample(x.__getitem__(sample))
      else:
        curr_loss, pred, gtruth = self.evaluate_sample(x.__getitem__(sample), return_preds=True)
        full_accuracy_loss += curr_loss
        preds.append(pred)
        gtruths.append(gtruth)
        
      self.avg_proc_time, self.avg_loss = (time.time() - evaluation_start)/(i+1), full_accuracy_loss/(i+1)
      
      if early_stop and early_stop <= (i/self.dataset_size) and self.avg_loss * 0.9 > self.min_loss:
        break   # stop evaluating if avg loss does not approach current best
    
    
    if return_preds and save_preds:
      if not hasattr(self, 'model_name'):
        self.params["4-ovrl_threshold"] = threshold
        self.save_model()
      self.save_predictions(gtruths, preds, x.get_spectra_path() + ('_REAL' if not skip_thresholding else ''))
    
    return (self.avg_loss) if not return_preds else (self.avg_loss, np.concatenate(preds), np.concatenate(gtruths))
  

  # function that evaluates the accuracy loss of the model
  # with the current parameter setup for a single sample
  def evaluate_sample(self, sample, return_preds=False):
    pred = self.predict(sample[0])
    accuracy_loss = self.LossMetrics.accuracy_loss(sample[1], pred)
    return accuracy_loss if not return_preds else (accuracy_loss, pred, sample[1])


  # function to create a filter used to weigh overtones
  def create_ot_filter(self, ot_filter_type, max_overtones=None):
    if max_overtones is None:
      max_overtones = self.max_overtones
    
    if isinstance(ot_filter_type, float):
      ot_filter = ot_filter_type ** np.arange(max_overtones)
      
    if isinstance(ot_filter_type, list):
      ot_filter = ot_filter_type
  
    if isinstance(ot_filter_type, type(lambda:0)):
      ot_filter = ot_filter_type(np.arange(max_overtones))
      
    if  isinstance(ot_filter_type, str):
      lambda_fn = {'lin': lambda ot : -(ot - max_overtones+1),
                   'quad':lambda ot : -(ot + 1.0) * (ot - 4.0),
                   'exp': lambda ot : 0.75 ** ot}[ot_filter_type]
      
      ot_filter = lambda_fn(np.arange(max_overtones))
    
    # flip the filter as np.convoke reverses it, clip and normalize to ensure general scale
    np.flip(np.clip(np.true_divide(ot_filter, np.max(ot_filter), out=ot_filter, where=(ot_filter != 0)), 0, 1))
      
    return ot_filter
    
  
  # function to calculate the range of bins a midi note belongs to
  def bin_range(self, midi_note, bin_midis=None):
    if bin_midis is None:
      bin_midis = self.bin_midis
      
    if self.model_type not in ['CEPS', 'PCEPS', 'AUTOCORR']:
      first, last = np.where(bin_midis<midi_note)[0][-1]+1, np.where(bin_midis>midi_note)[0][0]-1
    else: # frequencies belonging to cepstra and autocorrelation are ordered in reverse 
      first = (np.where(bin_midis>midi_note)[0][-1]+1) if midi_note < bin_midis[ 0] else 0
      last  = (np.where(bin_midis<midi_note)[0][ 0]-1) if midi_note > bin_midis[-1] else len(bin_midis)
    
    # if no bins belong to the given note, return the two closest bins
    return (first, last) if first <= last else (last, first)

  
  # function to generate the overtone (or undertone if ut_flag=True) series for a note
  @staticmethod
  def generate_overtone_series(note, overtone_nr=5, mode='midi', ut_flag=True):
    if mode=='midi':
      overtones = np.rint([librosa.hz_to_midi(librosa.midi_to_hz(note)*(1/i if ut_flag else i)) for i in range(1,overtone_nr+2)])
    if mode=='note':
      overtones = np.array([librosa.hz_to_note(librosa.note_to_hz(note)*(1/i if ut_flag else i)) for i in range(1,overtone_nr+2)])
    return overtones

  # function to predict the notes for a given input
  # additional parameters to fit with the prebuilt Keras predict function
  def predict(self, x, return_scores=False, skip_thresholding=False, sr=None,
              batch_size=None, callbacks=None, use_multiprocessing=None, verbose=None):
    
    input = np.array(x, copy=True)

    if self.model_type in ['FFT', 'CEPS', 'PCEPS']:
      input = input[1:] # discard DC values for FFT spectra

    if not hasattr(self, 'log_state'):
      self.log_state = f'Predicting sample: Frame FRAME_NR'
    log_template = self.log_state

    if sr is None:
      sr = self.sr
    
    if not hasattr(self,'bin_midis') or len(self.bin_midis) != input.shape[0]:
      if self.model_type == 'FFT':
        bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=input.shape[0]*2)[1:]
      if self.model_type in ['CEPS', 'PCEPS']: # create frequency bins corresponding to the cepstral quefrency bins
        bin_freqs = 1.0/librosa.fft_frequencies(sr=(input.shape[0]*4)/sr, n_fft=input.shape[0]*2)[1:]
        input[bin_freqs > 1000] = 0            # remove higher frequency bins from cepstra (most overtones contributed via FFT)
      if self.model_type in ['CQT', 'VQT']:
        bin_freqs = librosa.cqt_frequencies(n_bins=input.shape[0], bins_per_octave=self.bins_per_octave, fmin=self.fmin)
      if self.model_type == 'AUTOCORR':
        bin_freqs = np.ones(input.shape[0])*sr
        np.true_divide(bin_freqs, np.arange(input.shape[0]), where=(np.arange(input.shape[0]) != 0), out=bin_freqs)

      self.bin_midis = np.rint(librosa.hz_to_midi(bin_freqs))
      self.min_bin = np.where(self.bin_midis < self.min_midi)[0][-1]
      self.max_bin = np.where(self.bin_midis > self.max_midi)[0][0]



    pred = np.zeros((input.shape[1], 6), dtype=int)  # array for note predictions of every frame
    scores = np.empty((input.shape[1], self.max_midi - self.min_midi))  # array for scores of each note for every frame
    final_scores = np.empty(scores.shape)  # array for final scores of each note for every frame

    # array to hold prominence thresholds for peak filtering
    prominence_trsh = np.zeros(input.shape[0])

    # create filter for weighing overtones
    if not hasattr(self, 'overtone_filter'):
      self.overtone_filter = self.create_ot_filter(self.ot_filter_type)

    # filter for weighing note strengths in neighboring frames
    neighbor_filter = self.params["3-neighbor_weight"] ** np.arange(self.max_neighbors)
    neighbor_filter = neighbor_filter[neighbor_filter > 0.1]
    neighbor_filter = np.concatenate((np.flip(neighbor_filter[1:]), neighbor_filter))
    
    
    
    # iterate over the frames of the sample
    for frame, spectrum in enumerate(input.T):
      
      if self.scale_spectra:
        spectrum = spectrum / (np.max(spectrum) + np.finfo(float).eps)

      print(('\r' + log_template.replace('FRAME_NR', f'{frame+1}/{input.T.shape[0]}')).ljust(175), end='')


      prominence_trsh[self.min_bin:self.max_bin] = spectrum[self.min_bin:self.max_bin] * self.params["2-peak_prom_ratio"]
      peaks, peak_infos = scipy.signal.find_peaks(spectrum, prominence=prominence_trsh, threshold=None, distance=1, height=0.01)
      
      # score the possibility of every note being played according to the current spectrum
      for note in range(self.min_midi, self.max_midi):
        # calculate score from individual overtones (including self)
        overtone_scores = np.zeros(self.overtone_filter.size)
        overtone_series = self.generate_overtone_series(note, self.overtone_filter.size-1,ut_flag=(self.model_type in ['CEPS', 'PCEPS', 'AUTOCORR']))
        
        for i, overtone in enumerate(overtone_series):
          ot_first_bin, ot_last_bin = self.bin_range(overtone)

          # calculate overtone strength from amplitude over its frequency bins and possible peak prominance
          ot_prominence = peak_infos['prominences'][np.where(np.logical_and(peaks >= ot_first_bin, peaks <= ot_last_bin))]
          
          overtone_scores[i] = (np.average(spectrum[ot_first_bin:ot_last_bin+1]) + 
                              (np.average(ot_prominence) if ot_prominence.size else 0))
          
          
        # weigh the overtone scores into a single value for the current note
        scores[frame,note-self.min_midi] = np.convolve(overtone_scores, self.overtone_filter, mode='valid')[0]



    # iterate over notes and give a final score for all frames while weighing in neighboring frames as well
    scores = np.pad(scores, ((neighbor_filter.size//2, neighbor_filter.size//2), (0, 0)), mode='edge') # pad scores on both ends
    for note in range(self.max_midi - self.min_midi):
        final_scores[:, note] =  np.convolve(scores[:,note], neighbor_filter, mode='valid')
          
    # make predictions from the best note scores and (optionally) threshold them
    max_scores = np.argpartition(final_scores, range(0,-6, -1), axis=1)[:,-6:]
    
    for frame, ind in enumerate(max_scores):
      frame_pred = max_scores[frame][final_scores[frame,ind] > (self.params["4-ovrl_threshold"] if not skip_thresholding else 0.0)]
      pred[frame,:frame_pred.size] = frame_pred[::-1] + self.min_midi

    # shift midi labels to account for lowest note according to the data loader
    if self.shift_labels is not None and self.shift_labels:
      pred = pred - self.shift_labels
      pred = pred * (pred > 0)
    
    return (pred, final_scores) if return_scores else pred

