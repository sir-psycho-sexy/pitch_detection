import sys, os, time, datetime, warnings
import librosa, pickle, scipy
import numpy as np

try:
  from DataGenerator import DataGenerator
  from AnalyticModel import AnalyticModel
  from Metrics import *
except:
  print('Custom classes not found in directory!')


# Analytic model that uses Non-negative Matrix Factorization
# with predefined spectral templates to estimate pitches
# Use with provided custom generator set to analytic_mode=True
# (each batch will consist of the spectra of one full song) 
class NMFModel:
  
  def __init__(self, spectrum_type='CQT',       # type of spectra the model should work with
               dataGenerator=None,              # optionally calibrate model to custom data generator
               use_predefined_basis=True,       # whether to use predefined spectral templates from recordings
               use_attack_frames=True,          # whether to use frames from the note's attack as basis vectors
               use_noise_frames=True,           # whether to use frames of only noise as basis vectors
               use_picked_frames=True,          # whether to use frames of notes plucked with a plectrum as basis vectors
               use_finger_frames=True,          # whether to use frames of notes plucked by finger as basis vectors
               use_ebow_frames=True,            # whether to use frames of notes played via an E-Bow as basis vectors
               iterations=30,                   # number of iterations for finding activation values
               window_size=2048,                # window size to use when constructing basis vectors
               window_type='blackmanharris',    # window type to use when constructing basis vectors
               normalize_basis=True,            # whether to normalize basis vectors framewise
               filter_basis='bessel',           # filter type to use on basis vectors
                ):
    

    self.path_dataset = ((os.path.dirname(os.path.abspath(__file__)) + os.path.sep) if '__file__' in globals() 
                      else 'gdrive/MyDrive/MSc Diploma/') + 'data' + os.path.sep # colab support for mounted Drive
    
                         
    self.nmf_models_dir = self.path_dataset + 'models' + os.path.sep + 'NMF' + os.path.sep 

              
    self.use_predefined_basis = use_predefined_basis
    self.use_noise_frames     = use_noise_frames
    self.use_attack_frames    = use_attack_frames
    self.use_picked_frames    = use_picked_frames
    self.use_finger_frames    = use_finger_frames
    self.use_ebow_frames      = use_ebow_frames
    self.iterations           = iterations
    

    self.normalize_basis = 'framewise' if normalize_basis else None
    self.filter_basis = filter_basis
    self.window_type = window_type
    
    if not os.path.exists(self.nmf_models_dir):
      os.makedirs(self.nmf_models_dir)

    # if custom data generator is available, calibrate model parameters to fit it
    if dataGenerator is not None:
      self.load_generator_specs(dataGenerator)
    else:
      self.spectrum_type   = spectrum_type
      self.sr              = 22050
      self.shift_labels    = 39
      self.bins_per_octave = 60
      self.octave_number   = 7
      self.fmin            = librosa.note_to_hz('C2')
      self.window_size     = window_size
      self.spectrum_size   = self.bins_per_octave * self.octave_number if self.spectrum_type in ['CQT', 'VQT'] else (self.window_size//2 + 1)

    # array containing bin frequencies, for visualization purposes
    self.freq_vector = (librosa.fft_frequencies(sr=self.sr, n_fft=self.window_size) if self.spectrum_type == 'FFT' else
                  librosa.cqt_frequencies(n_bins=self.spectrum_size, bins_per_octave=self.bins_per_octave, fmin=self.fmin))

    if use_predefined_basis:
      self.load_basis_vectors()
    else:
      self.rank = self.spectrum_size

          
    self.model_name = (f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")}_NMF_'
            + f'{self.spectrum_type}_{self.window_size}_{window_type}'
            + (f'{"_PREDEFINED" if use_predefined_basis else ""}' 
            + f'{"_ATTACK" if use_attack_frames else ""}' 
            + f'{"_PICKED" if use_picked_frames else ""}' 
            + f'{"_FINGER" if use_finger_frames else ""}' 
            + f'{"_EBOW" if use_ebow_frames else ""}' 
            + f'{"_NOISE" if use_noise_frames else ""}' 
            + f'{"_NORMALIZED" if normalize_basis else ""}' 
            + f'_{filter_basis if filter_basis is not None else ""}' ) if use_predefined_basis else (f'REAL_{iterations}'))
            
    if not os.path.exists(self.nmf_models_dir + self.model_name + os.path.sep):
      os.makedirs(self.nmf_models_dir + self.model_name + os.path.sep)


  # function to load and preprocess specified spectral templates
  def load_basis_vectors(self):
    print('\r' + 'Loading and preprocessing spectral template files!'.ljust(60), end='')
    basis_dir = self.path_dataset + 'NMF spectral templates' + os.path.sep

    basis_files = np.array(sorted(os.listdir(basis_dir)))
    noise_files = basis_files[np.char.startswith(basis_files, 'Noise')]
    basis_files = basis_files[np.logical_not(np.char.startswith(basis_files, 'Noise'))]

    pick_files = basis_files[np.char.endswith(basis_files, 'pick', end=-4)]
    finger_files = basis_files[np.char.endswith(basis_files, 'finger', end=-4)]
    ebow_files = basis_files[np.char.endswith(basis_files, 'ebow', end=-4)]
    

    self.rank = ((pick_files.size if self.use_picked_frames else 0) * (2 if self.use_attack_frames else 1)
                + (finger_files.size if self.use_finger_frames else 0) * (2 if self.use_attack_frames else 1)
                + (ebow_files.size if self.use_ebow_frames else 0)
                + (noise_files.size if self.use_noise_frames else 0))
    

    # load and preprocess all specified raw template files,
    # and save selected frames (from the attack+tail or from the middle of the sample)
    self.note_dict, curr_ind = {}, 0
    self.basis_vectors = np.empty((self.rank, self.spectrum_size))
    if self.use_picked_frames:
      for i, file in enumerate(pick_files):
        note = file.split()[0]
        self.note_dict[curr_ind] = note

        spectra = self.load_template(basis_dir + file)
        
        if self.use_attack_frames:
          self.basis_vectors[curr_ind] = spectra[:, 0]
          self.basis_vectors[curr_ind+1] = spectra[:, -1]
          self.note_dict[curr_ind+1] = note
          curr_ind += 2
        else:
          self.basis_vectors[curr_ind] = spectra[:, spectra.shape[1]//2]
          curr_ind += 1

        """
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False)
        for axis in ax:
          axis.set_xscale('log')
          axis.axvline(librosa.note_to_hz(note), color='green')

        ax[0].plot(freq_vector, np.abs(spectra[:, 1]))
        ax[1].plot(freq_vector, np.abs(spectra[:, spectra.shape[1]//2]))
        ax[2].plot(freq_vector, np.abs(spectra[:, -2]))

        fig.set_size_inches(8, 12, forward=True)
        fig.set_dpi(100)
        plt.show()
        """

    if self.use_finger_frames:
      for i, file in enumerate(finger_files):
        note = file.split()[0]
        self.note_dict[curr_ind] = note

        spectra = self.load_template(basis_dir + file)
        
        if self.use_attack_frames:
          self.basis_vectors[curr_ind] = spectra[:, 0]
          self.basis_vectors[curr_ind+1] = spectra[:, -1]
          self.note_dict[curr_ind+1] = note
          curr_ind += 2
        else:
          self.basis_vectors[curr_ind] = spectra[:, spectra.shape[1]//2]
          curr_ind += 1

    if self.use_ebow_frames:
      for i, file in enumerate(ebow_files):
        self.note_dict[curr_ind] = file.split()[0]

        spectra = self.load_template(basis_dir + file)
        
        self.basis_vectors[curr_ind] = spectra[:, spectra.shape[1]//2]
        curr_ind += 1

    if self.use_noise_frames:
      for i, file in enumerate(noise_files):
        self.note_dict[curr_ind] = 'Noise'

        spectra = self.load_template(basis_dir + file)
        
        self.basis_vectors[curr_ind] = spectra[:, spectra.shape[1]//2]
        curr_ind += 1
        

    self.basis_vectors = DataGenerator.preprocess_spectra(self.basis_vectors.T, spectra_type=self.spectrum_type, filter_type=self.filter_basis, pad_size=0,
                                     input_scale='amplitude', output_scale='amplitude', sr=self.sr, normalize=self.normalize_basis,
                                     cqt_bins_per_octave=self.bins_per_octave, cqt_fmin=self.fmin)
    self.basis_vectors = self.basis_vectors / np.amax(self.basis_vectors, axis=0)
    
    
    # create a dictionary mapping the basis vector indices to their corresponding notes
    self.note_indices = {}
    for k, v in self.note_dict.items():
      if v in self.note_indices.keys():
        self.note_indices[v].append(k)
      else:
        self.note_indices[v] = [k]
    
    
    print('\r' + f'{self.basis_vectors.shape[1]} spectral template files successfully loaded!'.ljust(60))


  # load and preprocess a spectral template audio file
  def load_template(self, file_path):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore") # ignore warnings as some samples are too short for large FFT windows
      
      audio, sr = librosa.load(file_path, sr=self.sr)
      if self.spectrum_type == 'CQT':
        spectra = np.abs(librosa.cqt(audio,
                        sr=self.sr, 
                        hop_length=self.window_size, 
                        n_bins=self.bins_per_octave * self.octave_number, 
                        bins_per_octave=self.bins_per_octave,
                        fmin=self.fmin,
                        window=self.window_type))
      
      if self.spectrum_type == 'VQT':
        spectra = np.abs(librosa.vqt(audio,
                        sr=self.sr, 
                        hop_length=self.window_size, 
                        n_bins=self.bins_per_octave * self.octave_number, 
                        bins_per_octave=self.bins_per_octave,
                        fmin=self.fmin,
                        window=self.window_type))
      
      if self.spectrum_type == 'FFT':
        spectra = np.abs(librosa.stft(audio,
                        hop_length=self.window_size, 
                        n_fft=self.window_size,
                        window=self.window_type))
    return spectra

  # function to get indices of the basis vectors belonging to a specific note
  def get_note_indices(self, note):
    indices = []
    for k, v in self.note_dict.items():
      if v == note:
        indices.append(k)
    return indices


  # function to save predictions of a model evaluation
  def save_predictions(self, gtruths, preds, spectra_path):
    with open(self.nmf_models_dir + self.model_name + os.path.sep + spectra_path + '.pkl', 'wb') as file:
      pickle.dump({'gtruths': gtruths, 'preds':preds}, file)

  
  # function to load relevant information from the given data loader/generator
  def load_generator_specs(self, dataGenerator):
    dataLoader = dataGenerator.get_data_loader()
    self.spectrum_type = dataLoader["preproc_type"]
    self.sr = dataLoader["sampling_rate"]
    self.window_size = dataLoader["window_size"]
    self.shift_labels = dataLoader["shift_midi_notes"]
    self.bins_per_octave = dataLoader["cqt_bins_per_oct"]
    self.octave_number = dataLoader["octave_number"]
    self.fmin = librosa.note_to_hz(dataLoader["lowest_note"])

    self.spectrum_size = self.bins_per_octave * self.octave_number if self.spectrum_type in ['CQT', 'VQT'] else (self.window_size//2 + 1)
    
    self.calibrated_to_generator = True
    return



  # function that evaluates the accuracy of the model on a data generator, and returns the predictions
  def evaluate(self, x, load_generator_specs=False, threshold=0.5, save_preds=True):
  
    if load_generator_specs:
      self.load_generator_specs(x)
      
    dataset_size = x.__len__()
    evaluation_start, full_acc = time.time(), 0
    preds, gtruths = [], []
    for i in range(dataset_size):
    
      
      print(('\r' + f'Evaluating NMF model: Sample {i+1}/{dataset_size}.'
            + f' Average processing time: ' + (f'{self.avg_proc_time:.2f}' if i else 'NaN')
            + f' Average {"f1" if threshold else "recall"}: ' + (f'{self.avg_acc:.2f}' if i else 'NaN')).ljust(120), end='')
      
      sample = x.__getitem__(i)
      pred, NMF_error = self.predict(sample[0], threshold=threshold)
      
      full_acc += accuracy_metrics(sample[1], pred)['mean']['f1' if threshold else 'recall']
      preds.append(pred)
      gtruths.append(sample[1])
        
      self.avg_proc_time, self.avg_acc = (time.time() - evaluation_start)/(i+1), full_acc/(i+1)

    print('\n' + f'Evaluation completed in {time.time() - evaluation_start:.2f} seconds!')
    
    if save_preds:
      self.save_predictions(gtruths, preds, x.get_spectra_path() + ('_REAL' if not threshold else ''))

    preds, gtruths = np.concatenate(preds), np.concatenate(gtruths)
    return accuracy_metrics(gtruths, preds), preds, gtruths
  
  

  # function to predict the notes for a given input
  def predict(self, spectra, threshold=0):
    
    spectra = spectra / (np.amax(spectra, axis=0) + np.finfo(float).eps)
    
    if self.use_predefined_basis:
      basis_vectors = self.basis_vectors
      #activations, _, _, _ = np.linalg.lstsq(basis_vectors, spectra, rcond=None)
      
      activations = np.empty((self.rank, spectra.shape[1]))
      for i, frame in enumerate(spectra.T):
        activations[:, i] = scipy.optimize.nnls(basis_vectors, frame)[0]
      
    else:
      basis_vectors, activations, _ = self.NMF(spectra)
      NMF_error = np.linalg.norm(spectra - basis_vectors @ activations, 'fro')
      return activations, NMF_error   # TODO full NMF branch
    
    
    # add up activations belonging to the same note
    scores = np.zeros((len(self.note_indices), spectra.shape[1]))
    
    for i, (note, indices) in enumerate(self.note_indices.items()):
      scores[i] += np.sum(activations[indices], axis=0)
    
    # get predictions via thresholded top-K selection
    predictions = np.zeros((spectra.shape[1], 6), dtype=int)
    
    max_scores = np.argpartition(scores.T, range(0,-6, -1), axis=1)[:,-6:]
    for frame, ind in enumerate(max_scores):
      frame_pred = max_scores[frame][scores.T[frame,ind] > threshold]
      predictions[frame,:frame_pred.size] = frame_pred[::-1]


    # transform index predictions into corresponding midi labels
    for ind, row in enumerate(predictions):
      predictions[ind] = librosa.note_to_midi([self.note_dict[n] if (n and self.note_dict[n] != 'Noise')
                         else librosa.midi_to_note(self.shift_labels) for n in row]) - self.shift_labels

    return predictions, np.linalg.norm(spectra - basis_vectors @ activations, 'fro')


  # perform non-negative matrix factorization on given spectra
  # uses multiplicative update rule if 
  def NMF(self, spectra, iterations=None):
    if iterations is None:
      iterations = self.iterations

    H = np.random.uniform(1,2,(self.rank, spectra.shape[1]))
    W = np.random.uniform(1,2,(spectra.shape[0],self.rank))
    
    errors = []
    # iterate using multiplicative update rule to converge W @ H -> spectra
		# using element-wise calculations instead of matrix multiplications
    for _ in range(iterations):
      # Update activation vectors
      nominator = W.T @ spectra
      denominator = W.T @ W @ H + 1e-5
      for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i, j] = H[i, j] * nominator[i, j] / denominator[i, j]



      # Update basis vectors
      nominator = spectra @ H.T
      denominator =  W @ H @ H.T + 1e-5

      for i in range(W.shape[0]):
        for j in range(W.shape[1]):
          W[i, j] = W[i, j] * nominator[i, j] / denominator[i, j]

      # we use Frobenius norm/squared error as the distance metric
      errors.append(np.linalg.norm(spectra - W @ H, 'fro'))

    return (W, H, errors)

