import matplotlib.pyplot as plt
import librosa.display
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 8})

plot_cqt_note_ticks = [6.54063913e+01, 1.30812783e+02, 2.61625565e+02, 5.23251131e+02, 1.04650226e+03, 2.09300452e+03, 4.18600904e+03,]

# construct a CQT spectra shaped array with values only on the labels' note bins
# for plotting/visualization purposes
def labels_to_spectra(labels, truth_amplitude=100, n_bins=252, bins_per_octave=36, fmin='C2', label_shift=39):
  freqs = [(librosa.hz_to_midi(a)) for 
           a in librosa.cqt_frequencies(n_bins=n_bins,bins_per_octave=bins_per_octave,fmin=librosa.note_to_hz(fmin))]
  freqs = np.rint(np.array(freqs)).astype('int')
  spectra = np.zeros((freqs.shape[0], labels.shape[0]))
  
  for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
      if labels[i,j] != 0:
        spectra[np.where(freqs==labels[i,j]+label_shift)[0], i] = truth_amplitude
  
  return spectra


# get the predictions that are octave errors (for visualization purposes)
def all_octave_errors(gtruths, preds):
  octave_errors = np.zeros(preds.shape)
  for i, pred in enumerate(preds):
    gt, p = gtruths[i][gtruths[i] != 0], pred[pred != 0]
    oct_err = p[np.isin(p%12, np.intersect1d(gt%12, p%12))]

    octave_errors[i,:len(oct_err)] = oct_err
  return octave_errors


# get the predictions that are semiton errors (for visualization purposes)
def semitone_errors(gtruths, preds):
  semitone_errors = np.zeros(preds.shape)
  for i, pred in enumerate(preds):
    gt, p = gtruths[i][gtruths[i] != 0], pred[pred != 0]
    semitone_err = p * np.logical_or(np.isin(p-1, gt), np.isin(p+1, gt))
    semitone_errors[i,:len(semitone_err)] = semitone_err
  return semitone_errors


cmap = 'nipy_spectral'
# create an image which visualizes the specified labels on a CQT shaped spectrum
def visualize_labels(labels, axis):
  return librosa.display.specshow(labels_to_spectra(labels), sr=22050, x_axis='time', y_axis='cqt_note', 
                                  ax=axis, bins_per_octave=36, fmin=librosa.note_to_hz('C2'))

# create an image on the specified plot axis which visualizes correct guesses, missed labels and incorrect guesses
def visualize_predictions(gtruths, preds, axis, mark_oct_errors=True, mark_semi_errors=True):
  gtruth_spectra, pred_spectra = labels_to_spectra(gtruths), labels_to_spectra(preds)
  unguessed = np.maximum(gtruth_spectra - pred_spectra, 0)
  incorrect = np.maximum(pred_spectra - gtruth_spectra, 0)
  img = np.zeros(pred_spectra.shape)
  img[pred_spectra != 0] = 0.5
  img[incorrect != 0] = 0.995
  img[unguessed != 0] = 0.1

  if mark_semi_errors:
    semi_errors = np.maximum(labels_to_spectra(semitone_errors(gtruths, preds)) - gtruth_spectra, 0)
    img[semi_errors != 0] = 0.8
  if mark_oct_errors:
    oct_errors = np.maximum(labels_to_spectra(all_octave_errors(gtruths, preds)) - gtruth_spectra, 0)
    img[oct_errors != 0] = 0.3
  img[0,0] = 1 # set corner pixel to max value for correct color range

  img = librosa.display.specshow(img, cmap=cmap, sr=22050, x_axis='time', y_axis='cqt_note', ax=axis, bins_per_octave=36, fmin=librosa.note_to_hz('C2'))
  return img