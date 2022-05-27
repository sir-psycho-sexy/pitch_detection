# Metrics used to evaluate the models
import numpy as np

log_min = 1e-5    # lower clipping value to avoid log(0)

# Metrics for test result evaluation and analytic model tuning (non-binarized labels)

# for a single spectrum frame
def frame_precision(y_true, y_pred):
  corr_pred = np.count_nonzero(np.intersect1d(y_pred, y_true))
  all_pred  = np.count_nonzero(y_pred)
  return (corr_pred/all_pred) if all_pred else 1.0
  
def frame_recall(y_true, y_pred):
  corr_pred = np.count_nonzero(np.intersect1d(y_pred, y_true))
  all_true  = np.count_nonzero(y_true)
  return (corr_pred/all_true) if all_true else 1.0

def frame_f1_score(y_true, y_pred):
  corr_pred = np.count_nonzero(np.intersect1d(y_pred, y_true))
  all_pred  = np.count_nonzero(y_pred)
  all_true  = np.count_nonzero(y_true)
  prec = ((corr_pred/all_pred) if all_pred else 1.0)
  rec = ((corr_pred/all_true) if all_true else 1.0)
  return 2 * (prec * rec) / (prec + rec) if (prec and rec) else log_min

def frame_precision_recall_f1(y_true, y_pred):
  corr_pred = np.count_nonzero(np.intersect1d(y_pred, y_true))
  all_pred  = np.count_nonzero(y_pred)
  all_true  = np.count_nonzero(y_true)
  prec = ((corr_pred/all_pred) if all_pred else 1.0)
  rec = ((corr_pred/all_true) if all_true else 1.0)
  f1 = 2 * (prec * rec) / (prec + rec) if (prec and rec) else log_min
  return (prec, rec, f1)


# return the number of notes that are correct or within certain amount of semitones
def count_semi_correct_notes(y_true, y_pred, semitone_rng=1):
  semi_corr_pred, true_rem = 0, y_true[y_true != 0]
  for note in y_pred:
    if np.any(np.abs(true_rem - note)<=semitone_rng) and note:
      true_rem = true_rem[np.abs(true_rem - note)>semitone_rng]
      semi_corr_pred += 1
  
  return semi_corr_pred



# for an entire sample
def accuracy_metrics(y_true, y_pred, semitones_rng=0):
  metrics = {}
  corr_pred, all_pred, all_true, all_metrics = 0, 0, 0, np.zeros((y_true.shape[0], 3))
  for frame in range(y_true.shape[0]):
    if semitones_rng:
      curr_corr_pred = count_semi_correct_notes(y_true[frame], y_pred[frame])
    else:
      curr_corr_pred = np.count_nonzero(np.intersect1d(y_true[frame], y_pred[frame]))
    curr_all_pred  = np.count_nonzero(y_pred[frame])
    curr_all_true  = np.count_nonzero(y_true[frame])

    corr_pred += curr_corr_pred
    all_pred  += curr_all_pred
    all_true  += curr_all_true

    curr_prec = ((curr_corr_pred/curr_all_pred) if curr_all_pred else 1.0)
    curr_rec = ((curr_corr_pred/curr_all_true) if curr_all_true else 1.0)
    curr_f1 = 2 * (curr_prec * curr_rec) / (curr_prec + curr_rec) if (curr_prec and curr_rec) else log_min
    #print(corr_pred, all_pred, all_true, curr_corr_pred, curr_all_pred, curr_all_true, curr_prec, curr_rec, curr_f1)
    
    all_metrics[frame] = [curr_prec, curr_rec, curr_f1]

  mean_metrics = np.mean(all_metrics, axis=0)
  overall_prec = ((corr_pred/all_pred) if all_pred else 1.0)
  overall_rec = ((corr_pred/all_true) if all_true else 1.0)
  if overall_prec and overall_rec:
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec)
  else:
    overall_f1 = log_min
    
  metrics['overall'] = {'precision': overall_prec,
                        'recall': overall_rec,
                        'f1': overall_f1}
  metrics['mean'] = {'precision': mean_metrics[0],
                      'recall': mean_metrics[1],
                      'f1': mean_metrics[2]}
  return metrics
  

# return accuracy metrics while ignoring note octaves
def accuracy_chroma(y_true, y_pred):
  pred_ringing, true_ringing = np.nonzero(y_pred), np.nonzero(y_true)
  true, pred = np.zeros(y_true.shape), np.zeros(y_pred.shape)

  # transform midi labels into chroma groups for the 12 semitones (+1 for 0, which is no note)
  true[true_ringing] = (y_true%12)[true_ringing] + 1
  pred[pred_ringing] = (y_pred%12)[pred_ringing] + 1

  pred_chroma, true_chroma = np.zeros((pred.shape[0], 6), dtype=int), np.zeros((true.shape[0], 6), dtype=int)
  for i in range(pred.shape[0]):
    pred_unique, true_unique = np.unique(pred[i]), np.unique(true[i])

    pred_chroma[i,:len(pred_unique)] = pred_unique
    true_chroma[i,:len(true_unique)] = true_unique
    
  return accuracy_metrics(true_chroma, pred_chroma)



# Passable loss metric for analytic functions
class AnalyticMetrics:
  def __init__(self, metric='f1', # metric which should be used to calculate loss
               chroma_weight=0.5, # weight of the metric when ignoring octave errors in the loss
               semitone_weight=0.10, # weight of the metric when ignoring semitone errors in the loss 
               overall_mean_ratio=0.5, # ratio of metrics that were averaged over the entire sample set
                                       # vs averaging the metrics for each sample
               verbose=False # additional logging for each sample
               ):
    self.metric=metric
    self.chroma_weight=chroma_weight
    self.semitone_weight=semitone_weight
    self.overall_mean_ratio=overall_mean_ratio
    self.verbose=verbose

  # custom loss function calculated from accuracy metrics,
  # lessened by octave and semitone errors, used for tuning analytic models
  def accuracy_loss(self, y_true, y_pred, chroma_weight=None, semitone_weight=None, overall_mean_ratio=None, verbose=None):
    if chroma_weight is None:
      chroma_weight=self.chroma_weight
    if semitone_weight is None:
      semitone_weight=self.semitone_weight
    if overall_mean_ratio is None:
      overall_mean_ratio=self.overall_mean_ratio
    if verbose is None:
      verbose=self.verbose

    true = accuracy_metrics(y_true, y_pred)
    chroma = accuracy_chroma(y_true, y_pred)
    semitone = accuracy_metrics(y_true, y_pred, semitones_rng=1)
    
    ovrl_true     = max(true["overall"][self.metric], log_min)
    ovrl_chroma   = max(chroma["overall"][self.metric], log_min)
    ovrl_semitone = max(semitone["overall"][self.metric], log_min)
    mean_true     = max(true["mean"][self.metric], log_min)
    mean_chroma   = max(chroma["mean"][self.metric], log_min)
    mean_semitone = max(semitone["mean"][self.metric], log_min)

    with np.errstate(all='raise'):
      try:
        ovrl_loss = -((1/(1+chroma_weight+semitone_weight)) *
                    (np.log(ovrl_true) + chroma_weight*np.log(ovrl_chroma) + semitone_weight*np.log(ovrl_semitone)))
        mean_loss = -((1/(1+chroma_weight+semitone_weight)) *
                    (np.log(mean_true) + chroma_weight*np.log(mean_chroma) + semitone_weight*np.log(mean_semitone)))
                    
        weighted_loss = overall_mean_ratio * ovrl_loss + (1-overall_mean_ratio) * mean_loss
      except Exception as e:
        print(f'\n\t!!!{e}!!!\n{true}\n{chroma}\n{semitone}')
        weighted_loss = -np.log(log_min)
    
    
    if verbose:
      for row in range(y_true.shape[0]):
        print(f'true  - {y_true[row]}       pred  - {y_pred[row]}')
      print(f'overall: true - {-np.log(ovrl_true):.2f}   chrom - {-np.log(ovrl_chroma):.2f}   semit - {-np.log(ovrl_semitone):.2f}')
      print(f'mean   : true - {-np.log(mean_true):.2f}   chrom - {-np.log(mean_chroma):.2f}   semit - {-np.log(mean_semitone):.2f}')
      print(f'overall loss - {ovrl_loss:.2f}   mean loss - {mean_loss:.2f}')

    return weighted_loss

