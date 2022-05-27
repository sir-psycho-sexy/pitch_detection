# Metrics for binarized labels (for ML models)
# (using TF tensors and backend to ensure differentiability)
import tensorflow as tf

try:
  from Metrics import *
except:
  print('Metrics not found in directory!')


MAX_LOSS = tf.math.log(log_min)
PROB_THRESHOLD = 0.5

# function to calculate loss via mean absolute error between the ground truth and predictions
def loss_MAE(y_true, y_pred):
  absolute_difference = tf.abs(tf.cast(y_true, dtype=tf.float32) - tf.cast(y_pred, dtype=tf.float32))
  return tf.reduce_mean(absolute_difference, axis=-1)


# function to calculate loss via mean absolute squared between the ground truth and predictions
def loss_MSE(y_true, y_pred):
  squared_difference = tf.square(tf.cast(y_true, dtype=tf.float32) - tf.cast(y_pred, dtype=tf.float32))
  return tf.reduce_mean(squared_difference, axis=-1)


# function to calculate loss based on cosine similarity (->[0,1] interval from most to least similar)
def loss_cos_similarity(y_true, y_pred):
  true_normalized = tf.math.l2_normalize(1-tf.cast(y_true, dtype=tf.float32), axis=1)
  pred_normalized = tf.math.l2_normalize(1-tf.cast(y_pred, dtype=tf.float32), axis=1)
  
  return 1-tf.reduce_sum(true_normalized * pred_normalized, axis=1)

# calculate loss from precision for given predictions (correct guesses/total guesses)
def loss_precision(y_true, y_pred):
  unmatched_trues = tf.nn.relu(tf.cast(y_true, dtype=tf.float32)-tf.cast(y_pred, dtype=tf.float32))
  matches = tf.cast(y_true, dtype=tf.float32) - unmatched_trues

  #overall_precision = tf.reduce_sum(matches)/tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
  mean_precision = tf.math.divide_no_nan(tf.reduce_sum(matches, axis=1), tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32), axis=1))
  return tf.math.log(tf.clip_by_value(mean_precision, 1e-5, 1.0)) / MAX_LOSS

# calculate loss from recall for given predictions (correct guesses/total true labels)
def loss_recall(y_true, y_pred):
  unmatched_trues = tf.nn.relu(tf.cast(y_true, dtype=tf.float32)-tf.cast(y_pred, dtype=tf.float32))
  matches = tf.cast(y_true, dtype=tf.float32) - unmatched_trues

  #overall_recall = tf.reduce_sum(matches)/tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
  mean_recall = tf.math.divide_no_nan(tf.reduce_sum(matches, axis=1), tf.reduce_sum(tf.cast(y_true, dtype=tf.float32), axis=1))
  return tf.math.log(tf.clip_by_value(mean_recall, 1e-5, 1.0)) / MAX_LOSS


# calculate loss from F1 score of given predictions based on precision and recall
def loss_f1(y_true, y_pred):
  unmatched_trues = tf.nn.relu(tf.cast(y_true, dtype=tf.float32)-tf.cast(y_pred, dtype=tf.float32))
  matches = tf.cast(y_true, dtype=tf.float32) - unmatched_trues

  mean_precision = tf.math.divide_no_nan(tf.reduce_sum(matches, axis=1), tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32), axis=1))
  mean_recall = tf.math.divide_no_nan(tf.reduce_sum(matches, axis=1), tf.reduce_sum(tf.cast(y_true, dtype=tf.float32), axis=1))

  f1 = 2 * tf.math.divide_no_nan((mean_precision * mean_recall), (mean_precision + mean_recall))
  #f1 = tf.where(tf.math.logical_and(tf.math.is_nan(mean_precision), tf.math.is_nan(mean_recall)), tf.ones_like(f1), f1)
  #f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

  return tf.math.log(tf.clip_by_value(f1, 1e-5, 1.0)) / MAX_LOSS



# function to calculate loss based on cosine similarity, ignoring the octave of the notes
def loss_octave(y_true, y_pred, loss_metric=loss_cos_similarity):  # use sum of octave bins with cos similarity
  padded_true = tf.pad(y_true, [[0, 0],[0,-tf.shape(y_true)[1]%12]], 'CONSTANT')
  padded_pred = tf.pad(y_pred, [[0, 0],[0,-tf.shape(y_pred)[1]%12]], 'CONSTANT')
  
  chroma_true = tf.reduce_max(tf.reshape(padded_true, [tf.shape(padded_true)[0], tf.shape(padded_true)[1]//12, 12]), axis=1)
  chroma_pred = tf.reduce_max(tf.reshape(padded_pred, [tf.shape(padded_pred)[0], tf.shape(padded_pred)[1]//12, 12]), axis=1)
  return loss_metric(chroma_true, chroma_pred)

# function to calculate loss based on cosine similarity, ignoring the octave of the notes and the class for empty notes
def loss_octave_empty(y_true, y_pred, loss_metric=loss_cos_similarity):
  chroma_true = tf.reduce_sum(tf.reshape(y_true[:,1:], [tf.shape(y_true)[0], (tf.shape(y_true)[1]-1)//12, 12]), axis=1)
  chroma_pred = tf.reduce_sum(tf.reshape(y_pred[:,1:], [tf.shape(y_pred)[0], (tf.shape(y_pred)[1]-1)//12, 12]), axis=1)
  return loss_metric(chroma_true, chroma_pred)


# function to calculate loss based on cosine similarity, ignoring semitone errors
def loss_semitone(y_true, y_pred, loss_metric=loss_cos_similarity):
  # separate matched and unmatched notes
  unmatched_preds = tf.nn.relu(tf.cast(y_pred, dtype=tf.float32)-tf.cast(y_true, dtype=tf.float32))
  unmatched_trues = tf.nn.relu(tf.cast(y_true, dtype=tf.float32)-tf.cast(y_pred, dtype=tf.float32))
  matches = tf.cast(y_true, dtype=tf.float32) - unmatched_trues

  # get the semitone shifts of unmatched notes
  lower_preds = tf.roll(unmatched_preds, shift=-1, axis=1)
  upper_preds = tf.roll(unmatched_preds, shift= 1, axis=1)
  
  # get notes which are a match after the semitone shift
  semitone_matches = unmatched_trues - tf.nn.relu(unmatched_trues - lower_preds - upper_preds)

  # get notes that are still unmatched
  lower_unmatched = tf.roll(tf.nn.relu(lower_preds-semitone_matches), shift= 1, axis=1)
  upper_unmatched = tf.roll(tf.nn.relu(upper_preds-semitone_matches), shift=-1, axis=1)
  unmatched = tf.minimum(lower_unmatched, upper_unmatched)
  
  # calculate new loss with the semitone errors rectified
  return loss_metric(y_true, matches + semitone_matches + unmatched)



# Passable loss metric for ML model building function
class MLMetrics:
  def __init__(self, accuracy_metric='f1',  # metric which should be used to calculate loss (precision, recall, f1)
               chroma_weight=0.40,          # weight of the metric when ignoring octave errors in the loss
               semitone_weight=0.10,        # weight of the metric when ignoring semitone errors in the loss t
               primary_metric = 'accuracy', # primary metric when calculating loss (MAE, MSE, cos_sim, accuracy)
               secondary_metric = 'BCE',    # secondary metric when calculating loss (MAE, MSE, cos_sim, accuracy)
               secondary_metric_ratio = 0.5 # ratio to weigh cos similarity and secondary metric when calculating loss
               ):
    self.accuracy_metric={'precision': loss_precision, 'recall': loss_recall, 'f1':loss_f1}[accuracy_metric]
    self.chroma_weight=chroma_weight
    self.semitone_weight=semitone_weight
    
    self.primary_metric  ={'MAE': loss_MAE, 'MSE': loss_MSE, 'cos_sim': loss_cos_similarity, 'accuracy': self.accuracy_metric, 'BCE': tf.keras.losses.binary_crossentropy}[primary_metric]
    self.secondary_metric={'MAE': loss_MAE, 'MSE': loss_MSE, 'cos_sim': loss_cos_similarity, 'accuracy': self.accuracy_metric, 'BCE': tf.keras.losses.binary_crossentropy}[secondary_metric]
    self.secondary_metric_ratio=secondary_metric_ratio
    
    self.weight_sum = 1 + chroma_weight + semitone_weight

  # loss based on cosine similarity and a secondary distance metric (like MAE/MSE)
  # uses weighted version of the same metrics for when octave/semitone errors are ignored 
  def full_loss(self, y_true, y_pred):
    return (1 - self.secondary_metric_ratio) * self.primary_loss(y_true, y_pred) + self.secondary_metric_ratio * self.secondary_loss(y_true, y_pred)


  # loss based on primary metric defined in the class
  # uses weighted version of the same metric for when octave/semitone errors are ignored 
  def primary_loss(self, y_true, y_pred):
    
    primary_loss = 0.0
    primary_loss += self.primary_metric(y_true, y_pred)
    primary_loss += self.chroma_weight * loss_octave(y_true, y_pred, loss_metric=self.primary_metric)
    primary_loss += self.semitone_weight * loss_semitone(y_true, y_pred, loss_metric=self.primary_metric)

    return primary_loss/self.weight_sum

  # loss based on secondary metric defined in the class
  # uses weighted version of the same metric for when octave/semitone errors are ignored 
  def secondary_loss(self, y_true, y_pred):
    
    secondary_loss = 0.0
    secondary_loss += self.secondary_metric(y_true, y_pred)
    secondary_loss += self.chroma_weight * loss_octave(y_true, y_pred, loss_metric=self.secondary_metric)
    secondary_loss += self.semitone_weight * loss_semitone(y_true, y_pred, loss_metric=self.secondary_metric)

    return secondary_loss/self.weight_sum







# Metrics for ML model accuracy (using probability labels)
# calculate precision for given prediction probabilities (correct guesses/total guesses)
def pr_precision(y_true, y_pred):
  unmatched_trues = tf.nn.relu(tf.cast(y_true, dtype=tf.float32)-tf.cast(y_pred, dtype=tf.float32))
  matches = tf.cast(y_true, dtype=tf.float32) - unmatched_trues

  overall_precision = tf.reduce_sum(matches)/tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
  mean_precision = tf.reduce_sum(matches, axis=1)/tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32), axis=1)
  return tf.where(tf.math.is_nan(mean_precision), tf.ones_like(mean_precision), mean_precision)

# calculate recall for given prediction probabilities (correct guesses/total true labels)
def pr_recall(y_true, y_pred):
  unmatched_trues = tf.nn.relu(tf.cast(y_true, dtype=tf.float32)-tf.cast(y_pred, dtype=tf.float32))
  matches = tf.cast(y_true, dtype=tf.float32) - unmatched_trues

  overall_recall = tf.reduce_sum(matches)/tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
  mean_recall = tf.reduce_sum(matches, axis=1)/tf.reduce_sum(tf.cast(y_true, dtype=tf.float32), axis=1)
  return tf.where(tf.math.is_nan(mean_recall), tf.ones_like(mean_recall), mean_recall)


# calculate F1 score of given prediction probabilities based on precision and recall
def pr_f1(y_true, y_pred):
  unmatched_trues = tf.nn.relu(tf.cast(y_true, dtype=tf.float32)-tf.cast(y_pred, dtype=tf.float32))
  matches = tf.cast(y_true, dtype=tf.float32) - unmatched_trues

  mean_precision = tf.reduce_sum(matches, axis=1)/tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32), axis=1)
  mean_recall = tf.reduce_sum(matches, axis=1)/tf.reduce_sum(tf.cast(y_true, dtype=tf.float32), axis=1) 

  f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
  f1 = tf.where(tf.math.logical_and(tf.math.is_nan(mean_precision), tf.math.is_nan(mean_recall)), tf.ones_like(f1), f1)
  f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
  return f1
  



# Metrics for ML model accuracy (using binarized prediction labels)

# transforms probability output of model into binary predictions (max selection + thresholding)
def prob_to_pred(y_pred):
  max_probs = tf.math.top_k(y_pred, k=6).values
  thresholded = tf.math.maximum(tf.math.reduce_min(max_probs, axis=1), PROB_THRESHOLD)
  preds = tf.where(tf.greater_equal(y_pred, tf.expand_dims(thresholded, axis=1)), tf.ones_like(y_pred), tf.zeros_like(y_pred))
  return preds


# calculate precision for given predictions (correct guesses/total guesses)
def precision(y_true, y_pred):
  return pr_precision(tf.cast(y_true, dtype=tf.float32), prob_to_pred(tf.cast(y_pred, dtype=tf.float32)))

# calculate recall for given predictions (correct guesses/total true labels)
def recall(y_true, y_pred):
  return pr_recall(tf.cast(y_true, dtype=tf.float32), prob_to_pred(tf.cast(y_pred, dtype=tf.float32)))


# calculate F1 score of given predictions based on precision and recall
def f1(y_true, y_pred):
  return pr_f1(tf.cast(y_true, dtype=tf.float32), prob_to_pred(tf.cast(y_pred, dtype=tf.float32)))
