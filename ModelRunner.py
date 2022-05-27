import os, time
import pickle
import numpy as np

import matplotlib.pyplot as plt
import librosa, librosa.display

from DataGenerator import DataGenerator
from AnalyticModel import AnalyticModel
from NMFModel import NMFModel
from Metrics import *
from VisualizationFunctions import *




analytic_flag = True
NMF_flag = True
ML_model_type = 'conv_net2'

# Main python file to fit and evaluate models on a chosen preprocessed dataset
if __name__ == '__main__':
  path_data = ((os.path.dirname(os.path.abspath(__file__)) + os.path.sep) if '__file__' in locals() 
														else 'gdrive/MyDrive/MSc Diploma/') + 'data' + os.path.sep # colab support for mounted Drive
  path_preprocessed = path_data + 'preprocessed' + os.path.sep
  
  print('Available preprocessed datasets:')
  preprocessed_list = sorted(os.listdir(path_preprocessed))
  for i, dirname in enumerate(preprocessed_list):
    print(f'\t{i+1} - \t{dirname}')
  
  preproc = int(input('Enter the number for the preprocessed data to be used\n'))
  preproc = preproc - 1
  preproc_type = preprocessed_list[preproc].split('_')[2][:3]
  
  if analytic_flag:
    
    if NMF_flag:
    
      dataGen = DataGenerator(spectra_path=preprocessed_list[preproc], shuffle_after_epochs=False, use_aug_transposed=False, use_aug_noised=False,
                              analytic_mode=True, filter_type='bessel', spectra_scale='amplitude')
      nmfModel = NMFModel(dataGenerator=dataGen, use_attack_frames=False, use_noise_frames=False,        
               use_picked_frames=True, use_finger_frames=False, use_ebow_frames=False,        
               window_type='boxcar', normalize_basis=True, filter_basis='bessel')
      
      acc_metrics, preds, gtruths = nmfModel.evaluate(dataGen, threshold=0)
      print(acc_metrics)
      acc_metrics, preds, gtruths = nmfModel.evaluate(dataGen, threshold=0.53)
      print(acc_metrics)
      
    else:
      dataGen = DataGenerator(spectra_path=preprocessed_list[preproc], use_aug_transposed=False, use_aug_noised=False, analytic_mode=True, filter_type='bessel')
      fftModel = AnalyticModel(model_type=preproc_type, hyp_param_search='sequential', hyp_param_runs=100, hyp_param_grid=15, hyp_param_size=0.5, max_overtones=5,
                ot_filter_type='quad',peak_prom_ratio=0.59444,neighbor_weight=0.55555,ovrl_threshold=4.4444, #model_path='20220508_160118__FFT_sequential80_10_0.166'
                )
                
                
      fftModel.fit(dataGen)
      
      dataGen = DataGenerator(spectra_path=preprocessed_list[preproc], use_aug_transposed=False, use_aug_noised=False, analytic_mode=True, filter_type='bessel', shuffle_after_epochs=False)
      ac_loss, preds, gtruths = fftModel.evaluate(dataGen, return_preds=True, load_generator_specs=True, save_preds=True)
      
      dataGen = DataGenerator(spectra_path=preprocessed_list[preproc], use_aug_transposed=True, use_aug_noised=True, analytic_mode=True, filter_type='bessel', shuffle_after_epochs=False)
      ac_loss, preds, gtruths = fftModel.evaluate(dataGen, return_preds=True, load_generator_specs=True, save_preds=True, skip_thresholding=True)
      print()
      print(accuracy_metrics(gtruths, preds))
      print(ac_loss)
    
    
  else:
    from MLModels import *
    from MetricsBinarized import *
    
  
    dataGen = DataGenerator(spectra_path=preprocessed_list[preproc], batch_size=162, shuffle_after_epochs=True, shuffle_consecutive_frames=9, valid_test_split=[0.1, 0.1],
                          use_aug_transposed=True, use_aug_noised=True, analytic_mode=False, filter_type='random',spectra_scale='db',include_empty_frames=False,)
  
    m = MLModel(model_type=ML_model_type, dataGenerator=dataGen)
    m.fit_model()
    
    dataGen = DataGenerator(spectra_path=preprocessed_list[preproc], batch_size=162, shuffle_after_epochs=True, shuffle_consecutive_frames=9, valid_test_split=[0.1, 0.1],
                          use_aug_transposed=True, use_aug_noised=True, analytic_mode=False, filter_type='bessel',spectra_scale='db',include_empty_frames=False,)
    m.evaluate_model()
    dataGen.reset_generators()
    m.predict_model(x_generator=dataGen, separate_chunks=False)
    