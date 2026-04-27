//+------------------------------------------------------------------+
//| NeuralNetworkV4.mqh - CORRECTED VERSION                         |
//| Peter - Version 4.01                                             |
//|                                                                  |
//| Fixes Applied:                                                   |
//| - Inverted dropout (proper scaling)                             |
//| - L2 regularization                                             |
//| - Training mode flag                                            |
//| - Weight get/set for early stopping                             |
//| - SetTrainingMode() method added                                |
//+------------------------------------------------------------------+
#ifndef GOLDENGAUSS_NN_V4_MQH
#define GOLDENGAUSS_NN_V4_MQH

#include "Types.mqh"

//+------------------------------------------------------------------+
//| Neural Network V4 with He-Init + PReLU + Dropout + L2           |
//+------------------------------------------------------------------+
class CNeuralNetworkV4 {
private:
   int m_input_size;
   int m_hidden_size;
   int m_output_size;
   double m_learning_rate;
   double m_l2_lambda;                    // L2 regularization
   bool m_is_trained;
   bool m_training_mode;                  // Training vs inference
   
   double m_weights_ih[];
   double m_weights_ho[];
   double m_bias_h[];
   double m_bias_o[];
   
   // Dropout
   double m_dropout_rate;
   bool m_use_dropout;
   int m_dropout_mask[];
   
   // PReLU alpha
   double m_prelu_alpha;

public:
   //+----------------------------------------------------------+
   //| Constructor                                              |
   //+----------------------------------------------------------+
   CNeuralNetworkV4(int n_inputs, int n_hidden, int n_outputs, double lr = 0.001) {
      m_input_size = n_inputs;
      m_hidden_size = n_hidden;
      m_output_size = n_outputs;
      m_learning_rate = lr;
      m_l2_lambda = 0.0;
      m_is_trained = false;
      m_training_mode = false;            // Default: inference mode
      m_dropout_rate = 0.3;
      m_use_dropout = false;
      m_prelu_alpha = 0.01;
      
      ArrayResize(m_weights_ih, n_inputs * n_hidden);
      ArrayResize(m_weights_ho, n_hidden * n_outputs);
      ArrayResize(m_bias_h, n_hidden);
      ArrayResize(m_bias_o, n_outputs);
      ArrayResize(m_dropout_mask, n_hidden);
      
      HeInitialize(m_weights_ih, n_inputs, n_hidden);
      HeInitialize(m_weights_ho, n_hidden, n_outputs);
      
      ArrayInitialize(m_bias_h, 0.0);
      ArrayInitialize(m_bias_o, 0.0);
   }
   
   //+----------------------------------------------------------+
   //| He-Initialization                                        |
   //+----------------------------------------------------------+
   void HeInitialize(double &weights[], int n_in, int n_out) {
      double stddev = MathSqrt(2.0 / (double)n_in);
      for(int i = 0; i < ArraySize(weights); i++) {
         double u1 = (MathRand() + 1.0) / 32768.0;
         double u2 = (MathRand() + 1.0) / 32768.0;
         double z = MathSqrt(-2.0 * MathLog(u1 + 1e-10)) * MathCos(2.0 * M_PI * u2);
         weights[i] = z * stddev;
      }
   }
   
   //+----------------------------------------------------------+
   //| PReLU                                                    |
   //+----------------------------------------------------------+
   double PReLU(double x) {
      return (x >= 0) ? x : m_prelu_alpha * x;
   }
   
   double PReLUGrad(double x) {
      return (x >= 0) ? 1.0 : m_prelu_alpha;
   }
   
   //+----------------------------------------------------------+
   //| Softmax                                                  |
   //+----------------------------------------------------------+
   void Softmax(double &arr[], int size) {
      double max_val = arr[0];
      for(int i = 1; i < size; i++)
         max_val = MathMax(max_val, arr[i]);
      
      double sum = 0;
      for(int i = 0; i < size; i++)
         sum += MathExp(arr[i] - max_val);
      
      for(int i = 0; i < size; i++)
         arr[i] = MathExp(arr[i] - max_val) / sum;
   }
   
   //+----------------------------------------------------------+
   //| Inverted Dropout                                         |
   //+----------------------------------------------------------+
   void ApplyDropout(double &hidden[]) {
      if(!m_use_dropout || !m_training_mode) return;
      
      double scale = 1.0 / (1.0 - m_dropout_rate);
      
      for(int i = 0; i < m_hidden_size; i++) {
         if(MathRand() / 32767.0 < m_dropout_rate) {
            hidden[i] = 0.0;
         } else {
            hidden[i] *= scale;
         }
      }
   }
   
   //+----------------------------------------------------------+
   //| Forward pass                                             |
   //+----------------------------------------------------------+
   void Forward(const double &inputData[], double &output[]) {
      ArrayResize(output, m_output_size);
      
      double hidden[];
      ArrayResize(hidden, m_hidden_size);
      
      for(int j = 0; j < m_hidden_size; j++) {
         double sum = m_bias_h[j];
         for(int i = 0; i < m_input_size; i++)
            sum += inputData[i] * m_weights_ih[i * m_hidden_size + j];
         hidden[j] = PReLU(sum);
      }
      
      ApplyDropout(hidden);
      
      double raw_output[];
      ArrayResize(raw_output, m_output_size);
      
      for(int k = 0; k < m_output_size; k++) {
         raw_output[k] = m_bias_o[k];
         for(int j = 0; j < m_hidden_size; j++)
            raw_output[k] += hidden[j] * m_weights_ho[j * m_output_size + k];
      }
      
      Softmax(raw_output, m_output_size);
      for(int k = 0; k < m_output_size; k++)
         output[k] = raw_output[k];
   }
   
   //+----------------------------------------------------------+
   //| Train with L2 Regularization                             |
   //+----------------------------------------------------------+
   double Train(const double &features[], int label) {
      double output[];
      Forward(features, output);
      
      double target[];
      ArrayResize(target, m_output_size);
      ArrayInitialize(target, 0.0);
      if(label >= 0 && label < m_output_size)
         target[label] = 1.0;
      
      // Cross-Entropy Loss
      double loss = 0;
      for(int i = 0; i < m_output_size; i++)
         if(output[i] > 1e-10)
            loss -= target[i] * MathLog(output[i]);
      
      // Add L2 regularization term
      if(m_l2_lambda > 0) {
         double l2_sum = 0;
         for(int i = 0; i < ArraySize(m_weights_ih); i++)
            l2_sum += m_weights_ih[i] * m_weights_ih[i];
         for(int i = 0; i < ArraySize(m_weights_ho); i++)
            l2_sum += m_weights_ho[i] * m_weights_ho[i];
         loss += 0.5 * m_l2_lambda * l2_sum;
      }
      
      // Backpropagation
      double delta_output[];
      ArrayResize(delta_output, m_output_size);
      for(int i = 0; i < m_output_size; i++)
         delta_output[i] = output[i] - target[i];
      
      double hidden_sum[];
      ArrayResize(hidden_sum, m_hidden_size);
      double hidden[];
      ArrayResize(hidden, m_hidden_size);
      
      for(int j = 0; j < m_hidden_size; j++) {
         hidden_sum[j] = m_bias_h[j];
         for(int i = 0; i < m_input_size; i++)
            hidden_sum[j] += features[i] * m_weights_ih[i * m_hidden_size + j];
         hidden[j] = PReLU(hidden_sum[j]);
      }
      
      ApplyDropout(hidden);
      
      // Update hidden->output weights with L2
      for(int j = 0; j < m_hidden_size; j++) {
         for(int k = 0; k < m_output_size; k++) {
            int idx = j * m_output_size + k;
            double grad = delta_output[k] * hidden[j] + m_l2_lambda * m_weights_ho[idx];
            m_weights_ho[idx] -= m_learning_rate * grad;
         }
      }
      for(int k = 0; k < m_output_size; k++)
         m_bias_o[k] -= m_learning_rate * delta_output[k];
      
      // Gradient for input->hidden
      double delta_hidden[];
      ArrayResize(delta_hidden, m_hidden_size);
      for(int j = 0; j < m_hidden_size; j++) {
         double grad = 0;
         for(int k = 0; k < m_output_size; k++)
            grad += delta_output[k] * m_weights_ho[j * m_output_size + k];
         delta_hidden[j] = grad * PReLUGrad(hidden_sum[j]);
      }
      
      // Update input weights with L2
      for(int i = 0; i < m_input_size; i++) {
         for(int j = 0; j < m_hidden_size; j++) {
            int idx = i * m_hidden_size + j;
            double grad = delta_hidden[j] * features[i] + m_l2_lambda * m_weights_ih[idx];
            m_weights_ih[idx] -= m_learning_rate * grad;
         }
      }
      for(int j = 0; j < m_hidden_size; j++)
         m_bias_h[j] -= m_learning_rate * delta_hidden[j];
      
      m_is_trained = true;
      return loss;
   }
   
   //+----------------------------------------------------------+
   //| Predict                                                  |
   //+----------------------------------------------------------+
   void Predict(const double &features[], double &probabilities[]) {
      bool was_training = m_training_mode;
      m_training_mode = false;  // Always inference mode for prediction
      Forward(features, probabilities);
      m_training_mode = was_training;
   }
   
   //+----------------------------------------------------------+
   //| ✅ Set Training Mode (ADDED - Fixes compilation error)  |
   //+----------------------------------------------------------+
   void SetTrainingMode(bool training) {
      m_training_mode = training;
   }
   
   //+----------------------------------------------------------+
   //| Get Training Mode                                        |
   //+----------------------------------------------------------+
   bool GetTrainingMode() const {
      return m_training_mode;
   }
   
   //+----------------------------------------------------------+
   //| Set L2 Regularization                                    |
   //+----------------------------------------------------------+
   void SetL2Regularization(double lambda) {
      m_l2_lambda = MathMax(0.0, lambda);
   }
   
   //+----------------------------------------------------------+
   //| Get L2 Regularization                                    |
   //+----------------------------------------------------------+
   double GetL2Regularization() const {
      return m_l2_lambda;
   }
   
   //+----------------------------------------------------------+
   //| Get Weights (for early stopping)                         |
   //+----------------------------------------------------------+
   void GetWeights(double &weights_ih[], double &weights_ho[], 
                   double &bias_h[], double &bias_o[]) {
      ArrayCopy(weights_ih, m_weights_ih);
      ArrayCopy(weights_ho, m_weights_ho);
      ArrayCopy(bias_h, m_bias_h);
      ArrayCopy(bias_o, m_bias_o);
   }
   
   //+----------------------------------------------------------+
   //| Set Weights (for early stopping)                         |
   //+----------------------------------------------------------+
   void SetWeights(const double &weights_ih[], const double &weights_ho[], 
                   const double &bias_h[], const double &bias_o[]) {
      ArrayCopy(m_weights_ih, weights_ih);
      ArrayCopy(m_weights_ho, weights_ho);
      ArrayCopy(m_bias_h, bias_h);
      ArrayCopy(m_bias_o, bias_o);
   }
   
   bool IsTrained() const { return m_is_trained; }
   
   void SetDropout(double rate) {
      m_dropout_rate = MathMax(0.0, MathMin(0.5, rate));
      m_use_dropout = (m_dropout_rate > 0);
   }
   
   double GetDropout() const { return m_dropout_rate; }
   
   //+----------------------------------------------------------+
   //| Get Architecture Info                                    |
   //+----------------------------------------------------------+
   void GetInfo(int &inputs, int &hidden, int &outputs) {
      inputs = m_input_size;
      hidden = m_hidden_size;
      outputs = m_output_size;
   }
   
   //+----------------------------------------------------------+
   //| Save model                                               |
   //+----------------------------------------------------------+
   bool Save(const string filename) {
      string clean = PreparePath(filename);
      int file = FileOpen(clean, FILE_WRITE | FILE_BIN);
      if(file == INVALID_HANDLE) {
         Print("[NN-V4] Cannot open file: ", clean);
         return false;
      }
      
      FileWriteInteger(file, m_input_size);
      FileWriteInteger(file, m_hidden_size);
      FileWriteInteger(file, m_output_size);
      FileWriteDouble(file, m_l2_lambda);
      FileWriteDouble(file, m_dropout_rate);
      FileWriteDouble(file, m_prelu_alpha);
      
      for(int i = 0; i < ArraySize(m_weights_ih); i++)
         FileWriteDouble(file, m_weights_ih[i]);
      for(int i = 0; i < ArraySize(m_weights_ho); i++)
         FileWriteDouble(file, m_weights_ho[i]);
      for(int i = 0; i < ArraySize(m_bias_h); i++)
         FileWriteDouble(file, m_bias_h[i]);
      for(int i = 0; i < ArraySize(m_bias_o); i++)
         FileWriteDouble(file, m_bias_o[i]);
      
      FileClose(file);
      Print("[NN-V4] Saved: ", filename);
      return true;
   }
   
   //+----------------------------------------------------------+
   //| Load model                                               |
   //+----------------------------------------------------------+
   bool Load(const string filename) {
      string clean = PreparePath(filename);
      int file = FileOpen(clean, FILE_READ | FILE_BIN);
      if(file == INVALID_HANDLE) {
         Print("[NN-V4] Cannot open file: ", clean);
         return false;
      }
      
      int in_s = FileReadInteger(file);
      int hid_s = FileReadInteger(file);
      int out_s = FileReadInteger(file);
      
      if(in_s != m_input_size || hid_s != m_hidden_size || out_s != m_output_size) {
         Print("[NN-V4] Size mismatch!");
         FileClose(file);
         return false;
      }
      
      if(!FileIsEnding(file)) m_l2_lambda = FileReadDouble(file);
      if(!FileIsEnding(file)) m_dropout_rate = FileReadDouble(file);
      if(!FileIsEnding(file)) m_prelu_alpha = FileReadDouble(file);
      
      for(int i = 0; i < ArraySize(m_weights_ih); i++)
         m_weights_ih[i] = FileReadDouble(file);
      for(int i = 0; i < ArraySize(m_weights_ho); i++)
         m_weights_ho[i] = FileReadDouble(file);
      for(int i = 0; i < ArraySize(m_bias_h); i++)
         m_bias_h[i] = FileReadDouble(file);
      for(int i = 0; i < ArraySize(m_bias_o); i++)
         m_bias_o[i] = FileReadDouble(file);
      
      FileClose(file);
      m_is_trained = true;
      Print("[NN-V4] Loaded: ", filename);
      return true;
   }
   
   //+----------------------------------------------------------+
   //| Prepare path for FileOpen                                |
   //+----------------------------------------------------------+
   static string PreparePath(const string path) {
      string p = path;
      if(StringFind(p, "Files\\") == 0) return StringSubstr(p, 6);
      if(StringFind(p, "Files/") == 0) return StringSubstr(p, 7);
      return p;
   }
};

//+------------------------------------------------------------------+
//| Training Buffer V4                                               |
//+------------------------------------------------------------------+
class CTrainingBufferV4 {
private:
   STrainingSample m_samples[];
   int m_capacity;
   int m_count;

public:
   CTrainingBufferV4(int capacity = 10000) {
      m_capacity = capacity;
      m_count = 0;
      ArrayResize(m_samples, capacity);
   }
   
   void Add(const double &features[], int label, double outcome) {
      if(m_count >= m_capacity) return;
      
      int nf = ArraySize(features);
      ArrayResize(m_samples[m_count].features, nf);
      for(int i = 0; i < nf; i++)
         m_samples[m_count].features[i] = features[i];
      
      m_samples[m_count].label = label;
      m_samples[m_count].outcome = outcome;
      m_samples[m_count].timestamp = TimeCurrent();
      m_count++;
   }
   
   bool Get(int index, double &features[], int &label, double &outcome) {
      if(index < 0 || index >= m_count) return false;
      
      int nf = ArraySize(m_samples[index].features);
      ArrayResize(features, nf);
      for(int i = 0; i < nf; i++)
         features[i] = m_samples[index].features[i];
      
      label = m_samples[index].label;
      outcome = m_samples[index].outcome;
      return true;
   }
   
   int Size() const { return m_count; }
   
   void Clear() { m_count = 0; }
};

#endif
