//+------------------------------------------------------------------+
//| GoldenGauss_Trainer_V5.mq5                                      |
//| Clean Neural Network Trainer EA for GoldenGauss                 |
//| Version: 5.00                                                   |
//+------------------------------------------------------------------+
#property copyright "Peter"
#property version   "5.00"
#property description "GoldenGauss Neural Network Trainer EA"
#property strict

//+------------------------------------------------------------------+
//| INCLUDES                                                        |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
#include <GoldenGauss/Core/Types.mqh>
#include <GoldenGauss/Core/NeuralNetworkV4.mqh>
#include <GoldenGauss/Features/FeatureBuilder.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                |
//+------------------------------------------------------------------+
input group "=== Data Range ==="
input datetime StartDate      = D'2025.01.01';
input datetime EndDate        = D'2026.12.31';
input int       LookbackBars  = 3000;
input int       ForwardBars   = 10;

input group "=== Training ==="
input int       MinSamples    = 50;
input int       MaxEpochs     = 100;
input double    ValidationPct = 20.0;

input group "=== Neural Network ==="
input int       HiddenNeurons = 64;
input double    LearningRate  = 0.001;

input group "=== Output Files ==="
input string    DataFolder    = "GoldenGauss_Data";
input string    BuyModelFile  = "BULLISH_V5.nn";
input string    SellModelFile = "BEARISH_V5.nn";

input group "=== Auto-Training ==="
input bool      TrainOnStart  = true;
input int       RetrainBars   = 1000;

input group "=== Debug ==="
input bool      ShowDetails   = true;

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                |
//+------------------------------------------------------------------+
CNeuralNetworkV4*  g_nnBuy       = NULL;
CNeuralNetworkV4*  g_nnSell      = NULL;
CTrainingBufferV4* g_trainBuf    = NULL;
CTrainingBufferV4* g_valBuf      = NULL;
double             g_means[];
double             g_stds[];
int                g_numFeatures = NUM_FEATURES;

// Price data (matches FeatureBuilder externs)
double             g_close[];
double             g_open[];
double             g_high[];
double             g_low[];
long               g_volume[];
int                g_bars = 0;

// Training state
bool               g_training_done = false;
int                g_bars_count = 0;
datetime           g_last_bar_time = 0;

//+------------------------------------------------------------------+
//| ONINIT                                                          |
//+------------------------------------------------------------------+
int OnInit() {
   Print("========================================");
   Print("  GoldenGauss Trainer EA v5.00");
   Print("  Symbol: ", _Symbol, " | TF: ", EnumToString(Period()));
   Print("========================================");
   
   // Initialize arrays
   ArrayResize(g_means, g_numFeatures);
   ArrayResize(g_stds, g_numFeatures);
   ArrayInitialize(g_means, 0.0);
   ArrayInitialize(g_stds, 1.0);
   
   // Initialize price arrays
   ArrayResize(g_close, LookbackBars);
   ArrayResize(g_open, LookbackBars);
   ArrayResize(g_high, LookbackBars);
   ArrayResize(g_low, LookbackBars);
   ArrayResize(g_volume, LookbackBars);
   ArraySetAsSeries(g_close, true);
   ArraySetAsSeries(g_open, true);
   ArraySetAsSeries(g_high, true);
   ArraySetAsSeries(g_low, true);
   ArraySetAsSeries(g_volume, true);
   
   // Initialize networks
   g_nnBuy  = new CNeuralNetworkV4(g_numFeatures, HiddenNeurons, 2, LearningRate);
   g_nnSell = new CNeuralNetworkV4(g_numFeatures, HiddenNeurons, 2, LearningRate);
   g_trainBuf = new CTrainingBufferV4(10000);
   g_valBuf   = new CTrainingBufferV4(5000);
   
   // Register feature calculators
   g_featureBuilder.RegisterCalculator(new CVolatilityFeatures());
   g_featureBuilder.RegisterCalculator(new CMomentumFeatures());
   g_featureBuilder.RegisterCalculator(new CVWAPFeatures());
   g_featureBuilder.RegisterCalculator(new CVolumeFeatures());
   g_featureBuilder.RegisterCalculator(new CStructureFeatures());
   g_featureBuilder.RegisterCalculator(new CMicroFeatures());
   g_featureBuilder.RegisterCalculator(new CTemporalFeatures());
   
   Print("[EA] Initialized successfully");
   Print("[EA] Train on Start: ", TrainOnStart);
   Print("[EA] Retrain every ", RetrainBars, " bars");
   
   // Train immediately if enabled
   if(TrainOnStart) {
      Print("[EA] Starting initial training...");
      TrainModels();
      g_training_done = true;
   }
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| ONDEINIT                                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if(g_nnBuy   != NULL) delete g_nnBuy;
   if(g_nnSell  != NULL) delete g_nnSell;
   if(g_trainBuf != NULL) delete g_trainBuf;
   if(g_valBuf   != NULL) delete g_valBuf;
   g_featureBuilder.FreeCalculators();
   Print("[EA] Deinitialized");
}

//+------------------------------------------------------------------+
//| ONTICK                                                          |
//+------------------------------------------------------------------+
void OnTick() {
   // Count new bars
   datetime current_bar = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(current_bar != g_last_bar_time) {
      g_bars_count++;
      g_last_bar_time = current_bar;
   }
   
   // Periodic retraining
   if(RetrainBars > 0 && g_training_done) {
      if(g_bars_count >= RetrainBars) {
         Print("[EA] Retraining triggered (", g_bars_count, " bars)");
         TrainModels();
         g_bars_count = 0;
      }
   }
}

//+------------------------------------------------------------------+
//| MAIN TRAINING FUNCTION                                          |
//+------------------------------------------------------------------+
void TrainModels() {
   // Step 1: Load price data
   Print("\n[1/6] Loading price data...");
   
   ArraySetAsSeries(g_close,  true);
   ArraySetAsSeries(g_open,   true);
   ArraySetAsSeries(g_high,   true);
   ArraySetAsSeries(g_low,    true);
   ArraySetAsSeries(g_volume, true);
   
   g_bars = CopyClose(_Symbol, PERIOD_CURRENT, 0, LookbackBars, g_close);
   if(g_bars <= 0) {
      Print("ERROR: Cannot load price data!");
      return;
   }
   
   CopyOpen(_Symbol,   PERIOD_CURRENT, 0, LookbackBars, g_open);
   CopyHigh(_Symbol,   PERIOD_CURRENT, 0, LookbackBars, g_high);
   CopyLow(_Symbol,    PERIOD_CURRENT, 0, LookbackBars, g_low);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, LookbackBars, g_volume);
   
   datetime firstBar = iTime(_Symbol, PERIOD_CURRENT, g_bars - 1);
   datetime lastBar  = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   Print("  Loaded: ", g_bars, " bars");
   Print("  Range:  ", TimeToString(firstBar), " - ", TimeToString(lastBar));
   
   // Step 2: Collect training samples
   Print("\n[2/6] Collecting samples...");
   
   int buyCount  = 0;
   int sellCount = 0;
   int skipCount = 0;
   
   for(int i = 50; i < g_bars - ForwardBars - 5; i++) {
      datetime barTime = iTime(_Symbol, PERIOD_CURRENT, i);
      
      // Check date range
      if(barTime < StartDate || barTime > EndDate) {
         skipCount++;
         continue;
      }
      
      // Build feature vector (uses global g_close, g_open, etc.)
      double features[];
      ArrayResize(features, g_numFeatures);
      
      if(!g_featureBuilder.BuildFeatures(features, g_numFeatures, g_bars)) {
         skipCount++;
         continue;
      }
      
      // Get label
      int label = GetLabel(i);
      
      if(label == 0) {
         skipCount++;
         continue;
      }
      
      // Normalize features
      NormalizeFeatures(features);
      
      // Add to buffer
      g_trainBuf.Add(features, label, 0);
      
      if(label == 1) buyCount++;
      else           sellCount++;
      
      // Show first 5 samples
      if(ShowDetails && (buyCount + sellCount) <= 5) {
         Print("  Sample #", (buyCount + sellCount), ": ", TimeToString(barTime),
               " | Label: ", (label == 1) ? "BUY" : "SELL");
      }
   }
   
   int totalSamples = buyCount + sellCount;
   Print("  Total samples: ", totalSamples, " (BUY: ", buyCount, " | SELL: ", sellCount, ")");
   Print("  Skipped: ", skipCount);
   
   if(totalSamples < MinSamples) {
      Print("\nERROR: Not enough samples! Need ", MinSamples, ", got ", totalSamples);
      Print("\nSOLUTIONS:");
      Print("  1. Expand date range (StartDate/EndDate)");
      Print("  2. Increase LookbackBars (currently ", LookbackBars, ")");
      Print("  3. Check if historical data exists for this symbol");
      return;
   }
   
   // Step 3: Calculate normalization from ALL samples
   Print("\n[3/6] Computing normalization...");
   ComputeNormalization();
   
   // Step 4: Split train/validation
   Print("\n[4/6] Splitting data...");
   SplitData(ValidationPct);
   Print("  Train: ", g_trainBuf.Size(), " | Validation: ", g_valBuf.Size());
   
   // Step 5: Train models
   Print("\n[5/6] Training models...");
   
   Print("  Training BUY model...");
   double buyLoss = TrainModel(g_nnBuy, 1, "BUY");
   
   Print("  Training SELL model...");
   double sellLoss = TrainModel(g_nnSell, 2, "SELL");
   
   // Step 6: Save everything
   Print("\n[6/6] Saving files...");
   
   string folder = "MQL5/Files/" + DataFolder + "_" + _Symbol + "/Models/";
   
   bool ok1 = g_nnBuy.Save(folder + BuyModelFile);
   bool ok2 = g_nnSell.Save(folder + SellModelFile);
   bool ok3 = SaveNormalization(folder + "norm_params.dat");
   
   Print("\n========================================");
   if(ok1 && ok2 && ok3) {
      Print("  SUCCESS!");
      Print("  BUY Loss:  ", DoubleToString(buyLoss, 6));
      Print("  SELL Loss: ", DoubleToString(sellLoss, 6));
      Print("\n  Files saved:");
      Print("    ", folder, BuyModelFile);
      Print("    ", folder, SellModelFile);
      Print("    ", folder, "norm_params.dat");
   } else {
      Print("  PARTIAL FAILURE");
      if(!ok1) Print("    BUY model failed");
      if(!ok2) Print("    SELL model failed");
      if(!ok3) Print("    Normalization failed");
   }
   Print("========================================");
}

//+------------------------------------------------------------------+
//| GET LABEL - Simple & Relaxed                                    |
//+------------------------------------------------------------------+
int GetLabel(int idx) {
   if(idx < 8 || idx >= g_bars) return 0;
   
   // Calculate RSI (7 period)
   double gain = 0, loss = 0;
   for(int j = 1; j <= 7 && (idx + j) < g_bars; j++) {
      double diff = g_close[idx + j] - g_close[idx + j + 1];
      if(diff > 0) gain += diff;
      else         loss += MathAbs(diff);
   }
   
   double rsi = 50;
   if(loss > 0.00001) 
      rsi = 100.0 - (100.0 / (1.0 + gain / loss));
   
   // Calculate forward return
   double fwdRet = 0;
   if((idx + ForwardBars) < g_bars)
      fwdRet = (g_close[idx + ForwardBars] - g_close[idx]) / g_close[idx] * 100.0;
   
   // RELAXED criteria - should find samples
   double rsiBuy  = 45;
   double rsiSell = 55;
   double minRet  = 0.01;  // Very small threshold
   
   // BUY signal
   if(rsi < rsiBuy && fwdRet > minRet)
      return 1;
   
   // SELL signal
   if(rsi > rsiSell && fwdRet < -minRet)
      return 2;
   
   return 0;
}

//+------------------------------------------------------------------+
//| NORMALIZE FEATURES - Min-Max Scaling                            |
//+------------------------------------------------------------------+
void NormalizeFeatures(double &features[]) {
   if(ArraySize(features) == 0) return;
   
   double minVal = features[0];
   double maxVal = features[0];
   
   for(int j = 1; j < ArraySize(features); j++) {
      if(features[j] < minVal) minVal = features[j];
      if(features[j] > maxVal) maxVal = features[j];
   }
   
   double range = (maxVal - minVal);
   if(range < 0.00001) range = 1.0;
   
   for(int j = 0; j < ArraySize(features); j++) {
      features[j] = (features[j] - minVal) / range * 2.0 - 1.0;  // Scale to [-1, 1]
   }
}

//+------------------------------------------------------------------+
//| COMPUTE NORMALIZATION STATS                                     |
//+------------------------------------------------------------------+
void ComputeNormalization() {
   if(g_trainBuf.Size() == 0) return;
   
   // Initialize
   for(int j = 0; j < g_numFeatures; j++) {
      g_means[j] = 0;
      g_stds[j]  = 1;
   }
   
   // Calculate means
   for(int i = 0; i < g_trainBuf.Size(); i++) {
      double feat[]; int lbl; double out;
      if(g_trainBuf.Get(i, feat, lbl, out)) {
         for(int j = 0; j < g_numFeatures && j < ArraySize(feat); j++) {
            g_means[j] += feat[j];
         }
      }
   }
   
   for(int j = 0; j < g_numFeatures; j++)
      g_means[j] /= g_trainBuf.Size();
   
   // Calculate std
   for(int i = 0; i < g_trainBuf.Size(); i++) {
      double feat[]; int lbl; double out;
      if(g_trainBuf.Get(i, feat, lbl, out)) {
         for(int j = 0; j < g_numFeatures && j < ArraySize(feat); j++) {
            g_stds[j] += MathPow(feat[j] - g_means[j], 2);
         }
      }
   }
   
   for(int j = 0; j < g_numFeatures; j++)
      g_stds[j] = MathSqrt(g_stds[j] / g_trainBuf.Size()) + 0.0001;
   
   Print("  Mean range: [", DoubleToString(ArrayMinimum(g_means), 4), 
         ", ", DoubleToString(ArrayMaximum(g_means), 4), "]");
   Print("  Std range:  [", DoubleToString(ArrayMinimum(g_stds), 4), 
         ", ", DoubleToString(ArrayMaximum(g_stds), 4), "]");
}

//+------------------------------------------------------------------+
//| SPLIT DATA                                                      |
//+------------------------------------------------------------------+
void SplitData(double valPct) {
   int total   = g_trainBuf.Size();
   int valSize = (int)(total * valPct / 100.0);
   int trainSize = total - valSize;
   
   CTrainingBufferV4* temp = new CTrainingBufferV4(total);
   
   // Copy all to temp
   for(int i = 0; i < total; i++) {
      double feat[]; int lbl; double out;
      if(g_trainBuf.Get(i, feat, lbl, out))
         temp.Add(feat, lbl, out);
   }
   
   // Clear and resize
   delete g_trainBuf;
   delete g_valBuf;
   g_trainBuf = new CTrainingBufferV4(trainSize + 100);
   g_valBuf   = new CTrainingBufferV4(valSize + 100);
   
   // Split
   for(int i = 0; i < total; i++) {
      double feat[]; int lbl; double out;
      if(temp.Get(i, feat, lbl, out)) {
         if(i < trainSize)
            g_trainBuf.Add(feat, lbl, out);
         else
            g_valBuf.Add(feat, lbl, out);
      }
   }
   
   delete temp;
}

//+------------------------------------------------------------------+
//| TRAIN SINGLE MODEL                                              |
//+------------------------------------------------------------------+
double TrainModel(CNeuralNetworkV4* model, int label, string name) {
   double bestLoss = DBL_MAX;
   
   for(int epoch = 0; epoch < MaxEpochs; epoch++) {
      double totalLoss = 0;
      int    count     = 0;
      
      model.SetTrainingMode(true);
      
      for(int i = 0; i < g_trainBuf.Size(); i++) {
         double feat[]; int lbl; double out;
         if(g_trainBuf.Get(i, feat, lbl, out) && lbl == label) {
            totalLoss += model.Train(feat, 1);
            count++;
         }
      }
      
      if(count > 0) {
         totalLoss /= count;
         
         if(totalLoss < bestLoss)
            bestLoss = totalLoss;
         
         if(ShowDetails && (epoch % 20 == 0 || epoch == MaxEpochs - 1))
            Print("    ", name, " Epoch ", epoch, "/", MaxEpochs, 
                  " | Loss: ", DoubleToString(totalLoss, 6));
      }
   }
   
   model.SetTrainingMode(false);
   return bestLoss;
}

//+------------------------------------------------------------------+
//| SAVE NORMALIZATION                                              |
//+------------------------------------------------------------------+
bool SaveNormalization(const string filename) {
   int file = FileOpen(filename, FILE_WRITE | FILE_BIN);
   if(file == INVALID_HANDLE) {
      Print("ERROR: Cannot create ", filename);
      return false;
   }
   
   FileWriteInteger(file, g_numFeatures);
   
   for(int j = 0; j < g_numFeatures; j++) {
      FileWriteDouble(file, g_means[j]);
      FileWriteDouble(file, g_stds[j]);
   }
   
   long bytes = FileTell(file);  // ✅ FIXED: long instead of ulong
   FileClose(file);
   
   Print("  Normalization: ", bytes, " bytes");
   return true;
}
//+------------------------------------------------------------------+
