//+------------------------------------------------------------------+
//| GoldenGauss_EA_V5_Trainer.mq5                                   |
//| EA version of the Trainer - runs training on command             |
//| Peter 2026-04-27                                                 |
//+------------------------------------------------------------------+
#property copyright "Peter"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| INCLUDES                                                        |
//+------------------------------------------------------------------+
#include <GoldenGauss/Core/NeuralNetworkV4.mqh>
#include <GoldenGauss/Features/FeatureBuilder.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                |
//+------------------------------------------------------------------+
input group "=== Data Range ==="
input datetime StartDate      = D'2024.01.01';
input datetime EndDate        = D'2027.01.01';
input int       LookbackBars  = 5000;
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

input group "=== Trigger ==="
input bool      RunTrainingNow = false;  // Set to true to trigger training once

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                |
//+------------------------------------------------------------------+
double g_close[];
double g_open[];
double g_high[];
double g_low[];
long   g_volume[];
int    g_bars;

CNeuralNetworkV4*  g_nnBuy       = NULL;
CNeuralNetworkV4*  g_nnSell      = NULL;
CTrainingBufferV4* g_trainBuf    = NULL;
CTrainingBufferV4* g_valBuf      = NULL;
double             g_means[];
double             g_stds[];
int                g_numFeatures = NUM_FEATURES;
bool               g_trained = false;
datetime           g_last_train = 0;

//+------------------------------------------------------------------+
//| ONINIT                                                          |
//+------------------------------------------------------------------+
int OnInit() {
   Print("========================================");
   Print("  GoldenGauss EA V5 Trainer");
   Print("  Symbol: ", _Symbol, " | TF: ", EnumToString(Period()));
   Print("========================================");
   
   // Allocate arrays
   ArrayResize(g_means, g_numFeatures);
   ArrayResize(g_stds, g_numFeatures);
   
   // Pre-size price buffers
   ArrayResize(g_close, LookbackBars);
   ArrayResize(g_open, LookbackBars);
   ArrayResize(g_high, LookbackBars);
   ArrayResize(g_low, LookbackBars);
   ArrayResize(g_volume, LookbackBars);
   
   // Register feature calculators
   g_featureBuilder.RegisterCalculator(new CVolatilityFeatures());
   g_featureBuilder.RegisterCalculator(new CMomentumFeatures());
   g_featureBuilder.RegisterCalculator(new CVWAPFeatures());
   g_featureBuilder.RegisterCalculator(new CVolumeFeatures());
   g_featureBuilder.RegisterCalculator(new CStructureFeatures());
   g_featureBuilder.RegisterCalculator(new CMicroFeatures());
   g_featureBuilder.RegisterCalculator(new CTemporalFeatures());
   
   // If triggered, run training immediately
   if(RunTrainingNow) {
      TrainModels();
   }
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| ONDEINIT                                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   g_featureBuilder.FreeCalculators();
   if(g_nnBuy   != NULL) delete g_nnBuy;
   if(g_nnSell  != NULL) delete g_nnSell;
   if(g_trainBuf != NULL) delete g_trainBuf;
   if(g_valBuf   != NULL) delete g_valBuf;
}

//+------------------------------------------------------------------+
//| ONTICK                                                          |
//+------------------------------------------------------------------+
void OnTick() {
   // Check if training was triggered via chart comment "//train"
   string comment = ChartGetString(0, CHART_COMMENT);
   if(StringFind(comment, "//train") >= 0) {
      // Clear comment to prevent re-run
      ChartSetString(0, CHART_COMMENT, "");
      
      if(!g_trained || (TimeCurrent() - g_last_train) > 3600) {
         TrainModels();
      }
   }
}

//+------------------------------------------------------------------+
//| ONCHART_EVENT - custom user events                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam) {
   if(id == CHARTEVENT_CUSTOM + 1) {
      // Custom event 1 triggered - run training
      TrainModels();
   }
}

//+------------------------------------------------------------------+
//| MAIN TRAINING FUNCTION                                          |
//+------------------------------------------------------------------+
void TrainModels() {
   Print("\n========== TRAINING STARTED ==========");
   datetime start = TimeCurrent();
   
   // Step 1: Load price data
   Print("\n[1/6] Loading price data...");
   
   ArraySetAsSeries(g_close,  true);
   ArraySetAsSeries(g_open,   true);
   ArraySetAsSeries(g_high,   true);
   ArraySetAsSeries(g_low,    true);
   ArraySetAsSeries(g_volume, true);
   
   int totalBars = CopyClose(_Symbol, PERIOD_CURRENT, 0, LookbackBars, g_close);
   if(totalBars <= 0) {
      Print("ERROR: Cannot load price data!");
      return;
   }
   
   CopyOpen(_Symbol,   PERIOD_CURRENT, 0, LookbackBars, g_open);
   CopyHigh(_Symbol,   PERIOD_CURRENT, 0, LookbackBars, g_high);
   CopyLow(_Symbol,    PERIOD_CURRENT, 0, LookbackBars, g_low);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, LookbackBars, g_volume);
   
   g_bars = totalBars;
   
   datetime firstBar = iTime(_Symbol, PERIOD_CURRENT, totalBars - 1);
   datetime lastBar   = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   Print("  Loaded: ", totalBars, " bars");
   Print("  Range:  ", TimeToString(firstBar), " - ", TimeToString(lastBar));
   Print("  Date filter: ", TimeToString(StartDate), " - ", TimeToString(EndDate));
   
   // Initialize networks
   g_nnBuy  = new CNeuralNetworkV4(g_numFeatures, HiddenNeurons, 2, LearningRate);
   g_nnSell = new CNeuralNetworkV4(g_numFeatures, HiddenNeurons, 2, LearningRate);
   g_trainBuf = new CTrainingBufferV4(10000);
   g_valBuf   = new CTrainingBufferV4(5000);
   
   // Step 2: Collect training samples
   Print("\n[2/6] Collecting samples...");
   
   int buyCount  = 0;
   int sellCount = 0;
   int skipCount = 0;
   int dateSkip  = 0;
   int labelSkip = 0;
   
   for(int i = 50; i < totalBars - ForwardBars - 5; i++) {
      datetime barTime = iTime(_Symbol, PERIOD_CURRENT, i);
      
      // Check date range
      if(barTime < StartDate || barTime > EndDate) {
         dateSkip++;
         continue;
      }
      
      // Build feature vector
      double features[];
      ArrayResize(features, g_numFeatures);
      
      if(!g_featureBuilder.BuildFeatures(features, g_numFeatures, totalBars)) {
         skipCount++;
         continue;
      }
      
      // Get label
      int label = GetLabel(i, totalBars);
      
      if(label == 0) {
         labelSkip++;
         continue;
      }
      
      // Normalize features (in-place min-max to [-1,1])
      NormalizeFeatures(features);
      
      // Add to buffer
      g_trainBuf.Add(features, label, 0);
      
      if(label == 1) buyCount++;
      else           sellCount++;
   }
   
   int totalSamples = buyCount + sellCount;
   Print("  Total samples: ", totalSamples, " (BUY: ", buyCount, " | SELL: ", sellCount, ")");
   Print("  Skipped: date_filter=", dateSkip, " label=0/", labelSkip, " build_failed=", skipCount);
   
   if(totalSamples < MinSamples) {
      Print("\nERROR: Not enough samples! Need ", MinSamples, ", got ", totalSamples);
      Print("SOLUTIONS:");
      Print("  1. Expand date range (StartDate/EndDate)");
      Print("  2. Increase LookbackBars (currently ", LookbackBars, ")");
      Print("  3. Relax GetLabel() thresholds");
      return;
   }
   
   // Step 3: Compute normalization
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
   
   // Step 6: Save
   Print("\n[6/6] Saving files...");
   
   string folder = "GoldenGauss_Data_" + _Symbol + "//Models//";
   
   bool ok1 = g_nnBuy.Save(folder + BuyModelFile);
   bool ok2 = g_nnSell.Save(folder + SellModelFile);
   
   // Save normalization
   if(ok1 && ok2) {
      SaveNormalization(folder + "norm_params.dat");
   }
   
   datetime elapsed = TimeCurrent() - start;
   Print("\n========================================");
   Print("  TRAINING COMPLETE!");
   Print("  BUY Loss:  ", DoubleToString(buyLoss, 6));
   Print("  SELL Loss: ", DoubleToString(sellLoss, 6));
   Print("  Duration: ", elapsed, " sec");
   Print("  Files: ", folder);
   Print("========================================");
   
   g_trained = true;
   g_last_train = TimeCurrent();
   
   // Cleanup buffers
   delete g_trainBuf;
   delete g_valBuf;
   delete g_nnBuy;
   delete g_nnSell;
   g_trainBuf = NULL;
   g_valBuf = NULL;
   g_nnBuy = NULL;
   g_nnSell = NULL;
}

//+------------------------------------------------------------------+
//| GET LABEL - uses global g_close/g_high/g_low                    |
//+------------------------------------------------------------------+
int GetLabel(int idx, int bars) {
   // Calculate RSI (7 period)
   double gain = 0, loss = 0;
   for(int j = 1; j <= 7 && (idx + j + 1) < bars; j++) {
      double diff = g_close[idx + j] - g_close[idx + j + 1];
      if(diff > 0) gain += diff;
      else         loss += MathAbs(diff);
   }
   
   double rsi = 50;
   if(loss > 0.00001) 
      rsi = 100.0 - (100.0 / (1.0 + gain / loss));
   
   // Calculate forward return
   double fwdRet = 0;
   if((idx + ForwardBars) < bars)
      fwdRet = (g_close[idx + ForwardBars] - g_close[idx]) / g_close[idx] * 100.0;
   
   // More relaxed thresholds for better sample coverage
   double rsiBuy  = 50;   // was 45
   double rsiSell = 50;   // was 55
   double minRet  = 0.005; // 0.5% forward return (was 0.01)
   
   // BUY signal
   if(rsi < rsiBuy && fwdRet > minRet)
      return 1;
   
   // SELL signal
   if(rsi > rsiSell && fwdRet < -minRet)
      return 2;
   
   return 0;
}

//+------------------------------------------------------------------+
//| NORMALIZE FEATURES - in-place min-max to [-1, 1]                 |
//+------------------------------------------------------------------+
void NormalizeFeatures(double &features[]) {
   double minVal = features[0];
   double maxVal = features[0];
   
   for(int j = 1; j < ArraySize(features); j++) {
      if(features[j] < minVal) minVal = features[j];
      if(features[j] > maxVal) maxVal = features[j];
   }
   
   double range = (maxVal - minVal);
   if(range < 0.00001) range = 1.0;
   
   for(int j = 0; j < ArraySize(features); j++) {
      features[j] = (features[j] - minVal) / range * 2.0 - 1.0;
   }
}

//+------------------------------------------------------------------+
//| COMPUTE NORMALIZATION                                            |
//+------------------------------------------------------------------+
void ComputeNormalization() {
   if(g_trainBuf.Size() == 0) return;
   
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
            // Normalize label: {1=BUY, 2=SELL} -> {0, 1} for NN output
            int nnLabel = (label == 2) ? 1 : 0;
            totalLoss += model.Train(feat, nnLabel);
            count++;
         }
      }
      
      if(count > 0) {
         totalLoss /= count;
         
         if(totalLoss < bestLoss)
            bestLoss = totalLoss;
         
         if(epoch % 20 == 0 || epoch == MaxEpochs - 1)
            Print("    ", name, " Epoch ", epoch, "/", MaxEpochs, 
                  " | Loss: ", DoubleToString(totalLoss, 6), " | Samples: ", count);
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
   
   ulong bytes = FileTell(file);
   FileClose(file);
   
   Print("  Normalization: ", bytes, " bytes");
   return true;
}
//+------------------------------------------------------------------+