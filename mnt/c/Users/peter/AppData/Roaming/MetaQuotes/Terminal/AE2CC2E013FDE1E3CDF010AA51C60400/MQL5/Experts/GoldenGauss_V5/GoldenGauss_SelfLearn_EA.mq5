//+------------------------------------------------------------------+
//| GoldenGauss_SelfLearn_EA.mq5 - TRUE SELF-LEARNING VERSION        |
//| Peter - Version 5.00                                             |
//|                                                                  |
//| REAL Self-Learning Features:                                     |
//|   - Stores trade outcomes for model retraining                  |
//|   - Retrains NN periodically based on actual results            |
//|   - Adapts to changing market conditions                        |
//|   - GBrain integration for confirmation                         |
//|   - Full risk management with drawdown protection               |
//+------------------------------------------------------------------+
#property copyright "Peter"
#property version   "5.00"
#property description "GoldenGauss EA V5 - TRUE Self-Learning"
#property description "Retrains NN based on actual trade outcomes"

//+------------------------------------------------------------------+
//| INCLUDES                                                         |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
#include "Core/Types.mqh"
#include "Core/NeuralNetworkV4.mqh"
#include "Features/FeatureBuilder.mqh"
#include "Trading/TradeManager.mqh"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                 |
//+------------------------------------------------------------------+
input group "=== Trading ==="
input double          LotSize = 0.01;
input int             MaxSpreadPips = 30;
input int             MaxPositions = 1;

input group "=== Risk Management ==="
input double          MaxDailyLossPercent = 3.0;    
input double          MaxDrawdownPercent = 10.0;    
input bool            UseDynamicLotSizing = true;   
input double          RiskPerTrade = 1.0;  // Risk per Trade %         

input group "=== Time Filter ==="
input int             StartHour = 13;
input int             EndHour = 18;
input bool            UseTimeFilter = true;

input group "=== ATR Stop Management ==="
input int             ATRPeriod = 14;
input double          DefaultSLMultiplier = 1.5;
input double          DefaultTPMultiplier = 2.5;
input double          DefaultTrailingMultiplier = 0.75;

input group "=== Model V5 ==="
input string          BuyModelPath = "BULLISH_V5.nn";
input string          SellModelPath = "BEARISH_V5.nn";
input double          ProbabilityThreshold = 0.65;
input double          StrongThreshold = 0.80;
input bool            UseNormalization = true;

input group "=== TRUE Self-Learning ==="
input bool            EnableSelfLearning = true;
input int             RetrainEveryNTrades = 50;     
input int             MinSamplesForRetrain = 30;    
input double          RetrainLearningRate = 0.001;  
input int             RetrainEpochs = 20;           

input group "=== GBrain Integration ==="
input bool            UseGBrain = true;
input double          MinGBrainConfirm = 0.3;

input group "=== Data Management ==="
input string          DataDirectory = "GoldenGauss_Data";
input long            MagicNumber = 202604271823;

//+------------------------------------------------------------------+
//| GLOBAL OBJECTS                                                   |
//+------------------------------------------------------------------+
CTradeManager       g_tradeManager;
CNeuralNetworkV4*   g_nnBuy = NULL;
CNeuralNetworkV4*   g_nnSell = NULL;

// Price buffers
double            g_close[];
double            g_open[];
double            g_high[];
double            g_low[];
long              g_volume[];
int               g_bars;

// ATR handle
int               g_atr_handle = INVALID_HANDLE;

// GBrain handle
int               g_gbrain_handle = INVALID_HANDLE;

// Normalization
double            g_feature_mean[NUM_FEATURES];
double            g_feature_std[NUM_FEATURES];
bool              g_normalization_loaded = false;

// State
datetime          g_last_bar = 0;
bool              g_models_loaded = false;
int               g_total_trades = 0;
int               g_trades_since_retrain = 0;

// ✅ NEW: Trade history for retraining
struct SRetrainSample {
   double features[NUM_FEATURES];
   int    label;          // 1=win, 0=loss
   double profit_pips;
   datetime timestamp;
};
SRetrainSample g_retrain_samples[];
int            g_retrain_count = 0;

// Risk management
double          g_daily_start_balance = 0;
double          g_max_balance = 0;
bool            g_trading_halted = false;
datetime        g_last_halt_check = 0;

//+------------------------------------------------------------------+
//| LOAD NORMALIZATION PARAMETERS                                    |
//+------------------------------------------------------------------+
bool LoadNormalizationParams(const string filename) {
   string clean = CNeuralNetworkV4::PreparePath(filename);
   int file = FileOpen(clean, FILE_READ | FILE_BIN);
   if(file == INVALID_HANDLE) return false;
   
   int numFeatures = FileReadInteger(file);
   if(numFeatures != NUM_FEATURES) { FileClose(file); return false; }
   
   for(int j = 0; j < numFeatures; j++)
      g_feature_mean[j] = FileReadDouble(file);
   for(int j = 0; j < numFeatures; j++)
      g_feature_std[j] = FileReadDouble(file);
   
   FileClose(file);
   return true;
}

//+------------------------------------------------------------------+
//| NORMALIZE FEATURES                                               |
//+------------------------------------------------------------------+
void NormalizeFeatures(double &features[]) {
   if(!g_normalization_loaded) return;
   for(int j = 0; j < ArraySize(features); j++) {
      features[j] = (features[j] - g_feature_mean[j]) / g_feature_std[j];
      features[j] = MathMax(-5.0, MathMin(5.0, features[j]));
   }
}

//+------------------------------------------------------------------+
//| CHECK RISK LIMITS                                                |
//+------------------------------------------------------------------+
bool CheckRiskLimits() {
   if(g_trading_halted) return false;
   
   datetime now = TimeCurrent();
   if(now - g_last_halt_check < 60) return true;  // Check every minute
   g_last_halt_check = now;
   
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Daily loss check
   if(g_daily_start_balance == 0) g_daily_start_balance = balance;
   
   double daily_loss = (g_daily_start_balance - balance) / g_daily_start_balance * 100.0;
   if(daily_loss > MaxDailyLossPercent) {
      Print("[EA] DAILY LOSS LIMIT REACHED: ", DoubleToString(daily_loss, 2), "%");
      g_trading_halted = true;
      return false;
   }
   
   // Drawdown check
   if(g_max_balance == 0) g_max_balance = balance;
   if(balance > g_max_balance) g_max_balance = balance;
   
   double drawdown = (g_max_balance - equity) / g_max_balance * 100.0;
   if(drawdown > MaxDrawdownPercent) {
      Print("[EA] MAX DRAWDOWN REACHED: ", DoubleToString(drawdown, 2), "%");
      g_trading_halted = true;
      return false;
   }
   
   // Reset daily balance at new day
   MqlDateTime dt, last_dt;
   TimeToStruct(now, dt);
   TimeToStruct(g_last_halt_check, last_dt);
   if(dt.day != last_dt.day) {
      g_daily_start_balance = balance;
      g_trading_halted = false;
      Print("[EA] New trading day - limits reset");
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| CALCULATE DYNAMIC LOT SIZE                                       |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_pips) {
   if(!UseDynamicLotSizing) return LotSize;
   
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_amount = equity * RiskPerTrade / 100.0;
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   if(sl_pips <= 0 || tick_value <= 0 || tick_size <= 0) return LotSize;
   
   double lot = risk_amount / (sl_pips * tick_value / tick_size);
   
   // Normalize to broker lot steps
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lot = MathFloor(lot / lot_step) * lot_step;
   lot = MathMax(min_lot, MathMin(max_lot, lot));
   
   return lot;
}

//+------------------------------------------------------------------+
//| STORE TRADE FOR RETRAINING                                       |
//+------------------------------------------------------------------+
void StoreTradeForRetraining(double &features[], double profit_pips) {
   if(!EnableSelfLearning) return;
   if(g_retrain_count >= 1000) return;  // Max buffer size
   
   int label = (profit_pips > 0) ? 1 : 0;
   
   g_retrain_samples[g_retrain_count].label = label;
   g_retrain_samples[g_retrain_count].profit_pips = profit_pips;
   g_retrain_samples[g_retrain_count].timestamp = TimeCurrent();
   for(int j = 0; j < NUM_FEATURES; j++)
      g_retrain_samples[g_retrain_count].features[j] = features[j];
   
   g_retrain_count++;
   g_trades_since_retrain++;
   
   Print("[EA] Stored trade for retraining. Total: ", g_retrain_count, 
         " | PnL: ", DoubleToString(profit_pips, 1), " pips");
}

//+------------------------------------------------------------------+
//| RETRAIN MODELS BASED ON ACTUAL OUTCOMES                          |
//+------------------------------------------------------------------+
void RetrainModels() {
   if(!EnableSelfLearning) return;
   if(g_retrain_count < MinSamplesForRetrain) return;
   if(g_nnBuy == NULL || g_nnSell == NULL) return;
   
   Print("=== Starting Model Retraining ===");
   Print("Samples: ", g_retrain_count);
   
   // Separate winning and losing trades
   double win_features[][NUM_FEATURES];
   double loss_features[][NUM_FEATURES];
   ArrayResize(win_features, g_retrain_count);
   ArrayResize(loss_features, g_retrain_count);
   int win_count = 0, loss_count = 0;
   
   for(int i = 0; i < g_retrain_count; i++) {
      if(g_retrain_samples[i].label == 1) {
         for(int j = 0; j < NUM_FEATURES; j++)
            win_features[win_count][j] = g_retrain_samples[i].features[j];
         win_count++;
      } else {
         for(int j = 0; j < NUM_FEATURES; j++)
            loss_features[loss_count][j] = g_retrain_samples[i].features[j];
         loss_count++;
      }
   }
   
   Print("Winning trades: ", win_count, " | Losing trades: ", loss_count);
   
   // Retrain BUY model on winning trades
   if(win_count > 5) {
      g_nnBuy.SetTrainingMode(true);
      for(int epoch = 0; epoch < RetrainEpochs; epoch++) {
         for(int i = 0; i < win_count; i++) {
            double feat[];
            ArrayResize(feat, NUM_FEATURES);
            for(int j = 0; j < NUM_FEATURES; j++) feat[j] = win_features[i][j];
            g_nnBuy.Train(feat, 1);
         }
      }
      Print("BUY model retrained on ", win_count, " winning trades");
   }
   
   // Retrain SELL model on losing trades (learn what NOT to do)
   if(loss_count > 5) {
      g_nnSell.SetTrainingMode(true);
      for(int epoch = 0; epoch < RetrainEpochs; epoch++) {
         for(int i = 0; i < loss_count; i++) {
            double feat[];
            ArrayResize(feat, NUM_FEATURES);
            for(int j = 0; j < NUM_FEATURES; j++) feat[j] = loss_features[i][j];
            g_nnSell.Train(feat, 0);  // Learn to avoid these patterns
         }
      }
      Print("SELL model retrained on ", loss_count, " losing trades");
   }
   
   // Save updated models
   string sym = _Symbol;
   string dataPath = DataDirectory + "_" + sym + "//Models//";
   
   g_nnBuy.Save(dataPath + BuyModelPath);
   g_nnSell.Save(dataPath + SellModelPath);
   
   Print("=== Retraining Complete ===");
   
   // Reset counter
   g_trades_since_retrain = 0;
}

//+------------------------------------------------------------------+
//| ONINIT                                                           |
//+------------------------------------------------------------------+
int OnInit() {
   Print("==================================================");
   Print("  GoldenGauss EA V5.00 - TRUE SELF-LEARNING");
   Print("==================================================");
   
   // Pre-allocate arrays
   ArrayResize(g_close, 500); ArrayResize(g_open, 500);
   ArrayResize(g_high, 500); ArrayResize(g_low, 500);
   ArrayResize(g_volume, 500);
   ArraySetAsSeries(g_close, true); ArraySetAsSeries(g_open, true);
   ArraySetAsSeries(g_high, true); ArraySetAsSeries(g_low, true);
   ArraySetAsSeries(g_volume, true);
   
   // Initialize normalization
   ArrayInitialize(g_feature_mean, 0.0); ArrayInitialize(g_feature_std, 1.0);
   
   // Initialize ATR
   g_atr_handle = iATR(_Symbol, PERIOD_M1, ATRPeriod);
   if(g_atr_handle == INVALID_HANDLE) return INIT_FAILED;
   
   // Register feature calculators (without CrossAsset)
   g_featureBuilder.RegisterCalculator(new CVolatilityFeatures());
   g_featureBuilder.RegisterCalculator(new CMomentumFeatures());
   g_featureBuilder.RegisterCalculator(new CVWAPFeatures());
   g_featureBuilder.RegisterCalculator(new CVolumeFeatures());
   g_featureBuilder.RegisterCalculator(new CStructureFeatures());
   g_featureBuilder.RegisterCalculator(new CMicroFeatures());
   g_featureBuilder.RegisterCalculator(new CTemporalFeatures());
   
   // Configure trade manager
   g_tradeManager.SetMagic(MagicNumber);
   g_tradeManager.SetLotSize(LotSize);
   g_tradeManager.SetMaxPositions(MaxPositions);
   
   // Load normalization
   string sym = _Symbol;
   string dataPath = DataDirectory + "_" + sym;
   if(UseNormalization) {
      g_normalization_loaded = LoadNormalizationParams(dataPath + "//Models//norm_params.dat");
   }
   
   // Load models
   g_nnBuy = new CNeuralNetworkV4(NUM_FEATURES, 64, 2, 0.001);
   g_nnSell = new CNeuralNetworkV4(NUM_FEATURES, 64, 2, 0.001);
   
   string modelsPath = dataPath + "//Models//";
   bool buy_ok = g_nnBuy.Load(modelsPath + BuyModelPath);
   bool sell_ok = g_nnSell.Load(modelsPath + SellModelPath);
   g_models_loaded = buy_ok && sell_ok;
   
   // Initialize GBrain
   if(UseGBrain) {
      g_gbrain_handle = iCustom(_Symbol, PERIOD_M1, "GBrain", 10, 80, 200, 0.002, 20, 0.3, 0);
   }
   
   Print("[EA] Models loaded: ", g_models_loaded);
   Print("[EA] Self-Learning: ", EnableSelfLearning);
   Print("[EA] Retrain every ", RetrainEveryNTrades, " trades");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
   if(g_gbrain_handle != INVALID_HANDLE) IndicatorRelease(g_gbrain_handle);
   if(g_nnBuy != NULL) delete g_nnBuy;
   if(g_nnSell != NULL) delete g_nnSell;
   g_featureBuilder.FreeCalculators();
}

//+------------------------------------------------------------------+
void OnTick() {
   // Check risk limits first
   if(!CheckRiskLimits()) return;
   
   datetime current_bar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar == g_last_bar) return;
   g_last_bar = current_bar;
   
   // Update price data
   g_bars = CopyClose(_Symbol, PERIOD_CURRENT, 0, 500, g_close);
   if(g_bars <= 0) return;
   CopyOpen(_Symbol, PERIOD_CURRENT, 0, 500, g_open);
   CopyHigh(_Symbol, PERIOD_CURRENT, 0, 500, g_high);
   CopyLow(_Symbol, PERIOD_CURRENT, 0, 500, g_low);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 500, g_volume);
   
   // Check trading hours
   if(UseTimeFilter) {
      MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
      if(dt.hour < StartHour || dt.hour >= EndHour) return;
   }
   
   // Check spread
   if((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > MaxSpreadPips * 10) return;
   if(g_tradeManager.GetOpenPositions() >= MaxPositions) return;
   
   // Build features
   double features[];
   ArrayResize(features, NUM_FEATURES);
   if(!g_featureBuilder.BuildFeatures(features, NUM_FEATURES, g_bars)) return;
   NormalizeFeatures(features);
   
   // Get prediction
   double buy_probs[], sell_probs[];
   g_nnBuy.Predict(features, buy_probs);
   g_nnSell.Predict(features, sell_probs);
   
   double buy_prob = (ArraySize(buy_probs) >= 2) ? buy_probs[1] : 0.0;
   double sell_prob = (ArraySize(sell_probs) >= 2) ? sell_probs[1] : 0.0;
   
   // Check GBrain confirmation
   double gbrain_signal = 0.0;
   if(UseGBrain && g_gbrain_handle != INVALID_HANDLE) {
      double buf[];
      if(CopyBuffer(g_gbrain_handle, 3, 0, 1, buf) > 0)
         gbrain_signal = buf[0];
   }
   
   // Generate signal
   ENUM_TRADE_SIGNAL signal = SIGNAL_NONE;
   if(buy_prob >= ProbabilityThreshold && gbrain_signal >= MinGBrainConfirm)
      signal = (buy_prob >= StrongThreshold) ? SIGNAL_BUY_STRONG : SIGNAL_BUY;
   else if(sell_prob >= ProbabilityThreshold && gbrain_signal <= -MinGBrainConfirm)
      signal = (sell_prob >= StrongThreshold) ? SIGNAL_SELL_STRONG : SIGNAL_SELL;
   
   if(signal == SIGNAL_NONE) return;
   
   // Calculate stops
   double atr = 0;
   double buf[]; ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_atr_handle, 0, 0, 1, buf) > 0) atr = buf[0];
   
   double sl_pips = (atr / _Point) * DefaultSLMultiplier;
   double tp_pips = (atr / _Point) * DefaultTPMultiplier;
   double trailing_pips = (atr / _Point) * DefaultTrailingMultiplier;
   
   // Calculate lot size
   double lot = CalculateLotSize(sl_pips);
   g_tradeManager.SetLotSize(lot);
   
   // Open trade
   ulong ticket = 0;
   if(signal == SIGNAL_BUY || signal == SIGNAL_BUY_STRONG) {
      if(g_tradeManager.OpenBuyATR(buy_prob, sl_pips, tp_pips, ticket)) {
         Print("[EA] BUY opened | Ticket: ", ticket, " | Lot: ", lot);
      }
   } else if(signal == SIGNAL_SELL || signal == SIGNAL_SELL_STRONG) {
      if(g_tradeManager.OpenSellATR(sell_prob, sl_pips, tp_pips, ticket)) {
         Print("[EA] SELL opened | Ticket: ", ticket, " | Lot: ", lot);
      }
   }
   
   // Manage trailing stops
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
         ulong t = (ulong)PositionGetInteger(POSITION_TICKET);
         g_tradeManager.ManageTrailingStop(t, trailing_pips);
      }
   }
}

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result) {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
      ulong deal_ticket = trans.deal;
      if(!HistoryDealSelect(deal_ticket)) return;
      if(HistoryDealGetString(deal_ticket, DEAL_SYMBOL) != _Symbol) return;
      if(HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != MagicNumber) return;
      if(HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;
      
      // Get profit
      double profit_pips = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT) / _Point / LotSize;
      
      // Get features from last bar (approximation)
      double features[];
      ArrayResize(features, NUM_FEATURES);
      if(g_featureBuilder.BuildFeatures(features, NUM_FEATURES, g_bars)) {
         NormalizeFeatures(features);
         StoreTradeForRetraining(features, profit_pips);
      }
      
      g_total_trades++;
      Print("[EA] Trade closed | PnL: ", DoubleToString(profit_pips, 1), " pips | Total: ", g_total_trades);
      
      // Trigger retraining if enough trades
      if(g_trades_since_retrain >= RetrainEveryNTrades) {
         RetrainModels();
      }
   }
}
//+------------------------------------------------------------------+
