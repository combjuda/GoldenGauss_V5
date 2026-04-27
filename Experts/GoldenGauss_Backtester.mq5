//+------------------------------------------------------------------+
//| GoldenGauss_Backtester.mq5                                       |
//| Professional Backtesting Framework for GoldenGauss_V5            |
//| Peter - Version 1.02                                             |
//+------------------------------------------------------------------+
#property copyright "Peter"
#property version   "1.02"
#property description "GoldenGauss V5 - Professional Backtester"
#property tester_set "Presets/XAUUSD_BALANCED_M5.set"

#include <Trade/Trade.mqh>
#include <GoldenGauss/Core/Types.mqh>
#include <GoldenGauss/Core/NeuralNetworkV4.mqh>
#include <GoldenGauss/Features/FeatureBuilder.mqh>
#include <GoldenGauss/Trading/TradeManager.mqh>
#include <GoldenGauss/Backtest/BacktestMetrics.mqh>
#include <GoldenGauss/Backtest/BacktestLogger.mqh>
#include <GoldenGauss/Backtest/BacktestReport.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                 |
//+------------------------------------------------------------------+
input group "=== Backtest Configuration ==="
input double          InitialDeposit = 10000.0;
input double          SlippagePips = 1.0;
input double          CommissionPerLot = 3.5;
input bool            EnableLogging = true;
input string          LogFileName = "backtest_results.csv";

input group "=== Model Files ==="
input string          BuyModelPath = "BULLISH_V5.nn";
input string          SellModelPath = "BEARISH_V5.nn";
input bool            UseNormalization = true;

input group "=== Trading Parameters ==="
input double          LotSize = 0.01;
input int             MaxSpreadPips = 30;
input int             MaxPositions = 1;
input int             StartHour = 13;
input int             EndHour = 18;
input bool            UseTimeFilter = true;

input group "=== Risk Management ==="
input double          DefaultSLMultiplier = 1.5;
input double          DefaultTPMultiplier = 2.5;
input double          DefaultTrailingMultiplier = 0.75;
input int             ATRPeriod = 14;

input group "=== GBrain Integration ==="
input bool            UseGBrain = false;
input double          MinGBrainConfirm = 0.3;

input group "=== Data Directory ==="
input string          DataDirectory = "GoldenGauss_Data";

//+------------------------------------------------------------------+
//| GLOBAL OBJECTS                                                   |
//+------------------------------------------------------------------+
CTradeManager       g_tradeManager;
CNeuralNetworkV4*   g_nnBuy = NULL;
CNeuralNetworkV4*   g_nnSell = NULL;
CBacktestMetrics    g_metrics;
CBacktestLogger     g_logger;
CBacktestReport     g_report;

// Price buffers (pre-allocated to avoid reallocation on every tick)
double            g_close[];
double            g_open[];
double            g_high[];
double            g_low[];
long              g_volume[];
int               g_bars = 0;

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
double            g_peak_equity = 0;
double            g_max_drawdown = 0;

//+------------------------------------------------------------------+
//| HELPER: Build model file path for FileOpen                       |
//| FileOpen() searches in MQL5/Files/ automatically                |
//+------------------------------------------------------------------+
string ModelPath(string filename) {
   // FileOpen with relative path searches MQL5/Files/ subdirectory
   string fullPath = DataDirectory + "_" + _Symbol + "//Models//" + filename;
   return fullPath;
}

//+------------------------------------------------------------------+
//| LOAD NORMALIZATION PARAMETERS                                    |
//+------------------------------------------------------------------+
bool LoadNormalizationParams() {
   string filename = ModelPath("norm_params.dat");
   int file = FileOpen(filename, FILE_READ | FILE_BIN);
   if(file == INVALID_HANDLE) {
      Print("[Backtest] Cannot open norm file: ", filename);
      return false;
   }
   
   int numFeatures = FileReadInteger(file);
   if(numFeatures != NUM_FEATURES) {
      Print("[Backtest] Feature count mismatch! Expected ", NUM_FEATURES, ", got ", numFeatures);
      FileClose(file);
      return false;
   }
   
   for(int j = 0; j < numFeatures; j++)
      g_feature_mean[j] = FileReadDouble(file);
   for(int j = 0; j < numFeatures; j++)
      g_feature_std[j] = FileReadDouble(file);
   
   FileClose(file);
   Print("[Backtest] Normalization loaded: ", numFeatures, " features");
   return true;
}

//+------------------------------------------------------------------+
//| NORMALIZE FEATURES (z-score)                                     |
//+------------------------------------------------------------------+
void NormalizeFeatures(double &features[]) {
   if(!g_normalization_loaded) return;
   for(int j = 0; j < ArraySize(features) && j < NUM_FEATURES; j++) {
      if(g_feature_std[j] < 0.00001) g_feature_std[j] = 1.0;
      features[j] = (features[j] - g_feature_mean[j]) / g_feature_std[j];
      features[j] = MathMax(-5.0, MathMin(5.0, features[j]));
   }
}

//+------------------------------------------------------------------+
//| ONINIT                                                           |
//+------------------------------------------------------------------+
int OnInit() {
   Print("==================================================");
   Print("  GoldenGauss V5 - Professional Backtester");
   Print("  Symbol: ", _Symbol, " | TF: ", EnumToString(Period()));
   Print("==================================================");
   Print("Initial Deposit: $", InitialDeposit);
   Print("Slippage: ", SlippagePips, " pips");
   Print("Commission: $", CommissionPerLot, " per lot");
   
   // Initialize logger
   if(EnableLogging)
      g_logger.Initialize(LogFileName);
   
   // Initialize metrics
   g_metrics.Initialize(InitialDeposit);
   
   // Pre-allocate arrays with adequate size for lookback
   int lookbackSize = 5000;
   ArrayResize(g_close, lookbackSize);
   ArrayResize(g_open, lookbackSize);
   ArrayResize(g_high, lookbackSize);
   ArrayResize(g_low, lookbackSize);
   ArrayResize(g_volume, lookbackSize);
   ArraySetAsSeries(g_close, true);
   ArraySetAsSeries(g_open, true);
   ArraySetAsSeries(g_high, true);
   ArraySetAsSeries(g_low, true);
   ArraySetAsSeries(g_volume, true);
   
   // Initialize normalization arrays
   ArrayInitialize(g_feature_mean, 0.0);
   ArrayInitialize(g_feature_std, 1.0);
   
   // Initialize ATR
   g_atr_handle = iATR(_Symbol, Period(), ATRPeriod);
   if(g_atr_handle == INVALID_HANDLE) {
      Print("[Backtest] ERROR: Cannot create ATR handle!");
      return INIT_FAILED;
   }
   
   // Register feature calculators
   g_featureBuilder.RegisterCalculator(new CVolatilityFeatures());
   g_featureBuilder.RegisterCalculator(new CMomentumFeatures());
   g_featureBuilder.RegisterCalculator(new CVWAPFeatures());
   g_featureBuilder.RegisterCalculator(new CVolumeFeatures());
   g_featureBuilder.RegisterCalculator(new CStructureFeatures());
   g_featureBuilder.RegisterCalculator(new CMicroFeatures());
   g_featureBuilder.RegisterCalculator(new CTemporalFeatures());
   
   // Configure trade manager
   g_tradeManager.SetMagic(20260426);
   g_tradeManager.SetLotSize(LotSize);
   g_tradeManager.SetMaxPositions(MaxPositions);
   
   // Initialize neural networks
   g_nnBuy = new CNeuralNetworkV4(NUM_FEATURES, 64, 2, 0.001);
   g_nnSell = new CNeuralNetworkV4(NUM_FEATURES, 64, 2, 0.001);
   
   // Build model paths
   string buyPath = ModelPath(BuyModelPath);
   string sellPath = ModelPath(SellModelPath);
   
   Print("[Backtest] Loading BUY model:  ", buyPath);
   Print("[Backtest] Loading SELL model:  ", sellPath);
   
   bool buy_ok = g_nnBuy.Load(buyPath);
   bool sell_ok = g_nnSell.Load(sellPath);
   g_models_loaded = buy_ok && sell_ok;
   
   if(g_models_loaded) {
      Print("[Backtest] Models loaded successfully");
   } else {
      if(!buy_ok) Print("[Backtest] BUY model failed to load");
      if(!sell_ok) Print("[Backtest] SELL model failed to load");
      Print("[Backtest] Will use heuristic mode");
   }
   
   // Load normalization if models loaded
   if(UseNormalization && g_models_loaded) {
      g_normalization_loaded = LoadNormalizationParams();
      if(!g_normalization_loaded) {
         Print("[Backtest] WARNING: Normalization params not found");
         Print("[Backtest] Features will be unnormalized");
      }
   }
   
   // Initialize GBrain (on current chart period)
   if(UseGBrain) {
      g_gbrain_handle = iCustom(_Symbol, Period(), "GBrain", 10, 80, 200, 0.002, 20, 0.3, 0);
      if(g_gbrain_handle != INVALID_HANDLE)
         Print("[Backtest] GBrain indicator loaded");
      else
         Print("[Backtest] WARNING: GBrain indicator not found!");
   }
   
   // Initialize report
   g_report.SetSymbol(_Symbol);
   g_report.SetPeriod(Period());
   g_report.SetInitialDeposit(InitialDeposit);
   
   g_peak_equity = InitialDeposit;
   
   Print("[Backtest] Initialization complete");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| ONDEINIT                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   // Free feature calculators first
   g_featureBuilder.FreeCalculators();
   
   // Release indicators
   if(g_atr_handle != INVALID_HANDLE) {
      IndicatorRelease(g_atr_handle);
      g_atr_handle = INVALID_HANDLE;
   }
   if(g_gbrain_handle != INVALID_HANDLE) {
      IndicatorRelease(g_gbrain_handle);
      g_gbrain_handle = INVALID_HANDLE;
   }
   
   // Delete neural networks
   if(g_nnBuy != NULL) { delete g_nnBuy; g_nnBuy = NULL; }
   if(g_nnSell != NULL) { delete g_nnSell; g_nnSell = NULL; }
   
   // Generate final report
   Print("==================================================");
   Print("  BACKTEST COMPLETE");
   Print("==================================================");
   
   g_metrics.CalculateFinalMetrics();
   g_metrics.PrintSummary();
   
   // Save HTML report to Files directory
   string reportPath = DataDirectory + "_" + _Symbol + "//backtest_report.html";
   g_report.Generate(g_metrics, reportPath);
   
   if(EnableLogging)
      g_logger.Close();
   
   Print("Report: ", reportPath);
}

//+------------------------------------------------------------------+
//| ONTICK                                                           |
//+------------------------------------------------------------------+
void OnTick() {
   datetime current_bar = iTime(_Symbol, Period(), 0);
   if(current_bar == g_last_bar) return;
   g_last_bar = current_bar;
   
   // Load price data (use Period() for consistency with OnInit)
   g_bars = CopyClose(_Symbol, Period(), 0, 5000, g_close);
   if(g_bars <= 0) return;
   
   CopyOpen(_Symbol, Period(), 0, 5000, g_open);
   CopyHigh(_Symbol, Period(), 0, 5000, g_high);
   CopyLow(_Symbol, Period(), 0, 5000, g_low);
   CopyTickVolume(_Symbol, Period(), 0, 5000, g_volume);
   
   // Time filter
   if(UseTimeFilter) {
      MqlDateTime dt;
      TimeToStruct(current_bar, dt);
      if(dt.hour < StartHour || dt.hour >= EndHour) return;
   }
   
   // Spread check
   if((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > MaxSpreadPips * 10) return;
   
   // Max positions check
   if(g_tradeManager.GetOpenPositions() >= MaxPositions) return;
   
   // Build features
   double features[];
   ArrayResize(features, NUM_FEATURES);
   if(!g_featureBuilder.BuildFeatures(features, NUM_FEATURES, g_bars)) {
      Print("[Backtest] BuildFeatures failed at ", TimeToString(current_bar));
      return;
   }
   
   // Normalize
   NormalizeFeatures(features);
   
   // Get prediction
   double buy_prob = 0.0, sell_prob = 0.0;
   
   if(g_models_loaded) {
      double buy_probs[], sell_probs[];
      g_nnBuy.Predict(features, buy_probs);
      g_nnSell.Predict(features, sell_probs);
      
      buy_prob = (ArraySize(buy_probs) >= 2) ? buy_probs[1] : 0.0;
      sell_prob = (ArraySize(sell_probs) >= 2) ? sell_probs[1] : 0.0;
      
      // Debug: print signal every 100 bars
      static int s_debug_bar = 0;
      if(s_debug_bar % 100 == 0) {
         Print("[Signal] BUY=", DoubleToString(buy_prob, 3),
               " SELL=", DoubleToString(sell_prob, 3),
               " | gbrain=", DoubleToString(g_gbrain_signal()));
      }
      s_debug_bar++;
   } else {
      // Heuristic fallback
      buy_prob = HeuristicBuySignal();
      sell_prob = HeuristicSellSignal();
   }
   
   // GBrain signal
   double gbrain_sig = 0.0;
   if(UseGBrain && g_gbrain_handle != INVALID_HANDLE) {
      double buf[];
      if(CopyBuffer(g_gbrain_handle, 0, 0, 1, buf) > 0) {
         gbrain_sig = buf[0];
      }
   }
   
   // Signal generation
   ENUM_TRADE_SIGNAL signal = SIGNAL_NONE;
   double probability = 0.0;
   
   // Require minimum probability and GBrain confirmation
   if(buy_prob >= 0.55 && gbrain_sig >= MinGBrainConfirm) {
      signal = (buy_prob >= 0.75) ? SIGNAL_BUY_STRONG : SIGNAL_BUY;
      probability = buy_prob;
   } else if(sell_prob >= 0.55 && gbrain_sig <= -MinGBrainConfirm) {
      signal = (sell_prob >= 0.75) ? SIGNAL_SELL_STRONG : SIGNAL_SELL;
      probability = sell_prob;
   }
   
   if(signal == SIGNAL_NONE) return;
   
   // Calculate ATR stops
   double atr = 0;
   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(g_atr_handle, 0, 0, 1, atrBuf) > 0)
      atr = atrBuf[0];
   
   if(atr <= 0) {
      Print("[Backtest] ATR is zero or invalid!");
      return;
   }
   
   double sl_pips = (atr / _Point) * DefaultSLMultiplier;
   double tp_pips = (atr / _Point) * DefaultTPMultiplier;
   double trailing_pips = (atr / _Point) * DefaultTrailingMultiplier;
   
   // Enforce minimum stop distance
   double minStop = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(sl_pips * _Point < minStop) sl_pips = minStop / _Point;
   if(tp_pips * _Point < minStop) tp_pips = minStop / _Point;
   
   // Open trade
   ulong ticket = 0;
   
   if(signal == SIGNAL_BUY || signal == SIGNAL_BUY_STRONG) {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(g_tradeManager.OpenBuyATR(probability, sl_pips, tp_pips, ticket)) {
         g_metrics.RecordTradeOpen(ticket, ORDER_TYPE_BUY, ask, LotSize,
                                   sl_pips * _Point, tp_pips * _Point, current_bar);
         if(EnableLogging)
            g_logger.LogTradeOpen(ticket, ORDER_TYPE_BUY, ask, LotSize,
                                  sl_pips * _Point, tp_pips * _Point, current_bar, probability);
         Print("[Trade] BUY opened | Tkt: ", ticket, " | Prob: ", DoubleToString(probability, 3));
      }
   } else if(signal == SIGNAL_SELL || signal == SIGNAL_SELL_STRONG) {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(g_tradeManager.OpenSellATR(probability, sl_pips, tp_pips, ticket)) {
         g_metrics.RecordTradeOpen(ticket, ORDER_TYPE_SELL, bid, LotSize,
                                   sl_pips * _Point, tp_pips * _Point, current_bar);
         if(EnableLogging)
            g_logger.LogTradeOpen(ticket, ORDER_TYPE_SELL, bid, LotSize,
                                  sl_pips * _Point, tp_pips * _Point, current_bar, probability);
         Print("[Trade] SELL opened | Tkt: ", ticket, " | Prob: ", DoubleToString(probability, 3));
      }
   }
   
   // Manage trailing stops
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == 20260426) {
         g_tradeManager.ManageTrailingStop(PositionGetInteger(POSITION_TICKET), trailing_pips);
      }
   }
   
   // Update equity tracking
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_metrics.UpdateEquity(equity, current_bar);
   
   if(equity > g_peak_equity)
      g_peak_equity = equity;
   
   double dd = (g_peak_equity - equity) / g_peak_equity * 100.0;
   if(dd > g_max_drawdown)
      g_max_drawdown = dd;
}

//+------------------------------------------------------------------+
//| Helper: get current GBrain signal                                |
//+------------------------------------------------------------------+
double g_gbrain_signal() {
   if(g_gbrain_handle == INVALID_HANDLE) return 0.0;
   double buf[];
   if(CopyBuffer(g_gbrain_handle, 0, 0, 1, buf) > 0)
      return buf[0];
   return 0.0;
}

//+------------------------------------------------------------------+
//| ONTRADETRANSACTION                                               |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result) {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
      ulong deal_ticket = trans.deal;
      if(!HistoryDealSelect(deal_ticket)) return;
      if(HistoryDealGetString(deal_ticket, DEAL_SYMBOL) != _Symbol) return;
      if(HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != 20260426) return;
      if(HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;
      
      double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      datetime close_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
      
      double net_profit = profit - commission;
      
      g_metrics.RecordTradeClose(deal_ticket, net_profit, commission, close_time);
      
      if(EnableLogging)
         g_logger.LogTradeClose(type, net_profit, commission, close_time);
      
      g_total_trades++;
      Print("[Backtest] Trade #", g_total_trades,
            " | PnL: $", DoubleToString(net_profit, 2),
            " | Type: ", EnumToString(type));
   }
}

//+------------------------------------------------------------------+
//| ONTESTER                                                         |
//+------------------------------------------------------------------+
double OnTester() {
   g_metrics.CalculateFinalMetrics();
   double calmar = g_metrics.GetCalmarRatio();
   double sharpe = g_metrics.GetSharpeRatio();
   Print("[OnTester] Calmar: ", DoubleToString(calmar, 3),
         " | Sharpe: ", DoubleToString(sharpe, 3),
         " | Trades: ", g_total_trades,
         " | Balance: $", AccountInfoDouble(ACCOUNT_BALANCE));
   return calmar;
}

//+------------------------------------------------------------------+
//| HEURISTIC FALLBACK SIGNALS                                       |
//+------------------------------------------------------------------+
double HeuristicBuySignal() {
   if(g_bars < 20) return 0.0;
   
   double gain = 0, loss = 0;
   for(int j = 1; j <= 7 && (j + 1) < g_bars; j++) {
      double diff = g_close[j] - g_close[j + 1];
      if(diff > 0) gain += diff;
      else loss += MathAbs(diff);
   }
   
   double rsi = 50;
   if(loss > 0.00001)
      rsi = 100.0 - (100.0 / (1.0 + gain / loss));
   
   double roc = 0;
   if(5 < g_bars)
      roc = (g_close[0] - g_close[5]) / (g_close[5] + 0.00001) * 100.0;
   
   // More relaxed: RSI < 50 and positive ROC
   if(rsi < 50 && roc > 0.05) return 0.70;
   return 0.0;
}

double HeuristicSellSignal() {
   if(g_bars < 20) return 0.0;
   
   double gain = 0, loss = 0;
   for(int j = 1; j <= 7 && (j + 1) < g_bars; j++) {
      double diff = g_close[j] - g_close[j + 1];
      if(diff > 0) gain += diff;
      else loss += MathAbs(diff);
   }
   
   double rsi = 50;
   if(loss > 0.00001)
      rsi = 100.0 - (100.0 / (1.0 + gain / loss));
   
   double roc = 0;
   if(5 < g_bars)
      roc = (g_close[0] - g_close[5]) / (g_close[5] + 0.00001) * 100.0;
   
   if(rsi > 50 && roc < -0.05) return 0.70;
   return 0.0;
}
//+------------------------------------------------------------------+