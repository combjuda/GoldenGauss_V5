//+------------------------------------------------------------------+
//| GoldenGauss_Backtester.mq5                                       |
//| Professional Backtesting Framework for GoldenGauss_V5            |
//| Peter - Version 1.01                                             |
//+------------------------------------------------------------------+
#property copyright "Peter"
#property version   "1.01"
#property description "GoldenGauss V5 - Professional Backtester"

//+------------------------------------------------------------------+
//| INCLUDES                                                         |
//+------------------------------------------------------------------+
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
input string          SetFileName = "XAUUSD_Balanced.set";
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
double            g_peak_equity = 0;
double            g_max_drawdown = 0;

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
//| ONINIT                                                           |
//+------------------------------------------------------------------+
int OnInit() {
   Print("==================================================");
   Print("  GoldenGauss V5 - Professional Backtester");
   Print("==================================================");
   Print("Initial Deposit: $", InitialDeposit);
   Print("Slippage: ", SlippagePips, " pips");
   Print("Commission: $", CommissionPerLot, " per lot");
   
   // Initialize logger
   if(EnableLogging)
      g_logger.Initialize(LogFileName);
   
   // Initialize metrics with initial deposit
   g_metrics.Initialize(InitialDeposit);
   
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
   g_atr_handle = iATR(_Symbol, PERIOD_CURRENT, ATRPeriod);
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
   
   // Load models
   g_nnBuy = new CNeuralNetworkV4(NUM_FEATURES, 64, 2, 0.001);
   g_nnSell = new CNeuralNetworkV4(NUM_FEATURES, 64, 2, 0.001);
   
   string sym = _Symbol;
   string dataPath = "MQL5/Files/" + DataDirectory + "_" + sym + "/Models/";
   
   Print("[Backtest] Looking for models in: ", dataPath);
   
   bool buy_ok = g_nnBuy.Load(dataPath + BuyModelPath);
   bool sell_ok = g_nnSell.Load(dataPath + SellModelPath);
   g_models_loaded = buy_ok && sell_ok;
   
   if(g_models_loaded) {
      Print("[Backtest] Models loaded successfully");
   } else {
      Print("[Backtest] WARNING: Models not loaded!");
      Print("[Backtest] Please run GoldenGauss_EA_V5_Trainer first");
      Print("[Backtest] Expected files:");
      Print("  - ", dataPath, BuyModelPath);
      Print("  - ", dataPath, SellModelPath);
      Print("  - ", dataPath, "norm_params.dat");
      Print("[Backtest] Using heuristic signal mode (no NN models)");
   }
   
   // Load normalization
   if(UseNormalization) {
      g_normalization_loaded = LoadNormalizationParams(dataPath + "norm_params.dat");
      if(!g_normalization_loaded) {
         Print("[Backtest] WARNING: Normalization params not found");
         Print("[Backtest] Running without normalization");
      }
   }
   
   // Initialize GBrain
   if(UseGBrain) {
      g_gbrain_handle = iCustom(_Symbol, PERIOD_CURRENT, "GBrain", 10, 80, 200, 0.002, 20, 0.3, 0);
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
   // Cleanup in proper order
   g_featureBuilder.FreeCalculators();
   
   if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
   if(g_gbrain_handle != INVALID_HANDLE) IndicatorRelease(g_gbrain_handle);
   
   // Delete neural networks
   if(g_nnBuy != NULL) { delete g_nnBuy; g_nnBuy = NULL; }
   if(g_nnSell != NULL) { delete g_nnSell; g_nnSell = NULL; }
   
   // Generate final report
   Print("==================================================");
   Print("  BACKTEST COMPLETE");
   Print("==================================================");
   
   // Calculate final metrics
   g_metrics.CalculateFinalMetrics();
   
   // Print summary
   g_metrics.PrintSummary();
   
   // Generate HTML report
   g_report.Generate(g_metrics, "Backtest/backtest_results.html");
   
   // Close logger
   if(EnableLogging)
      g_logger.Close();
   
   Print("Full report saved to: Backtest/backtest_results.html");
}

//+------------------------------------------------------------------+
//| ONTICK                                                           |
//+------------------------------------------------------------------+
void OnTick() {
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
   double buy_prob = 0.0, sell_prob = 0.0;
   
   if(g_models_loaded) {
      double buy_probs[], sell_probs[];
      g_nnBuy.Predict(features, buy_probs);
      g_nnSell.Predict(features, sell_probs);
      buy_prob = (ArraySize(buy_probs) >= 2) ? buy_probs[1] : 0.0;
      sell_prob = (ArraySize(sell_probs) >= 2) ? sell_probs[1] : 0.0;
   }
   
   // Check GBrain confirmation
   double gbrain_signal = 0.0;
   if(UseGBrain && g_gbrain_handle != INVALID_HANDLE) {
      double buf[];
      if(CopyBuffer(g_gbrain_handle, 3, 0, 1, buf) > 0)
         gbrain_signal = buf[0];
   }
   
   // Generate signal
   ENUM_TRADE_SIGNAL signal = SIGNAL_NONE;
   double probability = 0.0;
   
   if(buy_prob >= 0.65 && gbrain_signal >= MinGBrainConfirm) {
      signal = (buy_prob >= 0.80) ? SIGNAL_BUY_STRONG : SIGNAL_BUY;
      probability = buy_prob;
   } else if(sell_prob >= 0.65 && gbrain_signal <= -MinGBrainConfirm) {
      signal = (sell_prob >= 0.80) ? SIGNAL_SELL_STRONG : SIGNAL_SELL;
      probability = sell_prob;
   }
   
   if(signal == SIGNAL_NONE) return;
   
   // Calculate stops
   double atr = 0;
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_atr_handle, 0, 0, 1, buf) > 0) atr = buf[0];
   
   double sl_pips = (atr / _Point) * DefaultSLMultiplier;
   double tp_pips = (atr / _Point) * DefaultTPMultiplier;
   double trailing_pips = (atr / _Point) * DefaultTrailingMultiplier;
   
   // Open trade
   ulong ticket = 0;
   double entry_price = 0;
   ENUM_ORDER_TYPE order_type = WRONG_VALUE;
   
   if(signal == SIGNAL_BUY || signal == SIGNAL_BUY_STRONG) {
      entry_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      order_type = ORDER_TYPE_BUY;
      if(g_tradeManager.OpenBuyATR(probability, sl_pips, tp_pips, ticket)) {
         g_metrics.RecordTradeOpen(ticket, order_type, entry_price, LotSize, 
                                   sl_pips * _Point, tp_pips * _Point, current_bar);
         if(EnableLogging)
            g_logger.LogTradeOpen(ticket, order_type, entry_price, LotSize, 
                                  sl_pips * _Point, tp_pips * _Point, current_bar, probability);
      }
   } else if(signal == SIGNAL_SELL || signal == SIGNAL_SELL_STRONG) {
      entry_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      order_type = ORDER_TYPE_SELL;
      if(g_tradeManager.OpenSellATR(probability, sl_pips, tp_pips, ticket)) {
         g_metrics.RecordTradeOpen(ticket, order_type, entry_price, LotSize,
                                   sl_pips * _Point, tp_pips * _Point, current_bar);
         if(EnableLogging)
            g_logger.LogTradeOpen(ticket, order_type, entry_price, LotSize,
                                  sl_pips * _Point, tp_pips * _Point, current_bar, probability);
      }
   }
   
   // Manage trailing stops
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == 20260426) {
         ulong t = (ulong)PositionGetInteger(POSITION_TICKET);
         g_tradeManager.ManageTrailingStop(t, trailing_pips);
      }
   }
   
   // Update equity metrics
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_metrics.UpdateEquity(equity, TimeCurrent());
   
   if(equity > g_peak_equity)
      g_peak_equity = equity;
   
   double drawdown = (g_peak_equity - equity) / g_peak_equity * 100.0;
   if(drawdown > g_max_drawdown)
      g_max_drawdown = drawdown;
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
      
      // Get trade details
      double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      datetime close_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
      
      // Calculate net profit (including slippage simulation)
      double net_profit = profit - commission - (SlippagePips * _Point * LotSize * 2);
      
      // Record trade close
      g_metrics.RecordTradeClose(deal_ticket, net_profit, commission, close_time);
      
      if(EnableLogging)
         g_logger.LogTradeClose(type, net_profit, commission, close_time);
      
      g_total_trades++;
      Print("[Backtest] Trade #", g_total_trades, 
            " | Profit: $", DoubleToString(net_profit, 2),
            " | Commission: $", DoubleToString(commission, 2));
   }
}

//+------------------------------------------------------------------+
//| ONTESTER                                                         |
//+------------------------------------------------------------------+
double OnTester() {
   g_metrics.CalculateFinalMetrics();
   double calmar = g_metrics.GetCalmarRatio();
   Print("[OnTester] Calmar Ratio: ", DoubleToString(calmar, 3));
   return calmar;
}
//+------------------------------------------------------------------+