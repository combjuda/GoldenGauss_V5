//+------------------------------------------------------------------+
//| Types.mqh - Common types and structures                         |
//+------------------------------------------------------------------+
#ifndef GOLDENGAUSS_TYPES_MQH
#define GOLDENGAUSS_TYPES_MQH

//+------------------------------------------------------------------+
//| Feature Vector                                                   |
//+------------------------------------------------------------------+
struct SFeatureVector {
   double features[];
   datetime bar_time;
   bool is_valid;
};

//+------------------------------------------------------------------+
//| Model Prediction Result                                          |
//+------------------------------------------------------------------+
struct SPrediction {
   double probability;
   int predicted_class;    // 0=NEUTRAL, 1=BUY, 2=SELL
   double confidence;
   bool is_valid;
};

//+------------------------------------------------------------------+
//| Training Sample                                                  |
//+------------------------------------------------------------------+
struct STrainingSample {
   double features[];
   int label;              // 0=no trade, 1=BUY, 2=SELL
   double outcome;
   datetime timestamp;
};

//+------------------------------------------------------------------+
//| Trade Signal                                                     |
//+------------------------------------------------------------------+
enum ENUM_TRADE_SIGNAL {
   SIGNAL_NONE        = 0,
   SIGNAL_BUY         = 1,
   SIGNAL_SELL        = 2,
   SIGNAL_BUY_STRONG  = 3,
   SIGNAL_SELL_STRONG = 4
};

//+------------------------------------------------------------------+
//| ATR Regime                                                       |
//+------------------------------------------------------------------+
enum ENUM_ATR_REGIME {
   ATR_LOW_VOLATILITY   = 0,
   ATR_NORMAL          = 1,
   ATR_HIGH_VOLATILITY = 2
};

//+------------------------------------------------------------------+
//| Candle Data                                                      |
//+------------------------------------------------------------------+
struct SCandleData {
   datetime time;
   double open;
   double high;
   double low;
   double close;
   long volume;
   double atr14;
   double atr50;
   double atr200;
};

//+------------------------------------------------------------------+
//| Trade Outcome                                                   |
//+------------------------------------------------------------------+
struct STradeOutcome {
   datetime entry_time;
   ulong position_ticket;
   double entry_price;
   double atr_at_entry;
   double sl_distance_pips;
   double tp_distance_pips;
   double actual_sl_pips;
   double actual_tp_pips;
   double gbrain_confirm;
   double roc_1;
   double rsi_14;
   double vwap_dev;
   bool is_win;
   double pnl_pips;
   int holding_bars;
   int exit_reason;
};

//+------------------------------------------------------------------+
//| ATR Calibration                                                  |
//+------------------------------------------------------------------+
struct SATRCalibration {
   double sl_multiplier;
   double tp_multiplier;
   double trailing_multiplier;
   int sample_count;
   datetime last_update;
};

//+------------------------------------------------------------------+
//| Indicator Weights (Self-Learning)                               |
//+------------------------------------------------------------------+
struct SIndicatorWeights {
   double gbrain_weight;
   double atr_regime_weight;
   double roc_weight;
   double vwap_weight;
   double rsi_weight;
   double momentum_weight;
   datetime last_update;
};

//+------------------------------------------------------------------+
//| Constants                                                        |
//+------------------------------------------------------------------+
#define NUM_FEATURES      240
#define MAX_TRADES       10000
#define MAX_CANDLES      43200

#endif
