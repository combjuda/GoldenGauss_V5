//+------------------------------------------------------------------+
//| FeatureBuilder.mqh - Modular Feature Extraction                   |
//| Peter 2026-04-26                                                 |
//| 240 features total across 8 categories                           |
//+------------------------------------------------------------------+
#ifndef GOLDENGAUSS_FEATURE_BUILDER_MQH
#define GOLDENGAUSS_FEATURE_BUILDER_MQH

#include "../Core/Types.mqh"

//+------------------------------------------------------------------+
//| External global arrays (declared in EA/Trainer)                  |
//+------------------------------------------------------------------+
extern double g_close[];
extern double g_open[];
extern double g_high[];
extern double g_low[];
extern long g_volume[];
extern int g_bars;

//+------------------------------------------------------------------+
//| Base Feature Calculator Interface                                |
//+------------------------------------------------------------------+
class IFeatureCalculator {
public:
   virtual int GetStartIndex() const = 0;
   virtual int GetCount() const = 0;
   virtual void Calculate(double &features[], int start_idx, int &bars) = 0;
   virtual string GetName() const = 0;
};

//+------------------------------------------------------------------+
//| Volatility Features (0-29)                                       |
//+------------------------------------------------------------------+
class CVolatilityFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 0; }
   int GetCount() const override { return 30; }
   string GetName() const override { return "Volatility"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      // ATR ratio (3 features)
      double atr = iATR(_Symbol, PERIOD_M1, 14);
      double atr_sma = 0;
      for(int i = 0; i < 20; i++)
         atr_sma += iATR(_Symbol, PERIOD_M1, 14);
      atr_sma /= 20.0;
      features[idx++] = atr / (atr_sma + 0.00001);
      features[idx++] = atr;
      features[idx++] = atr_sma;
      
      // Bollinger Band position (3 features)
      double bb_mid = 0, bb_std = 0;
      for(int i = 0; i < 20; i++)
         bb_mid += g_close[i];
      bb_mid /= 20.0;
      for(int i = 0; i < 20; i++)
         bb_std += MathPow(g_close[i] - bb_mid, 2);
      bb_std = MathSqrt(bb_std / 20.0);
      double bb_upper = bb_mid + 2 * bb_std;
      double bb_lower = bb_mid - 2 * bb_std;
      features[idx++] = (g_close[0] - bb_lower) / (bb_upper - bb_lower + 0.00001);
      features[idx++] = bb_mid;
      features[idx++] = bb_std;
      
      // Historical volatility (2 features)
      double hist_vol = 0;
      for(int i = 0; i < 20; i++) {
         double ret = (g_close[i] - g_close[i+1]) / (g_close[i+1] + 0.00001);
         hist_vol += ret * ret;
      }
      features[idx++] = MathSqrt(hist_vol / 20.0) * 100.0;
      features[idx++] = hist_vol;
      
      // Range position (2 features)
      double day_high = g_high[0], day_low = g_low[0];
      for(int i = 1; i < MathMin(1440, bars); i++) {
         day_high = MathMax(day_high, g_high[i]);
         day_low = MathMin(day_low, g_low[i]);
      }
      features[idx++] = (g_close[0] - day_low) / (day_high - day_low + 0.00001);
      features[idx++] = day_high - day_low;
      
      // Keltner Channel (5 features)
      double kc_mid = 0;
      for(int i = 0; i < 20; i++)
         kc_mid += g_close[i];
      kc_mid /= 20.0;
      double kc_atr = iATR(_Symbol, PERIOD_M1, 20);
      double kc_upper = kc_mid + 2 * kc_atr;
      double kc_lower = kc_mid - 2 * kc_atr;
      features[idx++] = (g_close[0] - kc_lower) / (kc_upper - kc_lower + 0.00001);
      features[idx++] = kc_mid;
      features[idx++] = kc_upper;
      features[idx++] = kc_lower;
      features[idx++] = kc_atr;
      
      // Donchian Channel (5 features)
      double dc_upper = g_high[0], dc_lower = g_low[0];
      for(int i = 0; i < 20; i++) {
         dc_upper = MathMax(dc_upper, g_high[i]);
         dc_lower = MathMin(dc_lower, g_low[i]);
      }
      double dc_mid = (dc_upper + dc_lower) / 2.0;
      features[idx++] = (g_close[0] - dc_lower) / (dc_upper - dc_lower + 0.00001);
      features[idx++] = dc_upper;
      features[idx++] = dc_lower;
      features[idx++] = dc_mid;
      features[idx++] = dc_upper - dc_lower;
      
      // Compression (2 features)
      double compression = 0;
      for(int i = 1; i < 20; i++)
         compression += MathAbs(g_close[i] - g_close[i-1]);
      compression /= 20.0;
      features[idx++] = compression;
      features[idx++] = compression / (atr + 0.00001);
      
      // Volatility regime (2 features)
      double vol_ratio = atr / (atr_sma + 0.00001);
      features[idx++] = (vol_ratio < 0.8) ? 1.0 : 0.0;
      features[idx++] = (vol_ratio > 1.2) ? 1.0 : 0.0;
      
      // Additional metrics (6 features)
      features[idx++] = atr / (_Point * 100);
      features[idx++] = (g_high[0] - g_low[0]) / (g_close[0] + 0.00001) * 100.0;
      features[idx++] = bb_std / (bb_mid + 0.00001) * 100.0;
      features[idx++] = (bb_upper - bb_lower) / (bb_mid + 0.00001);
      features[idx++] = MathMax(MathAbs(g_close[0] - bb_upper), MathAbs(g_close[0] - bb_lower)) / (atr + 0.00001);
      features[idx++] = vol_ratio * vol_ratio;
      
      bars = MathMax(bars, 1440);
   }
};

//+------------------------------------------------------------------+
//| Momentum Features (30-69)                                        |
//+------------------------------------------------------------------+
class CMomentumFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 30; }
   int GetCount() const override { return 40; }
   string GetName() const override { return "Momentum"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      // Multi-scale ROC (4 features)
      double roc1 = (g_close[0] - g_close[1]) / (g_close[1] + 0.00001) * 100.0;
      double roc5 = (g_close[0] - g_close[5]) / (g_close[5] + 0.00001) * 100.0;
      double roc15 = (g_close[0] - g_close[15]) / (g_close[15] + 0.00001) * 100.0;
      double roc60 = (g_close[0] - g_close[60]) / (g_close[60] + 0.00001) * 100.0;
      features[idx++] = roc1;
      features[idx++] = roc5;
      features[idx++] = roc15;
      features[idx++] = roc60;
      
      // RSI multiple periods (3 features)
      double rsi7 = iRSI(_Symbol, PERIOD_M1, 7, PRICE_CLOSE);
      double rsi14 = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
      double rsi28 = iRSI(_Symbol, PERIOD_M1, 28, PRICE_CLOSE);
      features[idx++] = rsi7 / 100.0;
      features[idx++] = rsi14 / 100.0;
      features[idx++] = rsi28 / 100.0;
      
      // Stochastic (2 features)
      static int stoch_handle = INVALID_HANDLE;
      double stoch_main = 0, stoch_signal = 0;
      if(stoch_handle == INVALID_HANDLE)
         stoch_handle = iStochastic(_Symbol, PERIOD_M1, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
      if(stoch_handle != INVALID_HANDLE) {
         double buf_main[], buf_sig[];
         ArraySetAsSeries(buf_main, true);
         ArraySetAsSeries(buf_sig, true);
         if(CopyBuffer(stoch_handle, 0, 0, 1, buf_main) > 0)
            stoch_main = buf_main[0];
         if(CopyBuffer(stoch_handle, 1, 0, 1, buf_sig) > 0)
            stoch_signal = buf_sig[0];
      }
      features[idx++] = stoch_main / 100.0;
      features[idx++] = stoch_signal / 100.0;
      
      // MACD (3 features)
      double macd_main = iMACD(_Symbol, PERIOD_M1, 12, 26, 9, PRICE_CLOSE);
      static double macd_buffer[];
      static double macd_ema = 0;
      static bool first_macd = true;
      if(first_macd) {
         ArrayResize(macd_buffer, 9);
         ArrayInitialize(macd_buffer, 0);
         first_macd = false;
      }
      for(int i = 8; i > 0; i--) macd_buffer[i] = macd_buffer[i-1];
      macd_buffer[0] = macd_main;
      double alpha = 2.0 / 10.0;
      if(macd_ema == 0) macd_ema = macd_main;
      macd_ema = macd_ema * (1 - alpha) + macd_main * alpha;
      double macd_hist = macd_main - macd_ema;
      features[idx++] = macd_main / _Point;
      features[idx++] = macd_ema / _Point;
      features[idx++] = macd_hist / _Point;
      
      // ADX (3 features)
      static int adx_handle = INVALID_HANDLE;
      double adx = 0, plus_di = 0, minus_di = 0;
      if(adx_handle == INVALID_HANDLE)
         adx_handle = iADX(_Symbol, PERIOD_M1, 14);
      if(adx_handle != INVALID_HANDLE) {
         double buf_adx[], buf_plus[], buf_minus[];
         ArraySetAsSeries(buf_adx, true);
         ArraySetAsSeries(buf_plus, true);
         ArraySetAsSeries(buf_minus, true);
         if(CopyBuffer(adx_handle, 0, 0, 1, buf_adx) > 0) adx = buf_adx[0];
         if(CopyBuffer(adx_handle, 1, 0, 1, buf_plus) > 0) plus_di = buf_plus[0];
         if(CopyBuffer(adx_handle, 2, 0, 1, buf_minus) > 0) minus_di = buf_minus[0];
      }
      features[idx++] = adx / 100.0;
      features[idx++] = plus_di / 100.0;
      features[idx++] = minus_di / 100.0;
      
      // CCI (4 features)
      double cci = iCCI(_Symbol, PERIOD_M1, 20, PRICE_TYPICAL);
      double cci5 = iCCI(_Symbol, PERIOD_M1, 5, PRICE_TYPICAL);
      double cci_raw = 0, cci_sma = 0;
      for(int i = 0; i < 20; i++)
         cci_raw += (g_high[i] + g_low[i] + g_close[i]) / 3.0;
      cci_sma = cci_raw / 20.0;
      double cci_mean_dev = 0;
      for(int i = 0; i < 20; i++)
         cci_mean_dev += MathAbs((g_high[i] + g_low[i] + g_close[i]) / 3.0 - cci_sma);
      cci_mean_dev /= 20.0;
      features[idx++] = cci / 100.0;
      features[idx++] = cci5 / 100.0;
      features[idx++] = cci_raw / (cci_mean_dev + 0.00001);
      features[idx++] = cci_sma;
      
      // Williams %R (2 features)
      double williams = iWPR(_Symbol, PERIOD_M1, 14);
      features[idx++] = MathAbs(williams) / 100.0;
      
      // Momentum (2 features)
      double mom = iMomentum(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
      double mom5 = iMomentum(_Symbol, PERIOD_M1, 5, PRICE_CLOSE);
      features[idx++] = mom / 1000.0;
      features[idx++] = mom5 / 1000.0;
      
      // RVI (2 features)
      double rvi = iRVI(_Symbol, PERIOD_M1, 10);
      features[idx++] = rvi / 100.0;
      features[idx++] = 0;  // Signal line reserved
      
      // Momentum signals (3 features)
      features[idx++] = (roc1 > 0) ? 1.0 : 0.0;
      features[idx++] = (roc5 > 0) ? 1.0 : 0.0;
      features[idx++] = (roc15 > 0) ? 1.0 : 0.0;
      
      // RSI signals (3 features)
      features[idx++] = (rsi7 < 30) ? 1.0 : 0.0;
      features[idx++] = (rsi7 > 70) ? 1.0 : 0.0;
      features[idx++] = (rsi7 < rsi14) ? 1.0 : 0.0;
      
      // Momentum regime (2 features)
      features[idx++] = (MathAbs(roc1) > MathAbs(roc5)) ? 1.0 : 0.0;
      features[idx++] = (roc60 > 0 && roc15 > 0) ? 1.0 : 0.0;
      
      bars = MathMax(bars, 60);
   }
};

//+------------------------------------------------------------------+
//| VWAP Features (70-99)                                            |
//+------------------------------------------------------------------+
class CVWAPFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 70; }
   int GetCount() const override { return 30; }
   string GetName() const override { return "VWAP"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      // Session VWAP (8 features)
      double vwap = 0, typical_sum = 0, volume_sum = 0;
      int vwap_bars = MathMin(100, bars);
      for(int i = 0; i < vwap_bars; i++) {
         double typical = (g_high[i] + g_low[i] + g_close[i]) / 3.0;
         typical_sum += typical * (double)g_volume[i];
         volume_sum += (double)g_volume[i];
      }
      if(volume_sum > 0) vwap = typical_sum / volume_sum;
      
      double vwap_dev = (g_close[0] - vwap) / (vwap + 0.00001);
      features[idx++] = vwap_dev;
      features[idx++] = vwap;
      features[idx++] = typical_sum;
      features[idx++] = volume_sum;
      features[idx++] = (g_close[0] < vwap) ? 1.0 : 0.0;
      features[idx++] = (g_close[0] > vwap) ? 1.0 : 0.0;
      features[idx++] = vwap_dev * 100.0;
      features[idx++] = MathAbs(vwap_dev);
      
      // Multi-session VWAP (4 features)
      double vwap50 = 0, vol50 = 0, vwap200 = 0, vol200 = 0;
      for(int i = 0; i < MathMin(50, bars); i++) {
         double typical = (g_high[i] + g_low[i] + g_close[i]) / 3.0;
         vwap50 += typical * (double)g_volume[i];
         vol50 += (double)g_volume[i];
      }
      for(int i = 0; i < MathMin(200, bars); i++) {
         double typical = (g_high[i] + g_low[i] + g_close[i]) / 3.0;
         vwap200 += typical * (double)g_volume[i];
         vol200 += (double)g_volume[i];
      }
      vwap50 = (vol50 > 0) ? vwap50 / vol50 : 0;
      vwap200 = (vol200 > 0) ? vwap200 / vol200 : 0;
      features[idx++] = (g_close[0] - vwap50) / (vwap50 + 0.00001);
      features[idx++] = (g_close[0] - vwap200) / (vwap200 + 0.00001);
      features[idx++] = (vwap50 - vwap200) / (vwap200 + 0.00001);
      features[idx++] = (vwap > vwap200) ? 1.0 : 0.0;
      
      // VWAP deviation bands (6 features)
      double vwap_std = 0;
      for(int i = 0; i < MathMin(50, bars); i++) {
         double typical = (g_high[i] + g_low[i] + g_close[i]) / 3.0;
         vwap_std += MathPow(typical - vwap, 2);
      }
      vwap_std = MathSqrt(vwap_std / MathMin(50, bars));
      features[idx++] = vwap_std;
      features[idx++] = vwap + 2 * vwap_std;
      features[idx++] = vwap - 2 * vwap_std;
      features[idx++] = vwap_dev / (vwap_std + 0.00001);
      features[idx++] = (g_close[0] > vwap + 2 * vwap_std) ? 1.0 : 0.0;
      features[idx++] = (g_close[0] < vwap - 2 * vwap_std) ? 1.0 : 0.0;
      
      // VWAP momentum (4 features)
      double vwap_rate = (vwap - g_close[5]) / (g_close[5] + 0.00001) * 100.0;
      features[idx++] = vwap_rate;
      features[idx++] = vwap_rate * 10;
      features[idx++] = (vwap_rate > 0) ? 1.0 : 0.0;
      features[idx++] = MathAbs(vwap_dev);
      
      // VWAP relative position (4 features)
      double daily_range = g_high[0] - g_low[0];
      double close_pos = (daily_range > 0) ? (g_close[0] - g_low[0]) / daily_range : 0.5;
      features[idx++] = close_pos;
      features[idx++] = (close_pos < 0.3) ? 1.0 : 0.0;
      features[idx++] = (close_pos > 0.7) ? 1.0 : 0.0;
      features[idx++] = vwap_dev;
      
      // Reserved (4 features)
      for(int i = 0; i < 4; i++)
         features[idx++] = 0.0;
      
      bars = MathMax(bars, 200);
   }
};

//+------------------------------------------------------------------+
//| Volume Features (100-129)                                        |
//+------------------------------------------------------------------+
class CVolumeFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 100; }
   int GetCount() const override { return 30; }
   string GetName() const override { return "Volume"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      // Volume MA (5 features)
      double vol_ma20 = 0;
      for(int i = 0; i < 20; i++)
         vol_ma20 += (double)g_volume[i];
      vol_ma20 /= 20.0;
      double vol_ratio = (vol_ma20 > 0) ? (double)g_volume[0] / vol_ma20 : 1.0;
      features[idx++] = vol_ratio;
      features[idx++] = vol_ma20;
      features[idx++] = (double)g_volume[0];
      features[idx++] = (vol_ratio > 2.0) ? 1.0 : 0.0;
      features[idx++] = (vol_ratio < 0.5) ? 1.0 : 0.0;
      
      // Volume spikes (3 features)
      double vol_std = 0;
      for(int i = 0; i < 20; i++) {
         vol_std += MathPow((double)g_volume[i] - vol_ma20, 2);
      }
      vol_std = MathSqrt(vol_std / 20.0);
      features[idx++] = vol_std;
      features[idx++] = ((double)g_volume[0] - vol_ma20) / (vol_std + 0.00001);
      features[idx++] = (vol_ratio > 1.5) ? 1.0 : 0.0;
      
      // OBV (4 features)
      double obv = 0;
      for(int i = 1; i < MathMin(20, bars); i++) {
         if(g_close[i] > g_close[i+1])
            obv += (double)g_volume[i];
         else if(g_close[i] < g_close[i+1])
            obv -= (double)g_volume[i];
      }
      double obv_ma = obv / 20.0;
      features[idx++] = obv;
      features[idx++] = obv_ma;
      features[idx++] = (obv > obv_ma) ? 1.0 : 0.0;
      features[idx++] = (obv > 0) ? 1.0 : 0.0;
      
      // Volume profile (6 features)
      double vol_up = 0, vol_down = 0;
      for(int i = 0; i < MathMin(20, bars); i++) {
         if(g_close[i] > g_open[i])
            vol_up += (double)g_volume[i];
         else if(g_close[i] < g_open[i])
            vol_down += (double)g_volume[i];
      }
      double vol_balance = (vol_up + vol_down > 0) ? (vol_up - vol_down) / (vol_up + vol_down) : 0;
      features[idx++] = vol_balance;
      features[idx++] = vol_up;
      features[idx++] = vol_down;
      features[idx++] = (vol_balance > 0.2) ? 1.0 : 0.0;
      features[idx++] = (vol_balance < -0.2) ? 1.0 : 0.0;
      features[idx++] = vol_balance * 100.0;
      
      // Delta volume (5 features)
      double delta = 0, delta_sum = 0;
      for(int i = 0; i < MathMin(20, bars); i++) {
         double bar_delta = ((g_close[i] - g_open[i]) / (_Point + 0.00001)) * (double)g_volume[i];
         delta_sum += bar_delta;
         if(i < 5) delta += bar_delta;
      }
      features[idx++] = delta;
      features[idx++] = delta_sum;
      features[idx++] = delta / (MathAbs(delta_sum) + 0.00001);
      features[idx++] = (delta > 0) ? 1.0 : 0.0;
      features[idx++] = delta_sum / (vol_ma20 * 20 + 0.00001);
      
      // Big trades detection (3 features)
      double big_trade_vol = 0;
      for(int i = 0; i < MathMin(50, bars); i++) {
         if(g_volume[i] > vol_ma20 * 3)
            big_trade_vol += (double)g_volume[i];
      }
      features[idx++] = big_trade_vol / (vol_ma20 * 50 + 0.00001);
      features[idx++] = (big_trade_vol > vol_ma20 * 10) ? 1.0 : 0.0;
      features[idx++] = big_trade_vol;
      
      // Reserved (4 features)
      for(int i = 0; i < 4; i++)
         features[idx++] = 0.0;
      
      bars = MathMax(bars, 50);
   }
};

//+------------------------------------------------------------------+
//| Structure Features (130-159)                                     |
//+------------------------------------------------------------------+
class CStructureFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 130; }
   int GetCount() const override { return 30; }
   string GetName() const override { return "Structure"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      // Swing detection (6 features)
      double swing_high = g_high[0], swing_low = g_low[0];
      int swing_bars = MathMin(50, bars);
      for(int i = 0; i < swing_bars; i++) {
         swing_high = MathMax(swing_high, g_high[i]);
         swing_low = MathMin(swing_low, g_low[i]);
      }
      double swing_pos = (swing_high - swing_low > 0) ? (g_close[0] - swing_low) / (swing_high - swing_low) : 0.5;
      features[idx++] = swing_pos;
      features[idx++] = swing_high;
      features[idx++] = swing_low;
      features[idx++] = swing_high - swing_low;
      features[idx++] = (swing_pos > 0.8) ? 1.0 : 0.0;
      features[idx++] = (swing_pos < 0.2) ? 1.0 : 0.0;
      
      // Support/Resistance proximity (6 features)
      double sr_level_high = swing_high * 0.995;
      double sr_level_low = swing_low * 1.005;
      double close_to_sr_high = MathAbs(g_close[0] - sr_level_high) / (_Point + 0.00001);
      double close_to_sr_low = MathAbs(g_close[0] - sr_level_low) / (_Point + 0.00001);
      features[idx++] = close_to_sr_high;
      features[idx++] = close_to_sr_low;
      features[idx++] = (close_to_sr_high < 50) ? 1.0 : 0.0;
      features[idx++] = (close_to_sr_low < 50) ? 1.0 : 0.0;
      features[idx++] = sr_level_high;
      features[idx++] = sr_level_low;
      
      // Trend structure (6 features)
      double higher_highs = 0, higher_lows = 0, lower_highs = 0, lower_lows = 0;
      for(int i = 1; i < MathMin(10, bars); i++) {
         if(g_high[i] > g_high[i+1]) higher_highs++;
         if(g_low[i] > g_low[i+1]) higher_lows++;
         if(g_high[i] < g_high[i+1]) lower_highs++;
         if(g_low[i] < g_low[i+1]) lower_lows++;
      }
      features[idx++] = higher_highs / 10.0;
      features[idx++] = higher_lows / 10.0;
      features[idx++] = lower_highs / 10.0;
      features[idx++] = lower_lows / 10.0;
      features[idx++] = (higher_highs > lower_highs) ? 1.0 : 0.0;
      features[idx++] = (higher_lows > lower_lows) ? 1.0 : 0.0;
      
      // Breakout detection (6 features)
      double atr = iATR(_Symbol, PERIOD_M1, 14);
      bool bullish_breakout = (g_close[0] > swing_high - atr * 0.5);
      bool bearish_breakout = (g_close[0] < swing_low + atr * 0.5);
      features[idx++] = (bullish_breakout) ? 1.0 : 0.0;
      features[idx++] = (bearish_breakout) ? 1.0 : 0.0;
      features[idx++] = (bullish_breakout && higher_highs > 5) ? 1.0 : 0.0;
      features[idx++] = (bearish_breakout && lower_lows > 5) ? 1.0 : 0.0;
      features[idx++] = (g_close[0] > swing_high) ? 1.0 : 0.0;
      features[idx++] = (g_close[0] < swing_low) ? 1.0 : 0.0;
      
      // Fractal approximation (6 features)
      features[idx++] = (g_high[0] >= g_high[1] && g_high[0] >= g_high[2]) ? 1.0 : 0.0;
      features[idx++] = (g_low[0] <= g_low[1] && g_low[0] <= g_low[2]) ? 1.0 : 0.0;
      features[idx++] = (g_high[0] >= g_high[1]) ? 1.0 : 0.0;
      features[idx++] = (g_low[0] <= g_low[1]) ? 1.0 : 0.0;
      features[idx++] = 0.0;  // Reserved
      features[idx++] = 0.0;  // Reserved
      
      bars = MathMax(bars, 50);
   }
};

//+------------------------------------------------------------------+
//| Cross-Asset Features (160-189)                                  |
//+------------------------------------------------------------------+
class CCrossAssetFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 160; }
   int GetCount() const override { return 30; }
   string GetName() const override { return "CrossAsset"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      // DXY correlation (5 features)
      double dxy_handle = iOpen(_Symbol, PERIOD_CURRENT, 0);
      features[idx++] = 0;  // DXY rate (placeholder)
      features[idx++] = 0;
      features[idx++] = 0;
      features[idx++] = 0;
      features[idx++] = 0;
      
      // Gold/USD correlation (5 features)
      features[idx++] = 0;  // XAUUSD placeholder
      features[idx++] = 0;
      features[idx++] = 0;
      features[idx++] = 0;
      features[idx++] = 0;
      
      // EUR/USD cross (5 features)
      features[idx++] = 0;  // EURUSD placeholder
      features[idx++] = 0;
      features[idx++] = 0;
      features[idx++] = 0;
      features[idx++] = 0;
      
      // Risk sentiment proxy (5 features)
      double risk_score = 0;
      if(iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE) < 50) risk_score += 0.3;
      if(iATR(_Symbol, PERIOD_M1, 14) > iATR(_Symbol, PERIOD_M1, 50)) risk_score += 0.2;
      features[idx++] = risk_score;
      features[idx++] = (risk_score > 0.5) ? 1.0 : 0.0;
      features[idx++] = 0;  // Reserved
      features[idx++] = 0;  // Reserved
      features[idx++] = 0;  // Reserved
      
      // Correlation matrix (10 features) - placeholders
      for(int i = 0; i < 10; i++)
         features[idx++] = 0;
      
      bars = MathMax(bars, 1);
   }
};

//+------------------------------------------------------------------+
//| Micro Features (190-219)                                          |
//+------------------------------------------------------------------+
class CMicroFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 190; }
   int GetCount() const override { return 30; }
   string GetName() const override { return "Micro"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      // Range metrics (5 features)
      double range = g_high[0] - g_low[0];
      double range_pct = range / (g_close[0] + 0.00001) * 100.0;
      features[idx++] = range;
      features[idx++] = range_pct;
      features[idx++] = range / (_Point * 100);
      features[idx++] = (range_pct > 2.0) ? 1.0 : 0.0;
      features[idx++] = (range_pct < 0.5) ? 1.0 : 0.0;
      
      // Close position in range (5 features)
      double close_pos = (range > 0) ? (g_close[0] - g_low[0]) / range : 0.5;
      features[idx++] = close_pos;
      features[idx++] = (close_pos > 0.8) ? 1.0 : 0.0;
      features[idx++] = (close_pos < 0.2) ? 1.0 : 0.0;
      features[idx++] = (close_pos > 0.5) ? 1.0 : 0.0;
      features[idx++] = MathAbs(close_pos - 0.5) * 2.0;
      
      // Gap detection (5 features)
      double gap = g_open[0] - g_close[1];
      double gap_pct = gap / (g_close[1] + 0.00001) * 100.0;
      features[idx++] = gap;
      features[idx++] = gap_pct;
      features[idx++] = (MathAbs(gap) > range * 0.5) ? 1.0 : 0.0;
      features[idx++] = (gap > 0) ? 1.0 : 0.0;
      features[idx++] = (gap < 0) ? 1.0 : 0.0;
      
      // Intraday momentum (5 features)
      double open_close_diff = g_close[0] - g_open[0];
      features[idx++] = open_close_diff;
      features[idx++] = open_close_diff / (_Point * 100);
      features[idx++] = (open_close_diff > 0) ? 1.0 : 0.0;
      features[idx++] = (g_close[0] > g_open[0]) ? 1.0 : 0.0;
      features[idx++] = MathAbs(open_close_diff) / (range + 0.00001);
      
      // Candle pattern detection (10 features)
      features[idx++] = (g_close[0] > g_open[0]) ? 1.0 : 0.0;  // Bullish candle
      features[idx++] = (g_close[0] < g_open[0]) ? 1.0 : 0.0;  // Bearish candle
      features[idx++] = (g_high[0] - g_low[0] < range * 0.3) ? 1.0 : 0.0;  // Small range
      features[idx++] = 0.0;  // Doji placeholder
      features[idx++] = (g_close[0] > g_high[0] - range * 0.1) ? 1.0 : 0.0;  // Close near high
      features[idx++] = (g_close[0] < g_low[0] + range * 0.1) ? 1.0 : 0.0;  // Close near low
      features[idx++] = (open_close_diff > range * 0.8) ? 1.0 : 0.0;  // Strong bullish
      features[idx++] = (open_close_diff < -range * 0.8) ? 1.0 : 0.0;  // Strong bearish
      features[idx++] = 0.0;  // Reserved
      features[idx++] = 0.0;  // Reserved
      
      bars = MathMax(bars, 2);
   }
};

//+------------------------------------------------------------------+
//| Temporal Features (220-239)                                      |
//+------------------------------------------------------------------+
class CTemporalFeatures : public IFeatureCalculator {
public:
   int GetStartIndex() const override { return 220; }
   int GetCount() const override { return 20; }
   string GetName() const override { return "Temporal"; }

   void Calculate(double &features[], int start_idx, int &bars) override {
      int idx = start_idx;
      
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      
      // Hour of day (3 features)
      features[idx++] = (double)dt.hour / 24.0;
      features[idx++] = (dt.hour >= 13 && dt.hour < 18) ? 1.0 : 0.0;  // Trading window
      features[idx++] = (dt.hour >= 20 || dt.hour < 5) ? 1.0 : 0.0;  // Off-hours
      
      // Day of week (4 features)
      features[idx++] = (double)dt.day_of_week / 7.0;
      features[idx++] = (dt.day_of_week >= 1 && dt.day_of_week <= 5) ? 1.0 : 0.0;  // Weekday
      features[idx++] = (dt.day_of_week == 0 || dt.day_of_week == 6) ? 1.0 : 0.0;  // Weekend
      features[idx++] = (dt.day_of_week == 5) ? 1.0 : 0.0;  // Friday
      
      // Time cyclicals (4 features)
      features[idx++] = MathSin(2 * M_PI * dt.hour / 24.0);
      features[idx++] = MathCos(2 * M_PI * dt.hour / 24.0);
      features[idx++] = MathSin(2 * M_PI * dt.day_of_week / 7.0);
      features[idx++] = MathCos(2 * M_PI * dt.day_of_week / 7.0);
      
      // Session time (4 features)
      features[idx++] = (dt.hour >= 8 && dt.hour < 12) ? 1.0 : 0.0;  // London
      features[idx++] = (dt.hour >= 12 && dt.hour < 17) ? 1.0 : 0.0;  // NY
      features[idx++] = (dt.hour >= 22 || dt.hour < 5) ? 1.0 : 0.0;  // Asia
      features[idx++] = 0.0;  // Reserved
      
      // Weekend proximity (5 features)
      features[idx++] = (dt.day_of_week == 4 && dt.hour >= 16) ? 1.0 : 0.0;  // Friday afternoon
      features[idx++] = (dt.day_of_week == 0) ? 1.0 : 0.0;  // Sunday
      features[idx++] = (dt.hour >= 20 && dt.hour < 24) ? 1.0 : 0.0;  // Evening
      features[idx++] = (double)dt.day / 31.0;  // Day of month normalized
      features[idx++] = (double)dt.day_of_year / 365.0;  // Day of year normalized
      
      bars = MathMax(bars, 1);
   }
};

//+------------------------------------------------------------------+
//| Feature Builder                                                  |
//+------------------------------------------------------------------+
class CFeatureBuilder {
private:
   IFeatureCalculator* m_calculators[8];
   int m_num_calculators;

public:
   CFeatureBuilder() {
      m_num_calculators = 0;
   }
   
   void RegisterCalculator(IFeatureCalculator* calc) {
      if(m_num_calculators < 8)
         m_calculators[m_num_calculators++] = calc;
   }
   
   void FreeCalculators() {
      for(int i = 0; i < m_num_calculators; i++) {
         if(m_calculators[i] != NULL) {
            delete m_calculators[i];
            m_calculators[i] = NULL;
         }
      }
      m_num_calculators = 0;
   }
   
   static bool Build(double &features[], int num_features, int bars,
                     const double &close[], const double &open[],
                     const double &high[], const double &low[],
                     const long &volume[]) {
      ArrayCopy(g_close, close);
      ArrayCopy(g_open, open);
      ArrayCopy(g_high, high);
      ArrayCopy(g_low, low);
      ArrayCopy(g_volume, volume);
      g_bars = bars;
      return g_featureBuilder.BuildFeatures(features, num_features, bars);
   }
   
   bool BuildFeatures(double &features[], int num_features, int bars) {
      ArrayResize(features, num_features);
      ArrayInitialize(features, 0.0);
      
      int total = 0;
      for(int i = 0; i < m_num_calculators; i++) {
         if(m_calculators[i] == NULL) continue;
         int start = m_calculators[i].GetStartIndex();
         m_calculators[i].Calculate(features, start, bars);
         total += m_calculators[i].GetCount();
      }
      
      return total == num_features;
   }
};

// Global instance
CFeatureBuilder g_featureBuilder;

#endif
