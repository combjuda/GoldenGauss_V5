//+------------------------------------------------------------------+
//| TradeManager.mqh - Trading Logic & Position Management           |
//| Peter 2026-04-26                                                 |
//+------------------------------------------------------------------+
#ifndef GOLDENGAUSS_TRADE_MANAGER_MQH
#define GOLDENGAUSS_TRADE_MANAGER_MQH

#include "../Core/Types.mqh"
#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| Trade Manager                                                    |
//+------------------------------------------------------------------+
class CTradeManager {
private:
   CTrade     m_trade;
   string     m_symbol;
   double     m_lot_size;
   int        m_max_positions;
   datetime   m_last_trade;
   ulong      m_magic;

public:
   CTradeManager() {
      m_symbol = _Symbol;
      m_lot_size = 0.01;
      m_max_positions = 1;
      m_last_trade = 0;
      m_magic = 20260426;
   }

   void SetMagic(ulong magic) { m_magic = magic; }
   void SetLotSize(double lot) { m_lot_size = lot; }
   void SetMaxPositions(int max) { m_max_positions = max; }

   //+----------------------------------------------------------+
   //| Get open positions count                                   |
   //+----------------------------------------------------------+
   int GetOpenPositions() {
      int count = 0;
      for(int i = PositionsTotal() - 1; i >= 0; i--) {
         if(PositionGetSymbol(i) == m_symbol && 
            PositionGetInteger(POSITION_MAGIC) == m_magic)
            count++;
      }
      return count;
   }

   //+----------------------------------------------------------+
   //| Check if we can trade                                     |
   //+----------------------------------------------------------+
   bool CanTrade() {
      if(GetOpenPositions() >= m_max_positions)
         return false;
      if(TimeCurrent() - m_last_trade < 30)
         return false;
      return true;
   }

   //+----------------------------------------------------------+
   //| Open BUY with ATR-based stops                              |
   //+----------------------------------------------------------+
   bool OpenBuyATR(double probability, double sl_pips, double tp_pips, ulong &ticket) {
      ticket = 0;
      if(!CanTrade())
         return false;
      
      double price = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
      double sl = sl_pips * _Point;
      double tp = tp_pips * _Point;
      
      m_trade.SetExpertMagicNumber(m_magic);
      if(!m_trade.Buy(m_lot_size, m_symbol, price, price - sl, price + tp, "GGV4_Buy")) {
         Print("[TRADE] Buy failed: ", GetLastError());
         return false;
      }
      
      ticket = m_trade.ResultOrder();
      m_last_trade = TimeCurrent();
      Print("[TRADE] BUY | Tkt: ", ticket, " | Prob: ", DoubleToString(probability, 4),
            " | SL: ", sl_pips, " | TP: ", tp_pips);
      return true;
   }

   //+----------------------------------------------------------+
   //| Open SELL with ATR-based stops                            |
   //+----------------------------------------------------------+
   bool OpenSellATR(double probability, double sl_pips, double tp_pips, ulong &ticket) {
      ticket = 0;
      if(!CanTrade())
         return false;
      
      double price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
      double sl = sl_pips * _Point;
      double tp = tp_pips * _Point;
      
      m_trade.SetExpertMagicNumber(m_magic);
      if(!m_trade.Sell(m_lot_size, m_symbol, price, price + sl, price - tp, "GGV4_Sell")) {
         Print("[TRADE] Sell failed: ", GetLastError());
         return false;
      }
      
      ticket = m_trade.ResultOrder();
      m_last_trade = TimeCurrent();
      Print("[TRADE] SELL | Tkt: ", ticket, " | Prob: ", DoubleToString(probability, 4),
            " | SL: ", sl_pips, " | TP: ", tp_pips);
      return true;
   }

   //+----------------------------------------------------------+
   //| Close position by ticket                                  |
   //+----------------------------------------------------------+
   bool ClosePosition(ulong ticket) {
      if(!m_trade.PositionClose(ticket)) {
         Print("[TRADE] Close failed: ", GetLastError());
         return false;
      }
      return true;
   }

   //+----------------------------------------------------------+
   //| Close all positions                                       |
   //+----------------------------------------------------------+
   void CloseAll() {
      for(int i = PositionsTotal() - 1; i >= 0; i--) {
         ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
         if(PositionGetSymbol(i) == m_symbol && 
            PositionGetInteger(POSITION_MAGIC) == m_magic)
            ClosePosition(ticket);
      }
   }

   //+----------------------------------------------------------+
   //| Trailing stop                                             |
   //+----------------------------------------------------------+
   void ManageTrailingStop(ulong ticket, double trail_distance) {
      if(!PositionSelectByTicket((long)ticket))
         return;
      
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      double bid = SymbolInfoDouble(m_symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
      
      double min_stop = SymbolInfoInteger(m_symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
      if(min_stop <= 0) min_stop = 50 * _Point;
      
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      
      if(type == POSITION_TYPE_BUY) {
         double new_sl = bid - trail_distance * _Point;
         if(new_sl <= bid - min_stop && new_sl > current_sl && new_sl < current_tp) {
            m_trade.PositionModify(ticket, new_sl, current_tp);
         }
      } else {
         double new_sl = ask + trail_distance * _Point;
         if(new_sl >= ask + min_stop && (new_sl < current_sl || current_sl == 0) && new_sl > current_tp) {
            m_trade.PositionModify(ticket, new_sl, current_tp);
         }
      }
   }
};

//+------------------------------------------------------------------+
//| Signal Generator                                                  |
//+------------------------------------------------------------------+
class CSignalGenerator {
private:
   double m_prob_threshold;
   double m_strong_threshold;

public:
   CSignalGenerator(double weak = 0.5, double strong = 0.8) {
      m_prob_threshold = weak;
      m_strong_threshold = strong;
   }

   ENUM_TRADE_SIGNAL GetSignal(const SPrediction &pred) {
      if(!pred.is_valid || pred.predicted_class == 0)
         return SIGNAL_NONE;
      
      double thr = (pred.confidence > 0.2) ? m_prob_threshold : m_strong_threshold;
      
      if(pred.predicted_class == 1 && pred.probability >= thr)
         return (pred.probability >= m_strong_threshold) ? SIGNAL_BUY_STRONG : SIGNAL_BUY;
      if(pred.predicted_class == 2 && pred.probability >= thr)
         return (pred.probability >= m_strong_threshold) ? SIGNAL_SELL_STRONG : SIGNAL_SELL;
      
      return SIGNAL_NONE;
   }
};

#endif
