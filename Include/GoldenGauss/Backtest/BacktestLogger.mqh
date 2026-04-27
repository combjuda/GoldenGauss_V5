//+------------------------------------------------------------------+
//| BacktestLogger.mqh - Trade Logging & Export                      |
//| Peter - Version 1.02 (Fixed FileWriteDouble)                     |
//+------------------------------------------------------------------+
#ifndef BACKTEST_LOGGER_MQH
#define BACKTEST_LOGGER_MQH

#include "BacktestMetrics.mqh"

//+------------------------------------------------------------------+
//| Backtest Logger                                                  |
//+------------------------------------------------------------------+
class CBacktestLogger {
private:
   int             m_file_handle;
   string          m_filename;
   bool            m_initialized;
   double          m_cumulative_profit;
   
   // Trade storage for completing open->close
   struct SOpenTrade {
      ulong    ticket;
      ENUM_ORDER_TYPE type;
      double   entry_price;
      double   lot_size;
      datetime open_time;
      double   probability;
   };
   SOpenTrade m_open_trades[];
   int        m_open_count;
   
public:
   CBacktestLogger() {
      m_initialized = false;
      m_file_handle = INVALID_HANDLE;
      m_cumulative_profit = 0;
      m_open_count = 0;
      ArrayResize(m_open_trades, 1000);
   }
   
   bool Initialize(const string filename) {
      m_filename = "MQL5/Files/Backtest/Results/" + filename;
      
      m_file_handle = FileOpen(m_filename, FILE_WRITE | FILE_CSV | FILE_ANSI);
      if(m_file_handle == INVALID_HANDLE) {
         Print("[Logger] ERROR: Cannot create log file: ", m_filename);
         return false;
      }
      
      // Write CSV header
      FileWriteString(m_file_handle, "Ticket;Type;EntryPrice;ExitPrice;LotSize;GrossProfit;Commission;Slippage;NetProfit;OpenTime;CloseTime;BarsHeld;Probability;CumulativeProfit\n");
      
      m_initialized = true;
      Print("[Logger] Log file created: ", m_filename);
      return true;
   }
   
   void LogTradeOpen(ulong ticket, ENUM_ORDER_TYPE type, double entry_price,
                     double lot_size, double sl, double tp, 
                     datetime open_time, double probability) {
      if(!m_initialized) return;
      
      // Store open trade for later completion
      if(m_open_count >= ArraySize(m_open_trades))
         ArrayResize(m_open_trades, ArraySize(m_open_trades) + 1000);
      
      m_open_trades[m_open_count].ticket = ticket;
      m_open_trades[m_open_count].type = type;
      m_open_trades[m_open_count].entry_price = entry_price;
      m_open_trades[m_open_count].lot_size = lot_size;
      m_open_trades[m_open_count].open_time = open_time;
      m_open_trades[m_open_count].probability = probability;
      m_open_count++;
   }
   
   void LogTradeClose(ulong ticket, double net_profit, double commission, 
                      datetime close_time) {
      if(!m_initialized) return;
      
      // Find matching open trade
      int trade_idx = -1;
      for(int i = 0; i < m_open_count; i++) {
         if(m_open_trades[i].ticket == ticket) {
            trade_idx = i;
            break;
         }
      }
      
      if(trade_idx < 0) {
         Print("[Logger] Warning: No open trade found for ticket ", ticket);
         return;
      }
      
      // Update cumulative profit
      m_cumulative_profit += net_profit;
      
      // ✅ FIXED: Write each field separately with proper formatting
      FileWriteInteger(m_file_handle, (int)ticket, INT_VALUE);
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, (m_open_trades[trade_idx].type == ORDER_TYPE_BUY) ? "BUY" : "SELL");
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(m_open_trades[trade_idx].entry_price, _Digits));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(0.0, _Digits));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(m_open_trades[trade_idx].lot_size, 2));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(0.0, 2));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(commission, 2));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(0.0, 2));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(net_profit, 2));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, TimeToString(m_open_trades[trade_idx].open_time));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, TimeToString(close_time));
      FileWriteString(m_file_handle, ";");
      FileWriteInteger(m_file_handle, 0, INT_VALUE);
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(m_open_trades[trade_idx].probability, 2));
      FileWriteString(m_file_handle, ";");
      FileWriteString(m_file_handle, DoubleToString(m_cumulative_profit, 2));
      FileWriteString(m_file_handle, "\n");
      
      // Remove from open trades array (swap with last and decrement)
      if(trade_idx < m_open_count - 1) {
         m_open_trades[trade_idx] = m_open_trades[m_open_count - 1];
      }
      m_open_count--;
   }
   
   void Close() {
      if(m_file_handle != INVALID_HANDLE) {
         FileClose(m_file_handle);
         Print("[Logger] Log file closed: ", m_filename);
      }
      m_initialized = false;
   }
   
   double GetCumulativeProfit() const { return m_cumulative_profit; }
   int GetOpenTradeCount() const { return m_open_count; }
};

#endif
