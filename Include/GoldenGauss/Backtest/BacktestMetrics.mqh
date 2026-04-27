//+------------------------------------------------------------------+
//| BacktestMetrics.mqh - Professional Performance Metrics           |
//| Peter - Version 1.00                                             |
//+------------------------------------------------------------------+
#ifndef BACKTEST_METRICS_MQH
#define BACKTEST_METRICS_MQH

//+------------------------------------------------------------------+
//| Trade Record Structure                                           |
//+------------------------------------------------------------------+
struct STradeRecord {
   ulong    ticket;
   ENUM_ORDER_TYPE type;
   double   entry_price;
   double   exit_price;
   double   lot_size;
   double   gross_profit;
   double   net_profit;
   double   commission;
   double   slippage_cost;
   datetime open_time;
   datetime close_time;
   int      bars_held;
   double   max_favorable;
   double   max_adverse;
   double   drawdown_contribution;
};

//+------------------------------------------------------------------+
//| Equity Curve Point                                               |
//+------------------------------------------------------------------+
struct SEquityPoint {
   datetime time;
   double   equity;
   double   balance;
   double   drawdown;
};

//+------------------------------------------------------------------+
//| Backtest Metrics Calculator                                      |
//+------------------------------------------------------------------+
class CBacktestMetrics {
private:
   double          m_initial_deposit;
   double          m_final_balance;
   double          m_final_equity;
   double          m_total_gross_profit;
   double          m_total_gross_loss;
   double          m_total_net_profit;
   double          m_total_commission;
   double          m_total_slippage;
   int             m_total_trades;
   int             m_winning_trades;
   int             m_losing_trades;
   double          m_max_drawdown;
   double          m_max_drawdown_amount;
   datetime        m_drawdown_start;
   datetime        m_drawdown_end;
   int             m_max_drawdown_bars;
   double          m_peak_equity;
   double          m_avg_win;
   double          m_avg_loss;
   double          m_largest_win;
   double          m_largest_loss;
   double          m_avg_bars_held;
   double          m_profit_factor;
   double          m_recovery_factor;
   double          m_sharpe_ratio;
   double          m_sortino_ratio;
   double          m_calmar_ratio;
   double          m_sterling_ratio;
   double          m_expectancy;
   double          m_win_rate;
   double          m_payoff_ratio;
   
   STradeRecord    m_trades[];
   int             m_trade_count;
   SEquityPoint    m_equity_curve[];
   int             m_equity_count;
   
public:
   CBacktestMetrics() {
      Initialize(10000.0);
   }
   
   void Initialize(double initial_deposit) {
      m_initial_deposit = initial_deposit;
      m_final_balance = initial_deposit;
      m_final_equity = initial_deposit;
      m_total_gross_profit = 0;
      m_total_gross_loss = 0;
      m_total_net_profit = 0;
      m_total_commission = 0;
      m_total_slippage = 0;
      m_total_trades = 0;
      m_winning_trades = 0;
      m_losing_trades = 0;
      m_max_drawdown = 0;
      m_max_drawdown_amount = 0;
      m_peak_equity = initial_deposit;
      m_trade_count = 0;
      m_equity_count = 0;
      
      ArrayResize(m_trades, 10000);
      ArrayResize(m_equity_curve, 100000);
   }
   
   void RecordTradeOpen(ulong ticket, ENUM_ORDER_TYPE type, double entry_price,
                        double lot_size, double sl, double tp, datetime open_time) {
      // Store open trade info (will be completed on close)
   }
   
   void RecordTradeClose(ulong ticket, double net_profit, double commission, datetime close_time) {
      if(m_trade_count >= ArraySize(m_trades))
         ArrayResize(m_trades, ArraySize(m_trades) + 10000);
      
      m_trades[m_trade_count].ticket = ticket;
      m_trades[m_trade_count].net_profit = net_profit;
      m_trades[m_trade_count].commission = commission;
      m_trades[m_trade_count].close_time = close_time;
      
      if(net_profit > 0) {
         m_winning_trades++;
         m_total_gross_profit += net_profit;
         if(net_profit > m_largest_win) m_largest_win = net_profit;
      } else {
         m_losing_trades++;
         m_total_gross_loss += MathAbs(net_profit);
         if(net_profit < m_largest_loss) m_largest_loss = net_profit;
      }
      
      m_total_net_profit += net_profit;
      m_total_commission += commission;
      m_final_balance += net_profit;
      m_trade_count++;
   }
   
   void UpdateEquity(double equity, datetime time) {
      if(m_equity_count >= ArraySize(m_equity_curve))
         ArrayResize(m_equity_curve, ArraySize(m_equity_curve) + 100000);
      
      m_equity_curve[m_equity_count].time = time;
      m_equity_curve[m_equity_count].equity = equity;
      m_equity_curve[m_equity_count].balance = m_final_balance;
      
      if(equity > m_peak_equity)
         m_peak_equity = equity;
      
      double dd = (m_peak_equity - equity) / m_peak_equity * 100.0;
      m_equity_curve[m_equity_count].drawdown = dd;
      
      if(dd > m_max_drawdown) {
         m_max_drawdown = dd;
         m_max_drawdown_amount = m_peak_equity - equity;
      }
      
      m_equity_count++;
      m_final_equity = equity;
   }
   
   void CalculateFinalMetrics() {
      if(m_total_trades == 0) m_total_trades = m_trade_count;
      
      // Win rate
      m_win_rate = (m_total_trades > 0) ? (double)m_winning_trades / m_total_trades * 100.0 : 0;
      
      // Profit factor
      if(m_total_gross_loss > 0)
         m_profit_factor = m_total_gross_profit / m_total_gross_loss;
      else
         m_profit_factor = (m_total_gross_profit > 0) ? 999.99 : 0;
      
      // Average win/loss
      m_avg_win = (m_winning_trades > 0) ? m_total_gross_profit / m_winning_trades : 0;
      m_avg_loss = (m_losing_trades > 0) ? m_total_gross_loss / m_losing_trades : 0;
      
      // Payoff ratio
      m_payoff_ratio = (m_avg_loss > 0) ? m_avg_win / m_avg_loss : 0;
      
      // Expectancy
      m_expectancy = (m_total_trades > 0) ? m_total_net_profit / m_total_trades : 0;
      
      // Recovery factor
      if(m_max_drawdown_amount > 0)
         m_recovery_factor = m_total_net_profit / m_max_drawdown_amount;
      else
         m_recovery_factor = 0;
      
      // Calmar ratio (annualized return / max drawdown)
      if(m_max_drawdown > 0)
         m_calmar_ratio = (m_total_net_profit / m_initial_deposit * 100.0) / m_max_drawdown;
      else
         m_calmar_ratio = 0;
      
      // Sharpe ratio (simplified - daily returns)
      m_sharpe_ratio = CalculateSharpeRatio();
      
      // Sortino ratio
      m_sortino_ratio = CalculateSortinoRatio();
      
      // Sterling ratio
      m_sterling_ratio = CalculateSterlingRatio();
      
      // Average bars held
      m_avg_bars_held = CalculateAvgBarsHeld();
   }
   
   double CalculateSharpeRatio() {
      if(m_equity_count < 2) return 0;
      
      double returns[];
      ArrayResize(returns, m_equity_count - 1);
      
      for(int i = 1; i < m_equity_count; i++) {
         returns[i-1] = (m_equity_curve[i].equity - m_equity_curve[i-1].equity) / 
                        m_equity_curve[i-1].equity;
      }
      
      double avg_return = 0;
      for(int i = 0; i < ArraySize(returns); i++)
         avg_return += returns[i];
      avg_return /= ArraySize(returns);
      
      double variance = 0;
      for(int i = 0; i < ArraySize(returns); i++)
         variance += MathPow(returns[i] - avg_return, 2);
      double std_dev = MathSqrt(variance / ArraySize(returns));
      
      if(std_dev == 0) return 0;
      
      // Annualize (assuming 252 trading days)
      double sharpe = (avg_return / std_dev) * MathSqrt(252);
      
      return sharpe;
   }
   
   double CalculateSortinoRatio() {
      if(m_equity_count < 2) return 0;
      
      double returns[];
      ArrayResize(returns, m_equity_count - 1);
      
      for(int i = 1; i < m_equity_count; i++) {
         returns[i-1] = (m_equity_curve[i].equity - m_equity_curve[i-1].equity) / 
                        m_equity_curve[i-1].equity;
      }
      
      double avg_return = 0;
      for(int i = 0; i < ArraySize(returns); i++)
         avg_return += returns[i];
      avg_return /= ArraySize(returns);
      
      // Downside deviation (only negative returns)
      double downside_sum = 0;
      int downside_count = 0;
      for(int i = 0; i < ArraySize(returns); i++) {
         if(returns[i] < 0) {
            downside_sum += MathPow(returns[i], 2);
            downside_count++;
         }
      }
      
      if(downside_count == 0) return 0;
      
      double downside_dev = MathSqrt(downside_sum / downside_count);
      
      if(downside_dev == 0) return 0;
      
      double sortino = (avg_return / downside_dev) * MathSqrt(252);
      
      return sortino;
   }
   
   double CalculateSterlingRatio() {
      if(m_max_drawdown <= 0) return 0;
      
      // Sterling = Annualized Return / (Max Drawdown + 10%)
      double annual_return = m_total_net_profit / m_initial_deposit * 100.0;
      double sterling = annual_return / (m_max_drawdown + 10.0);
      
      return sterling;
   }
   
   double CalculateAvgBarsHeld() {
      if(m_trade_count == 0) return 0;
      
      // Simplified - would need open/close bar tracking
      return 0;
   }
   
   void PrintSummary() {
      Print("==================================================");
      Print("           BACKTEST PERFORMANCE SUMMARY           ");
      Print("==================================================");
      Print("Initial Deposit:     $", DoubleToString(m_initial_deposit, 2));
      Print("Final Balance:       $", DoubleToString(m_final_balance, 2));
      Print("Total Net Profit:    $", DoubleToString(m_total_net_profit, 2));
      Print("Total Commission:    $", DoubleToString(m_total_commission, 2));
      Print("");
      Print("Total Trades:        ", m_total_trades);
      Print("Winning Trades:      ", m_winning_trades, " (", DoubleToString(m_win_rate, 1), "%)");
      Print("Losing Trades:       ", m_losing_trades);
      Print("");
      Print("Profit Factor:       ", DoubleToString(m_profit_factor, 2));
      Print("Recovery Factor:     ", DoubleToString(m_recovery_factor, 2));
      Print("Expectancy:          $", DoubleToString(m_expectancy, 2));
      Print("");
      Print("Max Drawdown:        ", DoubleToString(m_max_drawdown, 2), "%");
      Print("Max DD Amount:       $", DoubleToString(m_max_drawdown_amount, 2));
      Print("");
      Print("Sharpe Ratio:        ", DoubleToString(m_sharpe_ratio, 2));
      Print("Sortino Ratio:       ", DoubleToString(m_sortino_ratio, 2));
      Print("Calmar Ratio:        ", DoubleToString(m_calmar_ratio, 2));
      Print("Sterling Ratio:      ", DoubleToString(m_sterling_ratio, 2));
      Print("");
      Print("Average Win:         $", DoubleToString(m_avg_win, 2));
      Print("Average Loss:        $", DoubleToString(m_avg_loss, 2));
      Print("Payoff Ratio:        ", DoubleToString(m_payoff_ratio, 2));
      Print("Largest Win:         $", DoubleToString(m_largest_win, 2));
      Print("Largest Loss:        $", DoubleToString(m_largest_loss, 2));
      Print("==================================================");
   }
   
   // Getters for report generation
   double GetInitialDeposit() const { return m_initial_deposit; }
   double GetFinalBalance() const { return m_final_balance; }
   double GetTotalNetProfit() const { return m_total_net_profit; }
   double GetTotalCommission() const { return m_total_commission; }
   int GetTotalTrades() const { return m_total_trades; }
   int GetWinningTrades() const { return m_winning_trades; }
   int GetLosingTrades() const { return m_losing_trades; }
   double GetWinRate() const { return m_win_rate; }
   double GetProfitFactor() const { return m_profit_factor; }
   double GetRecoveryFactor() const { return m_recovery_factor; }
   double GetExpectancy() const { return m_expectancy; }
   double GetMaxDrawdown() const { return m_max_drawdown; }
   double GetMaxDrawdownAmount() const { return m_max_drawdown_amount; }
   double GetSharpeRatio() const { return m_sharpe_ratio; }
   double GetSortinoRatio() const { return m_sortino_ratio; }
   double GetCalmarRatio() const { return m_calmar_ratio; }
   double GetSterlingRatio() const { return m_sterling_ratio; }
   double GetAvgWin() const { return m_avg_win; }
   double GetAvgLoss() const { return m_avg_loss; }
   double GetPayoffRatio() const { return m_payoff_ratio; }
   double GetLargestWin() const { return m_largest_win; }
   double GetLargestLoss() const { return m_largest_loss; }
   
   STradeRecord GetTrade(int index) {
      if(index >= 0 && index < m_trade_count)
         return m_trades[index];
      STradeRecord empty;
      ZeroMemory(empty);
      return empty;
   }
   
   int GetTradeCount() const { return m_trade_count; }
};

#endif
