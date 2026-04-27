//+------------------------------------------------------------------+
//| BacktestReport.mqh - HTML Report Generator                       |
//| Peter - Version 1.03 (Safest String Handling)                    |
//+------------------------------------------------------------------+
#ifndef BACKTEST_REPORT_MQH
#define BACKTEST_REPORT_MQH

#include "BacktestMetrics.mqh"

//+------------------------------------------------------------------+
//| Backtest Report Generator                                        |
//+------------------------------------------------------------------+
class CBacktestReport {
private:
   string          m_symbol;
   ENUM_TIMEFRAMES m_period;
   double          m_initial_deposit;
   
public:
   CBacktestReport() {
      m_symbol = "";
      m_period = PERIOD_CURRENT;
      m_initial_deposit = 10000.0;
   }
   
   void SetSymbol(const string symbol) { m_symbol = symbol; }
   void SetPeriod(ENUM_TIMEFRAMES period) { m_period = period; }
   void SetInitialDeposit(double deposit) { m_initial_deposit = deposit; }
   
   bool Generate(CBacktestMetrics &metrics, const string filename) {
      string path = filename;
      int file = FileOpen(path, FILE_WRITE | FILE_TXT);
      if(file == INVALID_HANDLE) {
         Print("[Report] ERROR: Cannot create report file");
         return false;
      }
      
      // Generate HTML line by line
      FileWriteString(file, "<!DOCTYPE html>\n");
      FileWriteString(file, "<html><head>\n");
      FileWriteString(file, "<title>GoldenGauss V5 - Backtest Report</title>\n");
      WriteCSS(file);
      FileWriteString(file, "</head><body>\n");
      
      // Header
      FileWriteString(file, "<h1>GoldenGauss V5 - Backtest Report</h1>\n");
      FileWriteString(file, "<p><strong>Symbol:</strong> ");
      FileWriteString(file, m_symbol);
      FileWriteString(file, "</p>\n");
      
      FileWriteString(file, "<p><strong>Period:</strong> ");
      FileWriteString(file, EnumToString(m_period));
      FileWriteString(file, "</p>\n");
      
      FileWriteString(file, "<p><strong>Generated:</strong> ");
      FileWriteString(file, TimeToString(TimeCurrent()));
      FileWriteString(file, "</p>\n");
      
      // Performance Summary
      FileWriteString(file, "<h2>Performance Summary</h2>\n");
      FileWriteString(file, "<div>\n");
      
      WriteMetricBox(file, "Initial Deposit", DoubleToString(metrics.GetInitialDeposit(), 2));
      WriteMetricBox(file, "Final Balance", DoubleToString(metrics.GetFinalBalance(), 2));
      
      double net_profit = metrics.GetTotalNetProfit();
      WriteMetricBox(file, "Net Profit", DoubleToString(net_profit, 2), net_profit >= 0);
      
      WriteMetricBox(file, "Total Trades", IntegerToString(metrics.GetTotalTrades()));
      
      FileWriteString(file, "</div>\n");
      
      // Key Metrics Table
      FileWriteString(file, "<h2>Key Metrics</h2>\n");
      FileWriteString(file, "<table>\n");
      FileWriteString(file, "<tr><th>Metric</th><th>Value</th></tr>\n");
      
      WriteTableRow(file, "Win Rate", DoubleToString(metrics.GetWinRate(), 2) + "%");
      WriteTableRow(file, "Profit Factor", DoubleToString(metrics.GetProfitFactor(), 2));
      WriteTableRow(file, "Recovery Factor", DoubleToString(metrics.GetRecoveryFactor(), 2));
      WriteTableRow(file, "Expectancy", DoubleToString(metrics.GetExpectancy(), 2));
      WriteTableRow(file, "Max Drawdown", DoubleToString(metrics.GetMaxDrawdown(), 2) + "%", false);
      WriteTableRow(file, "Sharpe Ratio", DoubleToString(metrics.GetSharpeRatio(), 2));
      WriteTableRow(file, "Sortino Ratio", DoubleToString(metrics.GetSortinoRatio(), 2));
      WriteTableRow(file, "Calmar Ratio", DoubleToString(metrics.GetCalmarRatio(), 2));
      WriteTableRow(file, "Sterling Ratio", DoubleToString(metrics.GetSterlingRatio(), 2));
      
      FileWriteString(file, "</table>\n");
      
      // Trade Statistics
      FileWriteString(file, "<h2>Trade Statistics</h2>\n");
      FileWriteString(file, "<table>\n");
      FileWriteString(file, "<tr><th>Statistic</th><th>Value</th></tr>\n");
      
      WriteTableRow(file, "Winning Trades", IntegerToString(metrics.GetWinningTrades()), true);
      WriteTableRow(file, "Losing Trades", IntegerToString(metrics.GetLosingTrades()), false);
      WriteTableRow(file, "Average Win", DoubleToString(metrics.GetAvgWin(), 2), true);
      WriteTableRow(file, "Average Loss", DoubleToString(metrics.GetAvgLoss(), 2), false);
      WriteTableRow(file, "Payoff Ratio", DoubleToString(metrics.GetPayoffRatio(), 2));
      WriteTableRow(file, "Largest Win", DoubleToString(metrics.GetLargestWin(), 2), true);
      WriteTableRow(file, "Largest Loss", DoubleToString(metrics.GetLargestLoss(), 2), false);
      WriteTableRow(file, "Total Commission", DoubleToString(metrics.GetTotalCommission(), 2), false);
      
      FileWriteString(file, "</table>\n");
      
      // Footer
      FileWriteString(file, "<hr>\n");
      FileWriteString(file, "<p><em>Generated by GoldenGauss V5 Backtesting Framework</em></p>\n");
      FileWriteString(file, "<p><strong>Disclaimer:</strong> Past performance does not guarantee future results.</p>\n");
      FileWriteString(file, "</body></html>\n");
      
      FileClose(file);
      
      Print("[Report] HTML report generated: ", path);
      return true;
   }
   
private:
   void WriteCSS(int file) {
      FileWriteString(file, "<style>\n");
      FileWriteString(file, "body { font-family: Arial, sans-serif; margin: 20px; }\n");
      FileWriteString(file, "h1 { color: #2c3e50; }\n");
      FileWriteString(file, "h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }\n");
      FileWriteString(file, "table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
      FileWriteString(file, "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }\n");
      FileWriteString(file, "th { background-color: #3498db; color: white; }\n");
      FileWriteString(file, "tr:nth-child(even) { background-color: #f2f2f2; }\n");
      FileWriteString(file, ".positive { color: #27ae60; font-weight: bold; }\n");
      FileWriteString(file, ".negative { color: #e74c3c; font-weight: bold; }\n");
      FileWriteString(file, ".metric-box { display: inline-block; width: 200px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; text-align: center; }\n");
      FileWriteString(file, ".metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }\n");
      FileWriteString(file, ".metric-label { font-size: 12px; color: #7f8c8d; margin-top: 5px; }\n");
      FileWriteString(file, "</style>\n");
   }
   
   void WriteMetricBox(int file, const string label, const string value, bool positive = true) {
      FileWriteString(file, "<div class='metric-box'>\n");
      
      FileWriteString(file, "<div class='metric-value' style='color: ");
      if(positive) {
         FileWriteString(file, "#27ae60");
      } else {
         FileWriteString(file, "#e74c3c");
      }
      FileWriteString(file, "'>");
      
      FileWriteString(file, value);
      FileWriteString(file, "</div>\n");
      
      FileWriteString(file, "<div class='metric-label'>");
      FileWriteString(file, label);
      FileWriteString(file, "</div>\n");
      
      FileWriteString(file, "</div>\n");
   }
   
   void WriteTableRow(int file, const string label, const string value, bool positive = true) {
      FileWriteString(file, "<tr><td>");
      FileWriteString(file, label);
      FileWriteString(file, "</td><td class='");
      
      if(positive) {
         FileWriteString(file, "positive");
      } else {
         FileWriteString(file, "negative");
      }
      
      FileWriteString(file, "'>");
      FileWriteString(file, value);
      FileWriteString(file, "</td></tr>\n");
   }
};

#endif
