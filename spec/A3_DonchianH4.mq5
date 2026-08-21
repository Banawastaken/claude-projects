//+------------------------------------------------------------------+
//|  A3_DonchianH4.mq5                                               |
//|  H4 Donchian breakout for XAUUSD on a FundedNext Stellar 2-Step  |
//|  account. Implements spec/A3_TRADING_SPEC.md exactly.            |
//|                                                                  |
//|  Indicator values are read from COMPLETED H4 bars only (shift 1  |
//|  and back). Reading the forming bar is the usual reason a live   |
//|  EA beats its own backtest and then fails in front of money.     |
//+------------------------------------------------------------------+
#property copyright "Backtested strategy A3"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//--- strategy inputs (defaults are the backtested values) -----------
input int    InpChannel        = 20;      // Donchian lookback, H4 bars
input double InpAtrMult        = 1.0;     // stop = this x H4 ATR(14)
input double InpTpR            = 4.0;     // take profit in R
input int    InpAdxMin         = 15;      // minimum H4 ADX(14)
input double InpRiskPct        = 0.75;    // % of balance risked per trade
input double InpPhaseBoost     = 1.35;    // risk multiplier during phases 1-2
input bool   InpIsChallenge    = true;    // false once the account is funded

//--- management ------------------------------------------------------
input double InpBeAtR          = 1.2;     // move to break-even at this R
input double InpBeLockR        = 0.15;    // lock in this much R
input double InpTrailStartR    = 2.0;     // start trailing at this R
input double InpTrailDistR     = 1.5;     // trail this far behind price

//--- gates -----------------------------------------------------------
input int    InpSessionStart   = 1;       // UTC hour, inclusive
input int    InpSessionEnd     = 21;      // UTC hour, exclusive
input int    InpMaxTradesDay   = 1;
input int    InpMaxLossesDay   = 1;
input double InpDailyStopPct   = 2.0;     // halt for the day after this loss
input int    InpMaxConsecLoss  = 4;       // then stand down for the day
input double InpMedianSpread   = 0.62;    // instrument median, price units
input double InpMaxSpreadMult  = 1.8;
input double InpMaxTradeRisk   = 1.25;    // % of equity a single trade may risk

//--- account rules ---------------------------------------------------
input double InpStartBalance   = 6000.0;  // the account's initial balance
input double InpMaxLossPct     = 10.0;    // static, measured from the above
input double InpDailyLossPct   = 5.0;

input ulong  InpMagic          = 20260821;
input int    InpSlippagePoints = 30;

CTrade        trade;
CPositionInfo pos;

int    hAtr = INVALID_HANDLE;
int    hAdx = INVALID_HANDLE;

datetime curDay        = 0;
double   dayStartBal   = 0.0;
int      tradesToday   = 0;
int      lossesToday   = 0;
int      consecLosses  = 0;
bool     cooloffToday  = false;
double   beDoneTicket  = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);

   hAtr = iATR(_Symbol, PERIOD_H4, 14);
   hAdx = iADX(_Symbol, PERIOD_H4, 14);
   if(hAtr == INVALID_HANDLE || hAdx == INVALID_HANDLE)
   {
      Print("A3: failed to create H4 ATR/ADX handles");
      return INIT_FAILED;
   }

   dayStartBal = AccountInfoDouble(ACCOUNT_BALANCE);
   curDay      = ServerDay(TimeCurrent());
   PrintFormat("A3 started. channel=%d atrMult=%.2f tpR=%.1f risk=%.2f%%%s",
               InpChannel, InpAtrMult, InpTpR, InpRiskPct,
               InpIsChallenge ? " (challenge boost on)" : "");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(hAtr != INVALID_HANDLE) IndicatorRelease(hAtr);
   if(hAdx != INVALID_HANDLE) IndicatorRelease(hAdx);
}

//+------------------------------------------------------------------+
//| Day bucket. The broker's day rolls at server midnight, which is   |
//| what the daily loss limit resets on.                              |
//+------------------------------------------------------------------+
datetime ServerDay(datetime t)
{
   MqlDateTime s; TimeToStruct(t, s);
   s.hour = 0; s.min = 0; s.sec = 0;
   return StructToTime(s);
}

void RollDayIfNeeded()
{
   datetime d = ServerDay(TimeCurrent());
   if(d == curDay) return;
   curDay       = d;
   dayStartBal  = AccountInfoDouble(ACCOUNT_BALANCE);
   tradesToday  = 0;
   lossesToday  = 0;
   cooloffToday = false;   // a cool-off lasts to the end of the day only
}

//+------------------------------------------------------------------+
//| Count today's closed trades from history, so the EA survives a    |
//| restart mid-session without forgetting its own caps.              |
//+------------------------------------------------------------------+
void SyncTodayFromHistory()
{
   tradesToday  = 0;
   lossesToday  = 0;
   datetime from = curDay;
   if(!HistorySelect(from, TimeCurrent())) return;

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagic) continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                    + HistoryDealGetDouble(ticket, DEAL_SWAP)
                    + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      tradesToday++;
      if(profit < 0) lossesToday++;
   }
}

//+------------------------------------------------------------------+
double RealisedToday()
{
   double sum = 0.0;
   if(!HistorySelect(curDay, TimeCurrent())) return 0.0;
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagic) continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
      sum += HistoryDealGetDouble(ticket, DEAL_PROFIT)
           + HistoryDealGetDouble(ticket, DEAL_SWAP)
           + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
   }
   return sum;
}

//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(pos.SelectByIndex(i))
         if(pos.Symbol() == _Symbol && pos.Magic() == InpMagic)
            return true;
   return false;
}

//+------------------------------------------------------------------+
//| Highest high / lowest low of the previous InpChannel COMPLETED    |
//| H4 bars. Shift starts at 1: bar 0 is still forming.               |
//+------------------------------------------------------------------+
bool ChannelLevels(double &chHigh, double &chLow)
{
   int need = InpChannel;
   double highs[], lows[];
   if(CopyHigh(_Symbol, PERIOD_H4, 1, need, highs) != need) return false;
   if(CopyLow(_Symbol, PERIOD_H4, 1, need, lows)  != need) return false;
   chHigh = highs[ArrayMaximum(highs)];
   chLow  = lows[ArrayMinimum(lows)];
   return true;
}

bool CompletedH4(int handle, double &value)
{
   double buf[];
   if(CopyBuffer(handle, 0, 1, 1, buf) != 1) return false;
   value = buf[0];
   return true;
}

//+------------------------------------------------------------------+
double NormalizeLots(double lots)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minL = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxL = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0) step = 0.01;
   lots = MathRound(lots / step) * step;
   lots = MathMax(lots, minL);
   lots = MathMin(lots, maxL);
   return NormalizeDouble(lots, 2);
}

//--- money value of one price unit for one lot ----------------------
double UsdPerPricePerLot()
{
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickSize <= 0.0) return 100.0;         // gold fallback: $100 per $1
   return tickValue / tickSize;
}

//+------------------------------------------------------------------+
void TryEntry()
{
   if(HasOpenPosition()) return;

   //--- gate: session and weekday
   MqlDateTime s; TimeToStruct(TimeCurrent(), s);
   if(s.day_of_week == 0 || s.day_of_week == 6) return;
   if(s.hour < InpSessionStart || s.hour >= InpSessionEnd) return;

   //--- gate: daily caps
   if(tradesToday >= InpMaxTradesDay) return;
   if(lossesToday >= InpMaxLossesDay) return;
   if(cooloffToday) return;
   if(consecLosses >= InpMaxConsecLoss) { cooloffToday = true; return; }

   //--- gate: daily stop
   if(dayStartBal > 0.0 &&
      RealisedToday() <= -(InpDailyStopPct / 100.0) * dayStartBal) return;

   //--- gate: spread spike
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   if(spread > InpMaxSpreadMult * InpMedianSpread) return;

   //--- indicators from completed bars only
   double atr, adx, chHigh, chLow;
   if(!CompletedH4(hAtr, atr) || !CompletedH4(hAdx, adx)) return;
   if(!ChannelLevels(chHigh, chLow)) return;
   if(atr <= 0.0) return;
   if(adx < InpAdxMin) return;

   int dir = 0;
   if(bid > chHigh)      dir =  1;
   else if(bid < chLow)  dir = -1;
   if(dir == 0) return;

   //--- stop distance
   double slDist = InpAtrMult * atr;
   slDist = MathMax(slDist, 0.5 * atr);
   slDist = MathMin(slDist, 3.0 * atr);
   if(slDist <= 0.0) return;

   //--- sizing
   double balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskPct  = InpRiskPct / 100.0;
   if(InpIsChallenge) riskPct *= InpPhaseBoost;
   double riskUsd  = balance * riskPct;

   double perPrice = UsdPerPricePerLot();
   double minLot   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   //--- gate: is the account even big enough to size this trade?
   double minRisk = minLot * slDist * perPrice;
   if(minRisk > (InpMaxTradeRisk / 100.0) * equity)
   {
      PrintFormat("A3: skipping, min lot risks %.2f (%.2f%% of equity), stop %.2f",
                  minRisk, 100.0 * minRisk / equity, slDist);
      return;
   }

   //--- gate: keep clear of the daily and static limits
   double dailyFloor = dayStartBal * (1.0 - InpDailyLossPct / 100.0);
   double staticFloor = InpStartBalance * (1.0 - InpMaxLossPct / 100.0);
   double roomDaily  = MathMax(0.0, equity - dailyFloor) * 0.80;
   double roomTotal  = MathMax(0.0, equity - staticFloor) * 0.80;
   riskUsd = MathMin(riskUsd, MathMin(roomDaily, roomTotal));
   if(riskUsd <= 0.0) return;

   double lots = NormalizeLots(riskUsd / (slDist * perPrice));
   if(lots * slDist * perPrice > (InpMaxTradeRisk / 100.0) * equity)
   {
      double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      lots = NormalizeDouble(MathFloor((riskUsd / (slDist * perPrice)) / step) * step, 2);
   }
   if(lots < minLot) return;

   //--- place it
   double entry = (dir > 0) ? ask : bid;
   double sl = (dir > 0) ? entry - slDist : entry + slDist;
   double tp = (dir > 0) ? entry + InpTpR * slDist : entry - InpTpR * slDist;

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   bool ok = (dir > 0)
      ? trade.Buy(lots, _Symbol, 0.0, sl, tp, "A3 donch long")
      : trade.Sell(lots, _Symbol, 0.0, sl, tp, "A3 donch short");

   if(ok)
   {
      tradesToday++;
      PrintFormat("A3: %s %.2f lots, stop %.2f (%.2f), risk $%.2f, ADX %.1f",
                  dir > 0 ? "BUY" : "SELL", lots, sl, slDist,
                  lots * slDist * perPrice, adx);
   }
   else
      PrintFormat("A3: order failed, retcode %d", trade.ResultRetcode());
}

//+------------------------------------------------------------------+
void ManageOpen()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != InpMagic) continue;

      double entry = pos.PriceOpen();
      double sl    = pos.StopLoss();
      double tp    = pos.TakeProfit();
      bool   isBuy = (pos.PositionType() == POSITION_TYPE_BUY);

      // R is the ORIGINAL stop distance. Once the stop has been moved this
      // cannot be recovered from the position, so it is reconstructed from
      // the take-profit, which never moves: TP is always entry +/- tpR * R.
      double r = MathAbs(tp - entry) / InpTpR;
      if(r <= 0.0) continue;

      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double move = isBuy ? (bid - entry) : (entry - ask);
      double rMult = move / r;

      double newSl = sl;

      if(rMult >= InpBeAtR)
      {
         double be = isBuy ? entry + InpBeLockR * r : entry - InpBeLockR * r;
         newSl = isBuy ? MathMax(newSl, be) : MathMin(newSl, be);
      }
      if(rMult >= InpTrailStartR)
      {
         double tr = isBuy ? bid - InpTrailDistR * r : ask + InpTrailDistR * r;
         newSl = isBuy ? MathMax(newSl, tr) : MathMin(newSl, tr);
      }

      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      newSl = NormalizeDouble(newSl, digits);

      // never widen, and respect the broker's stop distance
      double stopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL)
                       * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      if(isBuy  && newSl > bid - stopLevel) newSl = bid - stopLevel;
      if(!isBuy && newSl < ask + stopLevel) newSl = ask + stopLevel;

      bool improves = isBuy ? (newSl > sl + _Point) : (newSl < sl - _Point);
      if(improves)
         trade.PositionModify(pos.Ticket(), newSl, tp);
   }
}

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagic) return;
   if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;

   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                 + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
                 + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   if(profit < 0.0) { lossesToday++; consecLosses++; }
   else             { consecLosses = 0; }

   PrintFormat("A3: closed for %.2f, consecutive losses now %d", profit, consecLosses);
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime before = curDay;
   RollDayIfNeeded();
   if(curDay != before) SyncTodayFromHistory();

   ManageOpen();
   TryEntry();
}
//+------------------------------------------------------------------+
