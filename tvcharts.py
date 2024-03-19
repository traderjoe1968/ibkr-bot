import asyncio
import io
import logging
from datetime import datetime
from typing import Final, List, Literal, Optional, Set, Union, Tuple

import pandas as pd
import pandas_ta as ta
from lightweight_charts import Chart
from lightweight_charts.abstract import AbstractChart, Candlestick, Histogram, Line
from PIL import Image

from ibkr_bot.indicators.ttm_squeeze import ttm_squeeze

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("log.txt", mode="w"),
        logging.StreamHandler(),
    ],
)

min_width: Final = 90
offset: Final = 50 * 24 * int(60 / 5)
display_bars: Final = 120
history_bars: Final = display_bars + 22

color_up = "rgba(34,158,131,0.8)"
color_down = "rgba(201,96,100,0.8)"

color_macd = "rgba(255,189,64,1.0)"
color_signal = "rgba(0,118,209,1.0)"

color_grow_above = color_up
color_fall_above = "rgba(104,113,105,1.0)"
color_grow_below = color_down
color_fall_below = "rgba(251,116,110,1.0)"


def macd_color(hist, hist_1):
    if hist >= 0:
        if hist_1 < hist:
            return color_grow_above
        else:
            return color_fall_above
    else:
        if hist_1 < hist:
            return color_grow_below
        else:
            return color_fall_below


datafile = r"ibkr_bot/broker/data/ES_cc.csv"
OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def load_data(filename: str) -> pd.DataFrame:
    cols = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(filename, header=0, names=cols, dtype_backend="pyarrow")
    df["Date_Time"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S"
    )
    df.set_index("Date_Time", inplace=True)
    df.index.rename("Date", inplace=True)
    df.sort_index(inplace=True)
    del df["Date"]
    del df["Time"]
    return df


class MyChart:

    def __init__(self, df: pd.DataFrame, timeframe="15min") -> None:
        self.watermark = timeframe
        self.timeframe = (
            timeframe.lower()
            .replace(" ", "")
            .replace("day", "d")
            .replace("hour", "h")
            .replace("hours", "h")
        )

        self.sma_length = 10

        self.macd_fast = 10
        self.macd_slow = 20
        self.macd_signal = 9

        self.rsi_length = 10

        self.ttm_window_size = 10

        self.bars = df.resample(self.timeframe).agg(OHLC)
        self.bars.dropna(inplace=True)
        # Calc Interval
        self.interval = self.bars.index[1] - self.bars.index[0]
        # Create Chart & Sup-plots
        self.main = self._candle(
            Chart(inner_height=0.50, inner_width=1.0, position="top", width=600, height=600)
        )
        
        # self.subchart1 = self.main.create_subchart(height=0.1, width=1.0, position="bottom", sync=True)
        # self.volume_chart = self._vol(self.subchart1)
        
        self.subchart2 = self.main.create_subchart(height=0.15, width=1.0, position="bottom", sync=True)
        self.macd_chart  = self._macd(
            self.subchart2,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        
        self.subchart3 = self.main.create_subchart(height=0.15, width=1.0, position="bottom", sync=True)
        self.rsi_chart = self._rsi(self.subchart3, length=self.rsi_length)

        self.subchart4 = self.main.create_subchart(height=0.15, width=1.0, position="bottom", sync=True)
        self.ttm_chart = self._ttm(self.subchart4, window_size=self.ttm_window_size)

        self.bars.drop(self.bars.index[0:-history_bars],inplace=True)


    def _candle(self, chart: AbstractChart) -> AbstractChart:
        bars = self.bars.drop(columns=["Volume"])
        chart.price_line(label_visible=True, line_visible=True)
        chart.layout(font_size=10)
        chart.price_scale(minimum_width=min_width)
        chart.time_scale(right_offset=10, visible=False)
        chart.watermark(self.watermark, color="rgba(180, 180, 240, 0.2)")
        chart.set(bars[-display_bars:])
        return chart

    def _vol(self, chart: AbstractChart) -> tuple[AbstractChart, AbstractChart]:
        chart.layout(font_size=10)
        chart.price_scale(auto_scale=True, minimum_width=min_width)
        chart.time_scale(right_offset=10, visible=False)
        vol_hist = chart.create_histogram(
            name="Volume",
            color="color",
            price_line=False,
            price_label=False,
            scale_margin_top=0.05,
            scale_margin_bottom=0.0,
        )
        # chart.histogram = [vol_hist]
        sma_line = chart.create_line(
            name=f"SMA_{self.sma_length}",
            color=color_signal,
            style="solid",
            width=2,
            price_line=False,
            price_label=False,
        )
        
        volume = self.bars.copy().drop(columns=["Open", "High", "Low", "Close"])
        volume["color"] = color_down
        volume.loc[self.bars["Close"] > self.bars["Open"], "color"] = color_up
        vol_hist.set(volume[["Volume","color"]][-display_bars:])

        sma = pd.DataFrame(ta.sma(volume["Volume"], length=self.sma_length))
        sma_line.set(sma[-display_bars:])
        return {'hist':vol_hist, 'sma':sma_line}


    def _rsi(self, chart: AbstractChart, length=10) -> AbstractChart:
        rsi = pd.DataFrame(ta.rsi(self.bars["Close"], length=length))

        chart.layout(font_size=10)
        chart.price_scale(auto_scale=True, minimum_width=min_width)
        chart.time_scale(right_offset=10, visible=False)

        rsi_line: Line = chart.create_line(
            name=f"RSI_{length}",
            color=color_signal,
            style="solid",
            width=2,
            price_line=False,
            price_label=False,
        )
        rsi_line.set(rsi[-display_bars:])
        rsi70 = rsi_line.horizontal_line(
            price=70,
            color="gray",
            width=1,
            style="large_dashed",
            axis_label_visible=False,
        )
        rsi30 = rsi_line.horizontal_line(
            price=30,
            color="gray",
            width=1,
            style="large_dashed",
            axis_label_visible=False,
        )
        return rsi_line

    def _macd(self, chart: AbstractChart, fast=10, slow=20, signal=9) -> tuple[AbstractChart, AbstractChart, AbstractChart]:
        as_mode = True
        _asmode = "AS" if as_mode else ""
        _props = f"_{fast}_{slow}_{signal}"
        macd = ta.macd(self.bars["Close"], fast, slow, signal, asmode=as_mode)
        macd[f"MACD{_asmode}h{_props}_1"] = macd[f"MACD{_asmode}h{_props}"].shift(1)
        macd["color"] = macd.apply(
            lambda x: macd_color(
                x[f"MACD{_asmode}h{_props}"], x[f"MACD{_asmode}h{_props}_1"]
            ),
            axis=1,
        )

        chart.layout(font_size=10)
        chart.price_scale(auto_scale=True, minimum_width=min_width)
        chart.time_scale(right_offset=10, visible=False)
        macd_hist = chart.create_histogram(
            name=f"MACD{_asmode}h{_props}",
            color="color",
            price_line=False,
            price_label=False,
            scale_margin_top=0.05,
            scale_margin_bottom=0.0,
        )
        macd_hist.set(macd[-display_bars:])

        macd = macd.drop(columns=["color"])
        macd_line = chart.create_line(
            name=f"MACD{_asmode}{_props}",
            color=color_macd,
            style="solid",
            width=2,
            price_line=False,
            price_label=False,
        )
        macd_line.set(macd[-display_bars:])

        macd_sig_line = chart.create_line(
            name=f"MACD{_asmode}s{_props}",
            color=color_signal,
            style="solid",
            width=2,
            price_line=False,
            price_label=False,
        )
        macd_sig_line.set(macd[-display_bars:])
        return {'hist':macd_hist, 'macd':macd_line, 'sig':macd_sig_line}

    def _ttm(self, chart: AbstractChart, window_size=10) -> tuple[AbstractChart, AbstractChart]:
        ttm = ttm_squeeze(self.bars, window_size)
        ttm["color"] = ttm["ttm_mom_color"]

        chart.layout(font_size=10)
        chart.price_scale(auto_scale=True, minimum_width=min_width)
        chart.time_scale(right_offset=10, visible=False)
        ttm_hist = chart.create_histogram(
            name="ttm_mom",
            color="color",
            price_line=False,
            price_label=False,
            scale_margin_top=0.05,
            scale_margin_bottom=0.05,
        )
        ttm_hist.set(ttm[-display_bars:])

        sub_chart = chart.create_subchart(
            height=0.2, width=1.0, position="bottom", sync=True
        )
        sub_chart.layout(font_size=10)
        sub_chart.price_scale(auto_scale=True, minimum_width=min_width)
        sub_chart.time_scale(right_offset=10, visible=False)
        ttm_sq_line = sub_chart.create_line(
            name="ttm_sq_line",
            color="black",
            style="solid",
            width=1,
            price_line=False,
            price_label=False,
        )
        ttm_sq_line.set(ttm[-display_bars:])
        for _, row in ttm[-display_bars:].iterrows():
            ttm_sq_line.marker(
                row.name, position="inside", color=row["ttm_sq_color"], shape="circle"
            )

        return {'hist':ttm_hist, 'dot':ttm_sq_line}

    async def update(self, ds: pd.Series):
        """
        Updates Main Chart & Sub Charts.\n
        """
        # Calc Update bar
        if ds.name < self.bars.index[-1]:
            self.bars.loc[self.bars.index[-1],"High"] = max(self.bars.loc[self.bars.index[-1],"High"], ds["High"])
            self.bars.loc[self.bars.index[-1],"Low"] = min(self.bars.loc[self.bars.index[-1],"Low"], ds["Low"])
            self.bars.loc[self.bars.index[-1],"Close"] = ds["Close"]
            self.bars.loc[self.bars.index[-1],"Volume"] += ds['Volume']
        else:
            ds.name = self.bars.index[-1] + self.interval
            self.bars = pd.concat([self.bars, ds.to_frame().T], axis=0)
        # Update Main Chart
        self.main.update(self.bars.iloc[-1].drop(labels=["Volume"]))
        #  Update Volume
        # volume = self.bars.copy()
        # volume["color"] = color_down
        # volume.loc[volume["Close"] > volume["Open"], "color"] = color_up
        # sma = pd.DataFrame(ta.sma(volume["Volume"], length=self.sma_length))

        # self.volume_chart['hist'].update(volume[["Volume","color"]].iloc[-1])
        # self.volume_chart['sma'].update(sma.iloc[-1])
        # Update MACD
        as_mode = True
        _asmode = "AS" if as_mode else ""
        _props = f"_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        macd = ta.macd(self.bars["Close"], self.macd_fast, self.macd_slow, self.macd_signal, asmode=as_mode)
        macd[f"MACD{_asmode}h{_props}_1"] = macd[f"MACD{_asmode}h{_props}"].shift(1)
        macd["color"] = macd.apply(
            lambda x: macd_color(
                x[f"MACD{_asmode}h{_props}"], x[f"MACD{_asmode}h{_props}_1"]
            ),
            axis=1,
        )
        self.macd_chart['hist'].update(macd.iloc[-1])
        macd["color"] = color_macd
        self.macd_chart['macd'].update(macd.iloc[-1])
        macd["color"] = color_signal
        self.macd_chart['sig'].update(macd.iloc[-1])
        # Update RSI
        rsi = pd.DataFrame(ta.rsi(self.bars["Close"], length=self.rsi_length))
        self.rsi_chart.update(rsi.iloc[-1])
        # Update TTM Charts
        ttm = ttm_squeeze(self.bars, self.ttm_window_size)
        ttm["color"] = ttm["ttm_mom_color"]
        self.ttm_chart['hist'].update(ttm.iloc[-1])

        self.ttm_chart['dot'].update(ttm.iloc[-1])
        self.ttm_chart['dot'].marker(ttm.iloc[-1].name, position="inside", color=ttm.iloc[-1]["ttm_sq_color"], shape="circle")

        if len(self.bars) > history_bars:
            self.bars.drop(self.bars.index[0],inplace=True)
                    
        


    @staticmethod
    def concat_images(*images):
        """Generate composite of all supplied images."""
        # Get the widest width.
        width = max(image.width for image in images)
        # Add up all the heights.
        height = sum(image.height for image in images)
        composite = Image.new('RGB', (width, height))
        # Paste each image below the one before it.
        y = 0
        for image in images:
            composite.paste(image, (0, y))
            y += image.height
        return composite
    
    def screenshot(self):
        images = (
                Image.open(io.BytesIO(self.main.screenshot())),
                Image.open(io.BytesIO(self.volume.screenshot())),
                Image.open(io.BytesIO(self.macd_chart.screenshot())),
                Image.open(io.BytesIO(self.rsi_chart.screenshot())),
                Image.open(io.BytesIO(self.ttm_chart.screenshot())),
                Image.open(io.BytesIO(self.ttm_line.screenshot())),
            )
        img = self.concat_images(*images)
        return img
        # pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)


async def do_updates(charts, bars: pd.DataFrame):
    for i, bar in bars.iterrows():
        for chart in charts:
            if not chart.main.win.loaded:
                continue
            if not chart.main.is_alive:
                return
            await chart.update(bar)
            await asyncio.sleep(0.0)


async def main():

    # Get Data
    df = load_data(filename=datafile)
    hist = df[:offset]
    bars = df[offset:]
    charts: List[MyChart] = [
        MyChart(hist, timeframe="15 min"),
    ]
    await asyncio.gather(
        charts[0].main.show_async(block=True), 
        do_updates(charts, bars),
    )


if __name__ == "__main__":
    # main()
    asyncio.run(main())
