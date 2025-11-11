from common import DIR_THIS, DIR_DB, DIR_OUT, DIR_TMP
from typing import Tuple, List, Dict, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import cProfile, pstats, io


import databento as db
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

plt.rcParams['path.simplify'] = False
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['lines.antialiased'] = False
plt.rcParams['patch.antialiased'] = False
plt.rcParams['text.antialiased'] = False

def validate(dataset_name) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    def to_pickle_filename(zst_filename):
        return f"{".".join(zst_filename.split(".")[:2])}.pkl"
    dir_dataset = DIR_DB / dataset_name

    symbols = set()
    dfs: Dict[str, pd.DataFrame] = dict()
    dbns = dict()
    for zst_file in dir_dataset.glob("*.zst"):
        zst_filename = zst_file.name
        print(zst_filename)
        dbn = db.DBNStore.from_file(zst_file)
        dbns[zst_filename] = dbn

        df_pickle_file = DIR_TMP / to_pickle_filename(zst_filename)
        if df_pickle_file.exists():
            dfs[zst_filename] = pd.read_pickle(df_pickle_file)
        else:
            DIR_TMP.mkdir(exist_ok=True)
            df = dbn.to_df()
            dfs[zst_filename] = df
            df.to_pickle(df_pickle_file)
        # print(dbn)
        dbn_symbols = set(dbn.mappings.keys())
        # print(len(dbn_symbols))
        if len(symbols) == 0:
            symbols = symbols.union(dbn_symbols)
        else:
            symbols = symbols.intersection(dbn_symbols)
    for zst_filename, dbn in dbns.items():
        dbn_symbols = set(dbn.mappings.keys())
        remaining = symbols - dbn_symbols
        if len(remaining) > 0:
            raise Exception(f"Missing some symbols in {zst_filename}")
    return sorted(list(symbols)), dfs 

def process_file(zst_filename, df, symbol):
    # pr = cProfile.Profile()
    # pr.enable()

    df_symbol = df[df["symbol"] == symbol]
    if df_symbol.empty:
        print(f"⚠️ No data for {symbol} in {zst_filename}")
        return None

    day = df_symbol.index[0].normalize()
    ts_index = pd.date_range(
        start=day,
        end=day + pd.Timedelta(days=1),
        freq="1min",
        inclusive="left"
    )

    volume = df_symbol["volume"].resample("1min").sum().fillna(0)
    volume = volume.reindex(ts_index, fill_value=0)
    volume_normalized = volume / volume.sum()

    dpi = 300 
    fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
    for spine in ax.spines.values():
        spine.set_antialiased(False)
    # bar_width = mdates.date2num(pd.Timestamp('2025-01-01 00:01')) - mdates.date2num(pd.Timestamp('2025-01-01 00:00'))

    candle_size = (df_symbol['close'] - df_symbol['open']).abs()
    intensity = 0.1 + 0.9 * (candle_size / candle_size.max())

    for ts, open_, close_, alpha in zip(df_symbol.index, df_symbol["open"], df_symbol["close"], intensity):
        if close_ >= open_:
            color = (0, alpha, 0)  # scale green channel
        else:
            color = (alpha, 0, 0)  # scale red channel
        ax.axvspan(ts, ts + pd.Timedelta(minutes=1), color=color, alpha=1)


    x = mdates.date2num(volume_normalized.index.to_pydatetime())
    y = volume_normalized.values

    fig_width_in_pixels = fig.get_size_inches()[0] * fig.dpi
    x_min = x[0]
    x_max = x[-1]
    data_width = x_max - x_min
    width_per_pixel = data_width / fig_width_in_pixels

    ax.bar(x, y, width=width_per_pixel, color='blue', edgecolor='none')

    ax.xaxis_date()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.set_title("Normalized 1-Minute Trade Volume with Candle-Size Backgrounds")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Volume")
    ax.grid(False)
    plt.tight_layout()

    out_file = DIR_OUT / f"{zst_filename.split('.')[0]}_{symbol}_trade_volume.png"
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    # ps.print_stats(10)  # top 10 slowest
    # print(s.getvalue())
    return out_file
 




def parallel_threads(dataset_name):
    symbols, dfs = validate(dataset_name)
    print(len(symbols))
    symbol = symbols[0]
    print(f"Symbol {symbol}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_file, zst_filename, df, symbol)
            for zst_filename, df in dfs.items()
        ]

        for future in as_completed(futures):
            try:
                out_file = future.result()
                print(f"Plot saved to {out_file}")
            except Exception as e:
                print(f"❌ Error in thread: {e}")


def parallel_processes(dataset_name):
    symbols, dfs = validate(dataset_name)
    print(len(symbols))
    symbol = symbols[0]
    print(f"Symbol {symbol}")

    multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        futures = [
            executor.submit(process_file, zst_filename, df, symbol)
            for zst_filename, df in dfs.items()
        ]

        for future in as_completed(futures):
            try:
                out_file = future.result()
                print(f"✅ Plot saved to {out_file}")
            except Exception as e:
                print(f"❌ Error in process: {e}")


def foo(dataset_name):
    symbols, dfs = validate(dataset_name)
    print(len(symbols))
    symbol = symbols[0]
    print(f"Symbol {symbol}")
    for zst_filename, df in dfs.items():
        out_file = process_file(zst_filename, df, symbol)
        print(f"Plot saved to {out_file}")
        break
    
    
if __name__ == "__main__":
    parallel_processes("XNAS-itch")