from common import DIR_THIS, DIR_DB, DIR_OUT, DIR_TMP
from utils import create_video
from typing import Tuple, List, Dict, cast
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import cProfile, pstats, io

import databento
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

def validate(dataset_name, n=-1) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    def to_pickle_filename(zst_filename):
        return f"{".".join(zst_filename.split(".")[:2])}.pkl"
    dir_dataset = DIR_DB / dataset_name

    symbols = set()
    dfs: Dict[str, pd.DataFrame] = dict()
    dbns = dict()
    all_zst_files = sorted(list(dir_dataset.glob("*.zst")))
    if n > 0:
        all_zst_files = all_zst_files[:n]
    sizes = []
    for i in range(len(all_zst_files)):
        zst_file = all_zst_files[i]
        zst_filename = zst_file.name
        print(f"validating [{i+1}/{len(all_zst_files)}]", dir_dataset / zst_filename)
        dbn = databento.DBNStore.from_file(zst_file)
        dbns[zst_filename] = dbn

        df_pickle_file = DIR_TMP / to_pickle_filename(zst_filename)
        if df_pickle_file.exists():
            print(f"  reading {df_pickle_file}: ", end="")
            dfs[zst_filename] = pd.read_pickle(df_pickle_file)
        else:
            print(f"  generating {df_pickle_file}: ", end="")
            DIR_TMP.mkdir(exist_ok=True)
            df = dbn.to_df()
            dfs[zst_filename] = df
            df.to_pickle(df_pickle_file)
        sizes.append(df_pickle_file.stat().st_size)
        print(display_bytesize(sizes[-1]))
        dbn_symbols = set(dbn.mappings.keys())
        if len(symbols) == 0:
            symbols = symbols.union(dbn_symbols)
        else:
            symbols = symbols.intersection(dbn_symbols)
    for zst_filename, dbn in dbns.items():
        dbn_symbols = set(dbn.mappings.keys())
        remaining = symbols - dbn_symbols
        if len(remaining) > 0:
            raise Exception(f"Missing some symbols in {zst_filename}")
    print(f"r/w total disk: {display_bytesize(sum(sizes))}")
    return sorted(list(symbols)), dfs 

def generate_plot(zst_filename, symbol, df_symbol, x, y, ylim):
    out_file = DIR_OUT / f"{zst_filename.split('.')[0]}_{symbol}_trade_volume.png"
    if out_file.exists():
        return out_file

    dpi = 300 
    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)
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
    fig_width_in_pixels = fig.get_size_inches()[0] * fig.dpi
    x_min = x[0]
    x_max = x[-1]
    data_width = x_max - x_min
    width_per_pixel = data_width / fig_width_in_pixels

    ax.bar(x, y, width=width_per_pixel, color='blue', edgecolor='none')
    ax.set_facecolor("black")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.set_ylim(top=ylim)
    ax.set_title(f"Normalized 1-Minute Trade Volume with Candle-Size Backgrounds [{df_symbol.index.date[0]}]")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Volume")
    ax.grid(False)
    plt.tight_layout()


    plt.savefig(out_file, dpi=dpi)#, bbox_inches='tight')
    plt.close(fig)

    return out_file

def process_file(zst_filename, df, symbol):
    df_symbol = df[df["symbol"] == symbol]
    if df_symbol.empty:
        print(f"⚠️ No data for {symbol} in {zst_filename}")

    day = df_symbol.index[0].normalize()
    ts_index = pd.date_range(
        start=day,
        end=day + pd.Timedelta(days=1),
        freq="1min",
        inclusive="left"
    )

    volume = df_symbol["volume"].resample("1min").sum().fillna(0)
    volume = volume.reindex(ts_index, fill_value=0)
    # volume_normalized = volume / volume.sum()
    x = mdates.date2num(volume.index.to_pydatetime())
    y = volume
    # x = mdates.date2num(volume_normalized.index.to_pydatetime())
    # y = volume_normalized.values
    return zst_filename, symbol, df_symbol, x, y



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

        results = []

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"❌ Error in thread: {e}")



def display_bytesize(n_bytes):
    units = ["B", "KB", "MB", "GB"]
    unit = 0
    out = float(n_bytes)
    while out >= 1024 and unit < len(units) - 1:
        out /= 1024
        unit += 1
    return f"{round(out, 1)} {units[unit]}"

    

def parallel_processes(dataset_name):
    symbols, dfs = validate(dataset_name, n=-1)
    print(f"Dataset: {dataset_name}")
    print(f"  from: {list(dfs.values())[0].index[0]}\n    to: {list(dfs.values())[-1].index[-1]}")
    print(f"symbols[{len(symbols)}] List[str] = {display_bytesize(sys.getsizeof(symbols))}")
    print(f"dfs[{len(dfs)}] Dict[str, pd.DataFrame] = {display_bytesize(sum(map(lambda kv: sys.getsizeof(kv[0]) + kv[1].memory_usage().sum(), dfs.items())))}")

    print("===========")
    symbol = symbols[0]
    print(f"Symbol: {symbol}")

    multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        # process file
        print("Processing files")
        futures = [
            executor.submit(process_file, zst_filename, df, symbol)
            for zst_filename, df in dfs.items()
        ]
        results = []
        i = 1
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Processed [{i}/{len(futures)}]")
            i += 1
        
        # normalize
        max_y = 1
        for _, _, _, _, y in results:
            max_y = max(y.max(), max_y)

        # plot
        futures = [
            executor.submit(generate_plot, zst_filename, sym, df_symbol, x, y.values, max_y)
            for zst_filename, sym, df_symbol, x, y in results
        ]
        plot_paths = []
        i = 1
        for future in as_completed(futures):
            out_file = future.result()
            plot_paths.append(out_file)
            print(f"Plot [{i}/{len(futures)}] saved to {out_file}")
            i += 1

    create_video(sorted(plot_paths), DIR_OUT, fps=10)
    print("DONE")



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
