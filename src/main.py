from typing import Tuple, List, Dict
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint 
import multiprocessing

import databento
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from common import DIR_DB, DIR_OUT, DIR_TMP
from utils import create_video, display_bytesize

plt.rcParams['path.simplify'] = False
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['lines.antialiased'] = False
plt.rcParams['patch.antialiased'] = False
plt.rcParams['text.antialiased'] = False

def to_pickle_filename(zst_filename):
    return DIR_TMP / f"{".".join(zst_filename.split(".")[:2])}.pkl"

def dataset_validate_file(zst_file):
        zst_filename = zst_file.name
        dbn = databento.DBNStore.from_file(zst_file)

        df_pickle_file = to_pickle_filename(zst_filename)
        if df_pickle_file.exists():
            print(f"reading {df_pickle_file}")
            df = pd.read_pickle(df_pickle_file)
        else:
            print(f"generating {df_pickle_file}")
            df = dbn.to_df()
            df.to_pickle(df_pickle_file)
        size = df_pickle_file.stat().st_size
        dbn_symbols = set(dbn.mappings.keys())
        return zst_filename, df, dbn_symbols, size

def dataset_validate(dataset_name, n=-1) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    DIR_TMP.mkdir(exist_ok=True)
    dir_dataset = DIR_DB / dataset_name

    symbols = set()
    dfs = dict()
    all_zst_files = sorted(list(dir_dataset.glob("*.zst")))
    if n > 0:
        all_zst_files = all_zst_files[:n]
    sizes = []

    nthreads = 64
    results = []
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        print(f"Validiating {len(all_zst_files)} files in {nthreads} threads")
        futures = [
            executor.submit(dataset_validate_file, zst_file)
            for zst_file in all_zst_files
        ]
        i = 1
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Validated [{i}/{len(futures)}] {result[0]} {display_bytesize(result[-1])}")
            i += 1

    for zst_filename, df, dbn_symbols, size in results:
        dfs[zst_filename] = df
        sizes.append(size)
        if len(symbols) == 0:
            symbols = symbols.union(dbn_symbols)
        else:
            symbols = symbols.intersection(dbn_symbols)
    for zst_filename, df, dbn_symbols, size in results:
        if len(symbols - dbn_symbols) > 0:
            raise Exception(f"Missing some symbols in {zst_filename}")

    print(f"r/w total disk: {display_bytesize(sum(sizes))}")
    return sorted(list(symbols)), dfs 


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
    x = mdates.date2num(volume.index.to_pydatetime())
    y = volume.values
    # y = volume.rolling(3, center=True, closed="both").mean()
    # y = (volume / volume.sum()).values
    return zst_filename, symbol, df_symbol, x, y


def generate_volume_average_plot(outname, title, x, y, ylim):
    out_file = DIR_OUT / f"{outname}.png"
    if out_file.exists():
        return out_file

    dpi = 300 
    fig, ax = plt.subplots(figsize=(21, 6), dpi=dpi)
    for spine in ax.spines.values():
        spine.set_antialiased(False)

    bar_width = 0.9 / len(x)
    print("bar_width:", bar_width)

    ax.bar(x, y, width=bar_width, color='blue', edgecolor='none')

    ax.set_facecolor("black")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.set_ylim(top=ylim)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Volume")
    ax.grid(False)
    plt.tight_layout()

    plt.savefig(out_file, dpi=dpi)#, bbox_inches='tight')
    plt.close(fig)

    return out_file


def generate_plot(zst_filename, symbol, df_symbol, x, y, ylim):
    out_file = DIR_OUT / f"{zst_filename.split('.')[0]}_{symbol}_trade_volume.png"

    if out_file.exists():
        return out_file

    dpi = 300 
    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)
    bar_width = 0.9 / len(x)

    candle_size = (df_symbol['close'] - df_symbol['open']).abs()
    intensity = 0.1 + 0.9 * (candle_size / candle_size.max())

    for ts, open_, close_, alpha in zip(df_symbol.index, df_symbol["open"], df_symbol["close"], intensity):
        if close_ >= open_:
            color = (0, alpha, 0)
        else:
            color = (alpha, 0, 0)
        ax.axvspan(ts, ts + pd.Timedelta(minutes=1), color=color, alpha=1)
        
    ax.bar(x, y, width=bar_width, color='blue', edgecolor='none')

    ax.set_facecolor("black")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.set_ylim(top=ylim)
    ax.set_title(f"Normalized 1-Minute Trade Volume with Candle-Size Backgrounds [{df_symbol.index.date[0]}]")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Volume")
    ax.grid(False)
    plt.tight_layout()

    plt.savefig(out_file, dpi=dpi)
    plt.close(fig)

    return out_file


def dataset_process(dataset_name):
    symbols, dfs = dataset_validate(dataset_name, n=-1)
    print(f"Dataset: {dataset_name}")
    print(f"  from: {list(dfs.values())[0].index[0]}\n    to: {list(dfs.values())[-1].index[-1]}")
    print(f"symbols[{len(symbols)}] List[str] = {display_bytesize(sys.getsizeof(symbols))}")
    print(f"dfs[{len(dfs)}] Dict[str, pd.DataFrame] = {display_bytesize(sum(map(lambda kv: sys.getsizeof(kv[0]) + kv[1].memory_usage().sum(), dfs.items())))}")

    print("===========")
    symbol = symbols[0]
    print(f"Symbol: {symbol}")

    multiprocessing.set_start_method("spawn", force=True)

    results = []

    nthreads = 64
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        print(f"Processing {len(dfs)} files in {nthreads} threads")
        futures = [
            executor.submit(process_file, zst_filename, df, symbol)
            for zst_filename, df in dfs.items()
        ]
        i = 1
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Processed [{i}/{len(futures)}]")
            i += 1


    # normalize
    max_y = 1
    for _, _, _, _, y in results:
        max_y = max(max(y), max_y)

    #
    avg_across_dataset = {}
    for _, _, _, x, y in results:
        date = str(mdates.num2date(x[0]).date())
        avg_across_dataset[date] = y
    df_avg_across_dataset = pd.DataFrame.from_dict(avg_across_dataset, orient="index")

    avg_across_dataset_x = results[0][3]
    avg_across_dataset_y = df_avg_across_dataset.sum() / len(df_avg_across_dataset)
    (DIR_OUT / "avg.txt").write_text("\n".join((f"{mdates.num2date(x)} {y}" for x, y in zip(avg_across_dataset_x, avg_across_dataset_y))))

    ncores = multiprocessing.cpu_count() - 1
    plot_paths = []
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        print(f"Generating plots on {ncores} cores")
        futures = [
            executor.submit(generate_plot, zst_filename, sym, df_symbol, x, y, max_y)
            for zst_filename, sym, df_symbol, x, y in results
        ]
        futures.append(executor.submit(
            generate_volume_average_plot, "volume_avg", "Volume average across dataset", avg_across_dataset_x, avg_across_dataset_y, max(avg_across_dataset_y)
        ))

        i = 1
        for future in as_completed(futures):
            out_file = future.result()
            plot_paths.append(out_file)
            print(f"Plot [{i}/{len(futures)}] saved to {out_file}")
            i += 1

    create_video(sorted(plot_paths), DIR_OUT, fps=10)
    print("DONE")
    
if __name__ == "__main__":
    dataset_process("XNAS-itch")

