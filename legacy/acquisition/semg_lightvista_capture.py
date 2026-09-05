import time
import csv
import os
import sys
from datetime import datetime, timedelta
from device_init import device_init
from recv_data import recv_data
from config import GLOBAL_SAMPLING_INTERVAL_MS
from config import GLOBAL_SAMPLING_RATE_HZ

STOP_SIGNAL_FILENAME = "stop_signal.tmp"

FP_CHANNEL_NUMBERS = {41,42,43,44,45,46,47,48,51,52,53,54,55,56,57,58}

class ConsoleLogger:
    def __init__(self, filepath, original_stdout):
        self.terminal = original_stdout
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def main():
    lightvista_exe = os.environ.get("LIGHTVISTA_EXE", r"D:\LightVista\software\LightVista.exe")
    ch_num, ch_no, u = device_init(lightvista_exe)
    print(f"已开启通道: {ch_no}")

    start_time = datetime.now()
    folder_name = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('sEMG_data', folder_name)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'sEMG_{folder_name}.csv')

    original_stdout = sys.stdout
    log_path = os.path.join(output_dir, f'console_log_{folder_name}.log')
    logger = ConsoleLogger(log_path, original_stdout)
    sys.stdout = logger

    print(f"sEMG采集启动：{csv_path}")
    print("检测到停止信号文件后将退出。")

    interval = 1.0 / GLOBAL_SAMPLING_RATE_HZ
    next_time = time.perf_counter()
    idx = 0

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Timestamp'] + [f'CH{num}' for num in ch_no]
            writer.writerow(header)

            while True:
                if os.path.exists(STOP_SIGNAL_FILENAME):
                    print("\n检测到停止信号文件，退出采集。")
                    break

                next_time += interval
                data_packets = recv_data(u, ch_no)
                if any(data_packets):
                    ts = start_time + timedelta(milliseconds=idx * GLOBAL_SAMPLING_INTERVAL_MS)
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    row = [ts_str] + [str(p) if p else '' for p in data_packets]
                    writer.writerow(row)
                    print(f"[{ts_str}] 数据采集中...", end='\r')
                    idx += 1

                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n日志保存在: {log_path}")
        print("sEMG采集结束。")

if __name__ == "__main__":
    main()
