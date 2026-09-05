import serial
import csv
from datetime import datetime, timedelta
import time
import threading
import os
import sys
from config import GLOBAL_SAMPLING_INTERVAL_MS, GLOBAL_SAMPLING_RATE_HZ

STOP_SIGNAL_FILENAME = "stop_signal.tmp"

# === 日志类 ===
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

# === 数据缓冲 ===
class LatestData:
    def __init__(self):
        self.value = None
        self.lock = threading.Lock()

    def set(self, value):
        with self.lock:
            self.value = value

    def get(self):
        with self.lock:
            return self.value

# === 串口读取线程 ===
def serial_reader_thread(ser, latest_data, stop_event):
    print("读取线程已启动。")
    while not stop_event.is_set():
        try:
            line_bytes = ser.readline()
            if line_bytes:
                data_str = line_bytes.decode('utf-8').strip()
                latest_data.set(data_str)
        except Exception:
            pass
    print("读取线程已停止。")

# === 主程序 ===
def main():
    SERIAL_PORT = os.environ.get("IMU_SERIAL_PORT", "COM10")
    BAUDRATE = 115200

    timestamp_file = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('IMU_data', timestamp_file)
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = os.path.join(save_dir, f'IMU_{timestamp_file}.csv')

    original_stdout = sys.stdout
    log_path = os.path.join(save_dir, f'console_log_{timestamp_file}.log')
    logger = ConsoleLogger(log_path, original_stdout)
    sys.stdout = logger

    print(f"IMU采集已启动：{csv_filename}")
    print("检测到停止信号文件后将自动退出。")

    ser = serial.Serial(SERIAL_PORT, BAUDRATE)
    latest_data = LatestData()
    stop_event = threading.Event()

    thread = threading.Thread(target=serial_reader_thread, args=(ser, latest_data, stop_event))
    thread.daemon = True
    thread.start()

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Data'])

            interval = 1.0 / GLOBAL_SAMPLING_RATE_HZ
            start_datetime = datetime.now()
            next_time = time.perf_counter()
            idx = 0

            while True:
                if os.path.exists(STOP_SIGNAL_FILENAME):
                    print("\n检测到停止信号文件，退出采集。")
                    break

                next_time += interval
                data = latest_data.get()

                if data is not None:
                    ts = start_datetime + timedelta(milliseconds=idx * GLOBAL_SAMPLING_INTERVAL_MS)
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    writer.writerow([ts_str, data])
                    print(f"[{ts_str}] {data}", end='\r')
                    idx += 1

                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=1)
        if ser.is_open:
            ser.close()
        sys.stdout = original_stdout
        logger.close()
        print(f"\n日志保存在: {log_path}")
        print("IMU采集结束。")

if __name__ == "__main__":
    main()
