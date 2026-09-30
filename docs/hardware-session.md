# AI PC hardware session

Everything in this document needs the Snapdragon AI PC. Everything else in the mentor
review (code, README, AI Hub cloud jobs, dashboard) is done on other machines.

Run the steps in order. Each one writes a file into `benchmarks/` or `docs/evidence/`,
and those files are the only thing this session commits. Do not change code here: if a
step fails, stop and send the full console output instead of working around it.

| Step | Needs | Output | Mentor item |
|---|---|---|---|
| 1. Setup | AI PC | a working environment | – |
| 2. Session record | AI PC | `benchmarks/session.json` | 2, 6 |
| 3. Path A latency on the NPU | AI PC | `benchmarks/path_a_local.json` | 2, 6 |
| 4. Path A execution-provider split | AI PC | `benchmarks/path_a_profile.json` | 2 |
| 5. Serial port and board evidence | AI PC | `docs/evidence/serial_ports.txt`, board photo | 6 |
| 6. Serial run with the buggy | AI PC + Arduino | `benchmarks/path_a_serial.json` | 6 |
| 7. Commit and push | AI PC | branch `hw/ai-pc-session` | – |
| 8. Later sessions | AI PC | added when those branches are pushed | 4, 5 |

---

## 1. Setup

Use a **native ARM64** Python 3.12 from python.org (the "Windows installer (ARM64)"). An
x64 Python runs under emulation and cannot load the ARM64 QNN libraries.

```powershell
git clone https://github.com/adt-kmr/DragVerse.git   # skip if already cloned
cd DragVerse
git fetch origin
git switch -c hw/ai-pc-session origin/fix/path-a-reproducibility

py -V:3.12-arm64 -m venv .venv
.venv\Scripts\activate
python -c "import platform; print(platform.python_version(), platform.machine())"
pip install -r requirements-aipc.txt
```

Expected: the `platform` line prints `3.12.x ARM64`. Only `requirements-aipc.txt` is
installed here. `requirements.txt` and `requirements-npu.txt` are not needed, and
`requirements-npu.txt` cannot install on Windows ARM64 (no `torch` wheel).

Check that the NPU is visible:

```powershell
python twin/inference.py --no-serial --iterations 5
```

Expected: `Found 1 QNN EP device(s)`, `Session created on [...]`, then a line starting
`5 inferences: p50` and a `Last output` line. If it prints `QNN EP device not found`, check that the
Python is ARM64 and that Device Manager shows the NPU under "Neural processors".

## 2. Session record

Records exactly which machine and software produced the numbers. Run from the repository
root with the virtual environment active:

```powershell
New-Item -ItemType Directory -Force benchmarks | Out-Null
$os = Get-CimInstance Win32_OperatingSystem
$info = [ordered]@{
  soc             = (Get-CimInstance Win32_Processor).Name
  npu             = (Get-PnpDevice -Class ComputeAccelerator -ErrorAction SilentlyContinue |
                     Select-Object -First 1 -ExpandProperty FriendlyName)
  npu_driver      = (Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
                     Where-Object { $_.DeviceClass -eq "COMPUTEACCELERATOR" } |
                     Select-Object -First 1 -ExpandProperty DriverVersion)
  windows         = "$($os.Caption) build $($os.BuildNumber)"
  python          = (python -c "import platform; print(platform.python_version(), platform.machine())")
  onnxruntime     = (python -c "import onnxruntime; print(onnxruntime.__version__)")
  onnxruntime_qnn = (python -c "import onnxruntime_qnn; print(onnxruntime_qnn.__version__)")
  git_commit      = (git rev-parse HEAD)
  recorded_utc    = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
# WriteAllText writes UTF-8 without a byte-order mark.
[IO.File]::WriteAllText("$PWD\benchmarks\session.json", ($info | ConvertTo-Json))
Get-Content benchmarks\session.json
```

Check `soc` by eye. The README currently says "Snapdragon X Elite"; if this machine is an
X2 Elite (or anything else), say so when you send the results so the README can be
corrected. If `npu` or `npu_driver` is empty, open Device Manager, expand "Neural
processors", and add the device name and driver version to the file by hand.

## 3. Path A latency on the NPU

```powershell
python twin/inference.py --no-serial --iterations 1000 --log benchmarks/path_a_local.json
```

Expected: `1000 inferences: p50 ... ms, p95 ... ms, max ... ms` and `wrote
benchmarks/path_a_local.json`. The file records latency percentiles, the SHA-256 of
`Buggy.onnx`, the execution providers, and the ONNX Runtime and QNN versions. This run has
no profiling, so its latency is the number the README will quote.

Close other heavy applications first and keep the laptop on mains power.

## 4. Path A execution-provider split

```powershell
python twin/inference.py --no-serial --iterations 200 --profile --log benchmarks/path_a_profile.json
Remove-Item onnxruntime_profile_*.json
```

Expected: the usual output plus `Kernels by provider: {...}` and `CPU fallback nodes:
[...]`. This shows which parts of the graph ran on the NPU and which fell back to the CPU.
ML-Agents' sampling nodes (`RandomNormalLike`, `Multinomial`) are expected on the CPU.
Profiling slows every inference, so this file is for the split only, not for latency.
The raw ONNX Runtime profile is deleted because the summary is already in the log.

## 5. Serial port and board evidence

Mentor item 6 asks which COM port the buggy used and which Arduino board was on it.
Windows remembers every serial device that has been plugged in, including disconnected
ones:

```powershell
New-Item -ItemType Directory -Force docs\evidence | Out-Null
Get-PnpDevice -Class Ports |
  Select-Object Status, FriendlyName, InstanceId |
  Format-Table -AutoSize -Wrap |
  Out-File -Encoding utf8 docs\evidence\serial_ports.txt
Get-Content docs\evidence\serial_ports.txt
```

`Status` is `OK` for connected devices and `Unknown` for ones seen before. The COM number
is in `FriendlyName`. The USB vendor and product IDs in `InstanceId` identify the board:

| `InstanceId` contains | Board |
|---|---|
| `VID_2341` or `VID_2A03` | Genuine Arduino (the PID tells the model: `PID_0043` Uno R3) |
| `VID_1A86&PID_7523` | CH340 USB-serial chip, common on Uno and Nano clones |
| `VID_0403&PID_6001` | FTDI FT232R, older Nano and clones |
| `VID_10C4&PID_EA60` | Silicon Labs CP210x |

Also add, if available:

- a photo of the buggy showing the board, as `docs/evidence/img/buggy_board.jpg`
- the exact script used at the event, as `docs/evidence/event_inference.py`, unmodified

In your message, state which entry was the buggy's and how you know. If it cannot be
determined, say so; the README will then mark the board and port as unconfirmed.

## 6. Serial run with the buggy (only if the buggy and its Arduino are available)

1. Flash `twin/buggy_motor_controller.ino` with the Arduino IDE (115200 baud). The Serial
   Monitor shows `READY` after about two seconds. Close the Serial Monitor afterwards, or
   the port stays busy.
2. **Lift the drive wheels off the ground.**
3. Run, with the COM port from step 5. The script waits for the sketch's `READY` line
   (up to 5 s) before sending commands, and warns if it never arrives:

```powershell
python twin/inference.py --port COM5 --iterations 200 --log benchmarks/path_a_serial.json
```

Expected: the steering servo moves and the ESC responds for about four seconds (200
commands at the default 50 Hz), then both return to neutral. The script always sends a
neutral command at the end, and the Arduino also goes neutral on its own after 500 ms
without a command. Record a short phone video of the run for the evidence folder if
possible.

Skip this step if no Arduino is available and say so in your message.

## 7. Commit and push

```powershell
git add benchmarks/session.json benchmarks/path_a_local.json benchmarks/path_a_profile.json
git add docs/evidence
git add benchmarks/path_a_serial.json   # only if step 6 ran
git status --short
```

Check that `git status` lists only files under `benchmarks/` and `docs/evidence/`. Then:

```powershell
git commit -m "bench: Path A on the AI PC NPU (session, latency, EP split, serial evidence)"
git push -u origin hw/ai-pc-session
```

Send the console output of steps 1 to 6 along with the push, especially anything that
did not match the "Expected" lines.

## 8. Next session

Pull `main` first (`git switch main && git pull`), then set up as in step 1. Both steps
write a file into `benchmarks/`; step 2 may instead fail, and its console output is then
the evidence.

1. Accuracy of the NPU build on the NPU, against the original model on the CPU:

```powershell
python -m twin.compare_models --log benchmarks/path_a_accuracy_npu.json
```

   Expected: `"within_tolerance": true` and exit code 0. The largest steer and throttle
   differences are in the output; on the CPU they were 0.019 and 0.035.

2. Where the original float `Buggy.onnx` runs through the QNN execution provider:

```powershell
python twin/inference.py --model twin/Buggy.onnx --no-serial --profile --log benchmarks/path_a_profile_float.json
```

   The README says its nodes stayed on the CPU locally; this run is the evidence. If the
   session fails to open, save the full console output as
   `docs/evidence/path_a_float_qnn.txt` instead.

3. Steps 5 and 6, if the buggy's Arduino is now available.

Commit on a new branch and open a pull request:

```powershell
git switch -c hw/ai-pc-session-2
git add benchmarks docs/evidence
git status --short
git commit -m "bench: Path A accuracy on the NPU and float model placement"
git push -u origin hw/ai-pc-session-2
```
