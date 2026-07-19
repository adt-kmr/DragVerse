"""
Buggy RL Policy - NPU Inference via QNN Execution Provider
Runs the trained ML-Agents policy on Snapdragon NPU hardware.
"""

import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import numpy as np
import time
import serial

MODEL_PATH = "Buggy.onnx"  # update to your actual model path

# --- 1. Register QNN EP library ---
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

import serial

arduino = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)  # adjust port
time.sleep(2) 

def send_to_arduino(steer, throttle):
    arduino.write(f"{steer:.4f},{throttle:.4f}\n".encode())

# --- 2. Find QNN EP devices ---
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [d for d in all_ep_devices if d.ep_name == ep_registration_name]

if len(selected_ep_devices) == 0:
    raise RuntimeError("QNN EP device not found")

print(f"Found {len(selected_ep_devices)} QNN EP device(s)")

# --- 3. Configure QNN EP options (HTP = NPU backend) ---
ep_options = {
    "backend_path": qnn_ep.get_qnn_htp_path(),
    "htp_performance_mode": "burst",
}

session_options = ort.SessionOptions()
session_options.add_provider_for_devices(selected_ep_devices, ep_options)

# --- 4. Create the inference session ---
session = ort.InferenceSession(MODEL_PATH, sess_options=session_options)
print("Session created successfully on QNN EP.")

output_names = [o.name for o in session.get_outputs()]
det_idx = output_names.index("deterministic_continuous_actions")


def build_observation(
    swivel_angle_rad,      # current front wheel swivel angle, in radians
    prev_steer,             # last steer action sent (-1 to 1)
    prev_throttle,          # last throttle action sent (-1 to 1)
    local_target_pos,       # Vector3 (x,y,z) of current waypoint, in buggy's local space
    forward_dot,            # dot(forward, direction_to_target)
    right_dot,               # dot(right, direction_to_target)
    distance_to_target,     # scalar distance to current waypoint
    local_velocity,          # Vector3 (x,y,z) of buggy's velocity, in local space
    curriculum_progress      # currentTargetIndex / totalTargets, 0-1
):
    """
    Builds the 14-value observation vector, matching the EXACT order
    used in BuggyAgent.CollectObservations():
      [sin(swivel), cos(swivel), prevSteer, prevThrottle,
       localTarget.x, localTarget.y, localTarget.z,
       forwardDot, rightDot, distance,
       localVel.x, localVel.y, localVel.z,
       curriculumProgress]
    """
    obs = np.array([[
        np.sin(swivel_angle_rad),
        np.cos(swivel_angle_rad),
        prev_steer,
        prev_throttle,
        local_target_pos[0],
        local_target_pos[1],
        local_target_pos[2],
        forward_dot,
        right_dot,
        distance_to_target,
        local_velocity[0],
        local_velocity[1],
        local_velocity[2],
        curriculum_progress,
    ]], dtype=np.float32)
    return obs


def run_inference(obs):
    """
    Runs one inference step. Returns (steer, throttle).
    """
    action_masks = np.array([[1.0]], dtype=np.float32)  # dummy, no discrete actions used

    result = session.run(None, {
        "obs_0": obs,
        "action_masks": action_masks
    })

    steer, throttle = result[det_idx][0]
    return float(steer), float(throttle)


# --- Example usage / test loop ---
if __name__ == "__main__":
    # Replace this with your real sensor-reading function
    def get_live_observation():
        # TODO: replace with real sensor data from your hardware
        return build_observation(
            swivel_angle_rad=0.0,
            prev_steer=0.0,
            prev_throttle=0.0,
            local_target_pos=(0.0, 0.0, 5.0),
            forward_dot=1.0,
            right_dot=0.0,
            distance_to_target=5.0,
            local_velocity=(0.0, 0.0, 0.0),
            curriculum_progress=0.0,
        )

    prev_steer = 0.0
    prev_throttle = 0.0

    # Quick timing test - confirm inference speed is real-time capable
    num_test_iterations = 200
    start = time.time()

    for i in range(num_test_iterations):
        obs = get_live_observation()
        steer, throttle = run_inference(obs)
        prev_steer, prev_throttle = steer, throttle

        send_to_arduino(steer, throttle)  # send the output to Arduino

    elapsed = time.time() - start
    print(f"\n{num_test_iterations} inferences in {elapsed:.3f}s")
    print(f"Average: {elapsed/num_test_iterations*1000:.2f} ms/inference")
    print(f"Max control frequency: {num_test_iterations/elapsed:.1f} Hz")

    print(f"\nLast output -> Steer: {steer:.4f}, Throttle: {throttle:.4f}")

    # --- Clean up ---
    del session
    ort.unregister_execution_provider_library(ep_registration_name)