using UnityEngine;

public class BuggyController : MonoBehaviour
{
    public WheelCollider frontWheel;
    public WheelCollider rearLeftWheel, rearRightWheel;
    public Transform frontMesh, rearLeftMesh, rearRightMesh;

    public Vector3 meshRotationOffset = new Vector3(0, 0, 90);

    public float maxMotorTorque = 15f;
    public float maxSwivelSpeed = 100f;

    private float currentSwivelAngle = 0f; // UNBOUNDED - no wraparound, avoids physics snap

    // For observations only - Sin/Cos are naturally continuous, no wrap needed
    public float CurrentSwivelAngleRadians => currentSwivelAngle * Mathf.Deg2Rad;

    public void ResetSwivel()
    {
        currentSwivelAngle = 0f;
        frontWheel.steerAngle = 0f;
    }

    public void ApplyControl(float steerInput, float throttleInput)
    {
        steerInput = Mathf.Clamp(steerInput, -1f, 1f);
        throttleInput = Mathf.Clamp(throttleInput, -1f, 1f);

        // accumulate freely - no clamp, no wrap, avoids the 360->0 snap
        currentSwivelAngle += steerInput * maxSwivelSpeed * Time.fixedDeltaTime;
        currentSwivelAngle = Mathf.Clamp(currentSwivelAngle, -40f, 40f);
        frontWheel.steerAngle = currentSwivelAngle;

        frontWheel.motorTorque = throttleInput * maxMotorTorque;

        UpdateWheelMesh(frontWheel, frontMesh);
        UpdateWheelMesh(rearLeftWheel, rearLeftMesh);
        UpdateWheelMesh(rearRightWheel, rearRightMesh);
    }

    void UpdateWheelMesh(WheelCollider collider, Transform mesh)
    {
        collider.GetWorldPose(out Vector3 pos, out Quaternion rot);
        mesh.position = pos;
        mesh.rotation = rot * Quaternion.Euler(meshRotationOffset);
    }
}